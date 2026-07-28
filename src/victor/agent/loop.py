"""The ReAct loop.

Think, call a tool, read the result, repeat until there is an answer. The
interesting parts are the budgets: this agent runs against a free tier of
around 1,000 requests a day and 8,000 tokens a minute, so an unbounded loop is
not merely slow, it is the thing that ends the day's allowance.

Three limits, all reported rather than silently enforced:

* **steps** - how many think-act cycles before giving up;
* **tokens** - a ceiling on one task's total spend;
* **repetition** - identical tool calls in a row are refused, because a model
  stuck in a loop will happily burn the entire budget making the same mistake.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import VictorError
from ..providers import Router
from ..quota import QuotaLedger
from ..safety.killswitch import Aborted
from ..tools import ToolRegistry, ToolResult, build_registry, describe_environment
from ..tracing import Trace
from .llm import ChatClient, Reply, ToolCall
from .prompts import system_prompt

DEFAULT_MAX_STEPS = 8
DEFAULT_TOKEN_BUDGET = 20_000


class Outcome(StrEnum):
    ANSWERED = "answered"
    STEP_LIMIT = "step-limit"
    TOKEN_LIMIT = "token-limit"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class Step:
    """One think-act cycle."""

    index: int
    reply: Reply
    calls: tuple[tuple[ToolCall, ToolResult], ...] = ()

    @property
    def used_tools(self) -> bool:
        return bool(self.calls)


@dataclass
class AgentResult:
    """Everything one task produced."""

    task: str
    answer: str
    outcome: Outcome
    steps: list[Step] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.ANSWERED

    @property
    def tool_calls(self) -> list[tuple[ToolCall, ToolResult]]:
        return [pair for step in self.steps for pair in step.calls]

    def summary(self) -> str:
        tools = ", ".join(str(call) for call, _ in self.tool_calls) or "none"
        return (
            f"{self.outcome} in {len(self.steps)} steps, {self.total_tokens} tokens, "
            f"{self.duration_ms:.0f}ms; tools: {tools}"
        )


class Agent:
    """Runs a task to an answer, inside a budget."""

    def __init__(
        self,
        settings: Settings,
        client: ChatClient,
        registry: ToolRegistry,
        *,
        trace: Trace | None = None,
        cwd: Path | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        voice: bool = False,
        on_step: Callable[[Step], None] | None = None,
        kill_switch: Any | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.registry = registry
        self.trace = trace or Trace.disabled()
        self.cwd = Path(cwd or Path.cwd())
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.voice = voice
        self.on_step = on_step
        self.kill_switch = kill_switch
        self.messages: list[dict[str, Any]] = []

    # -- conversation ------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh conversation, keeping the same tools and client."""
        self.messages = [
            {
                "role": "system",
                "content": system_prompt(describe_environment(self.cwd), voice=self.voice),
            }
        ]

    def run(self, task: str) -> AgentResult:
        """Work on ``task`` until answered or out of budget.

        Conversation state persists between calls, so a follow-up question sees
        the earlier turns. Call :meth:`reset` to start over.
        """
        if not self.messages:
            self.reset()
        self.messages.append({"role": "user", "content": task})

        result = AgentResult(task=task, answer="", outcome=Outcome.FAILED)
        started = time.perf_counter()
        recent_calls: list[str] = []

        with self.trace.span("agent.run", task=task) as span:
            for index in range(1, self.max_steps + 1):
                # Checkpoint one: never spend another request on a stopped run.
                if self.kill_switch is not None and self.kill_switch.tripped:
                    result.outcome = Outcome.ABORTED
                    result.answer = "Stopped."
                    break

                if result.total_tokens >= self.token_budget:
                    result.outcome = Outcome.TOKEN_LIMIT
                    result.answer = (
                        "I stopped: this task used more than its token budget. "
                        "Try asking for something narrower."
                    )
                    break

                try:
                    reply = self.client.complete(
                        self.messages, tools=self.registry.schemas()
                    )
                except Aborted:
                    result.outcome = Outcome.ABORTED
                    result.answer = "Stopped."
                    break
                except VictorError as exc:
                    result.outcome = Outcome.FAILED
                    result.error = str(exc)
                    result.answer = f"I could not reach a model: {exc}"
                    break

                result.prompt_tokens += reply.prompt_tokens
                result.completion_tokens += reply.completion_tokens
                self.messages.append(reply.to_message())

                if not reply.wants_tools:
                    result.steps.append(Step(index, reply))
                    result.answer = reply.content
                    result.outcome = Outcome.ANSWERED
                    if self.on_step is not None:
                        self.on_step(result.steps[-1])
                    break

                try:
                    executed = self._execute(reply.tool_calls, recent_calls)
                except Aborted:
                    result.outcome = Outcome.ABORTED
                    result.answer = "Stopped."
                    result.steps.append(Step(index, reply))
                    break

                step = Step(index, reply, tuple(executed))
                result.steps.append(step)
                if self.on_step is not None:
                    self.on_step(step)
            else:
                result.outcome = Outcome.STEP_LIMIT
                result.answer = (
                    f"I stopped after {self.max_steps} steps without finishing. "
                    "Here is what I found so far. "
                    + (result.steps[-1].reply.content if result.steps else "")
                ).strip()

            result.duration_ms = (time.perf_counter() - started) * 1000
            span["outcome"] = str(result.outcome)
            span["steps"] = len(result.steps)
            span["tokens"] = result.total_tokens

        self.trace.event(
            "agent.answer",
            outcome=str(result.outcome),
            answer=result.answer,
            tokens=result.total_tokens,
        )
        return result

    # -- tools -------------------------------------------------------------

    def _execute(
        self, calls: tuple[ToolCall, ...], recent: list[str]
    ) -> list[tuple[ToolCall, ToolResult]]:
        executed: list[tuple[ToolCall, ToolResult]] = []

        for call in calls:
            # Checkpoint two: the model has chosen, but nothing has run yet.
            if self.kill_switch is not None:
                self.kill_switch.check()

            signature = str(call)
            if recent and recent[-1] == signature:
                # Identical consecutive calls mean the model is stuck. Saying so
                # costs one cheap message; letting it continue costs the budget.
                result = ToolResult(
                    ok=False,
                    error=(
                        "you already made this exact call and got the result above. "
                        "Try something different, or answer with what you know."
                    ),
                )
            else:
                with self.trace.span("tool.run", tool=call.name, arguments=call.arguments) as sp:
                    result = self.registry.run(call.name, call.arguments)
                    sp["ok"] = result.ok
                    sp["output_chars"] = len(result.output)
            recent.append(signature)

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result.for_model(),
                }
            )
            executed.append((call, result))

        return executed

    def close(self) -> None:
        self.client.close()


def build_agent(
    settings: Settings,
    *,
    trace: Trace | None = None,
    cwd: Path | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    voice: bool = False,
    registry: ToolRegistry | None = None,
    kill_switch: Any | None = None,
    confirmer: Any | None = None,
) -> Agent:
    """Wire an agent with the standard router, tools, client and safety layer."""
    from ..safety import ActionJournal, SafetyInterceptor, build_confirmer

    trace = trace or Trace.disabled()
    workdir = Path(cwd or Path.cwd())
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    router = Router(settings, ledger, on_select=trace.selection)

    if registry is None:
        journal = ActionJournal(settings.paths.journal_file, session=trace.session_id)
        interceptor = SafetyInterceptor(
            confirmer=confirmer or build_confirmer(),
            kill_switch=kill_switch,
            journal=journal,
            trace=trace,
            dry_run=settings.dry_run,
            require_confirmation=settings.confirm_destructive,
        )
        registry = build_registry(
            settings, cwd=workdir, interceptor=interceptor, kill_switch=kill_switch
        )

    return Agent(
        settings,
        ChatClient(settings, router, trace=trace),
        registry,
        trace=trace,
        cwd=workdir,
        max_steps=max_steps,
        voice=voice,
        kill_switch=kill_switch,
    )
