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

import contextlib
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

DEFAULT_HISTORY_BUDGET = 12_000
"""Characters of conversation carried between turns of a session.

Roughly 3,000 tokens. Free tiers meter tokens *per minute* - 6,000 on the
smallest Groq model - so an unbounded history stops the session on a limit
the user never spent anything to reach.
"""


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

    # -- what the run cost -------------------------------------------------
    #
    # The project's claim is that most of what an agent does on a desktop needs
    # no API call at all, because the operating system already knows what is on
    # screen. A claim like that is worth nothing unless it is counted, so tools
    # report what they spent in ``metadata["cost"]`` - requests, not dollars -
    # and these three properties are what turn that into the number quoted in
    # the README.

    @property
    def billed_tool_calls(self) -> int:
        """Tool calls that spent a provider request. Vision is the only one."""
        return sum(1 for _, res in self.tool_calls if res.metadata.get("cost", 0))

    @property
    def free_tool_calls(self) -> int:
        return len(self.tool_calls) - self.billed_tool_calls

    @property
    def api_calls(self) -> int:
        """Provider requests: one per think-act cycle, plus any billed tool."""
        return len(self.steps) + self.billed_tool_calls

    @property
    def zero_cost_ratio(self) -> float:
        """Share of tool calls that cost nothing. 1.0 when nothing was billed."""
        total = len(self.tool_calls)
        return 1.0 if not total else self.free_tool_calls / total

    def summary(self) -> str:
        tools = ", ".join(str(call) for call, _ in self.tool_calls) or "none"
        cost = (
            f"{self.free_tool_calls}/{len(self.tool_calls)} tool calls free"
            if self.tool_calls
            else "no tool calls"
        )
        return (
            f"{self.outcome} in {len(self.steps)} steps, {self.api_calls} API calls, "
            f"{self.total_tokens} tokens, {self.duration_ms:.0f}ms; "
            f"{cost}; tools: {tools}"
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
        history_budget: int = DEFAULT_HISTORY_BUDGET,
        voice: bool = False,
        on_step: Callable[[Step], None] | None = None,
        kill_switch: Any | None = None,
        memory: Any | None = None,
        owns_memory: bool = False,
    ) -> None:
        self.settings = settings
        self.client = client
        self.registry = registry
        self.trace = trace or Trace.disabled()
        self.cwd = Path(cwd or Path.cwd())
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.history_budget = history_budget
        self.voice = voice
        self.on_step = on_step
        self.kill_switch = kill_switch
        self.memory = memory
        self.owns_memory = owns_memory
        """Whether :meth:`close` should shut the memory down. See :meth:`close`."""
        self.watcher = None
        if memory is not None:
            from ..rag.ingest import ErrorFixWatcher

            self.watcher = ErrorFixWatcher(memory)
        self.messages: list[dict[str, Any]] = []
        self.recalled = 0
        """How many times memory had something useful to say this run."""

    # -- conversation ------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh conversation, keeping the same tools and client."""
        self.messages = [
            {
                "role": "system",
                "content": system_prompt(
                    describe_environment(self.cwd),
                    voice=self.voice,
                    # Ask the registry rather than a flag: the prompt then
                    # cannot drift out of step with what was actually wired in.
                    desktop="click" in self.registry,
                ),
            }
        ]

    def run(self, task: str) -> AgentResult:
        """Work on ``task`` until answered or out of budget.

        Conversation state persists between calls, so a follow-up question sees
        the earlier turns. Call :meth:`reset` to start over.
        """
        if not self.messages:
            self.reset()
        self._forget_old_turns()
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

    def _forget_old_turns(self) -> None:
        """Keep the conversation, but stop it growing without limit.

        ``converse`` reuses one agent so a follow-up can say "and the other
        one", which is worth having. But nothing dropped anything, and a desktop
        turn appends a whole accessibility tree - so five turns reached 9,000
        tokens and the run died on tokens-per-minute rather than on anything the
        user asked for:

            no provider available for text:
              groq:llama-3.1-8b-instant: rate limited (5744/6000 tok/min)

        Trimming from the front keeps the system prompt, which carries the tool
        rules, and keeps the most recent exchanges, which is where a pronoun
        points. A tool result is dropped with the assistant turn that requested
        it, because a tool message whose call is gone is a protocol error at
        every provider.
        """
        budget = self.history_budget
        if budget <= 0 or len(self.messages) < 2:
            return

        head, rest = self.messages[:1], self.messages[1:]
        while rest and sum(len(str(m.get("content") or "")) for m in rest) > budget:
            del rest[0]
            # Never leave a tool result orphaned by the call it answers.
            while rest and rest[0].get("role") == "tool":
                del rest[0]
        self.messages = head + rest

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
                    # Marked so memory can tell "this failed" from "this never
                    # ran". Recording a refusal as a failure would make the next
                    # success look like a fix for it.
                    metadata={"refused": "repeat"},
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
            self._remember(call, result)

        return executed

    def _remember(self, call: ToolCall, result: ToolResult) -> None:
        """Feed the result to memory, and inject a past fix if there is one.

        Both halves hang off the same place because they are the same event: an
        action failed. One half asks whether this has been seen before, the
        other starts watching for what resolves it.

        This used to run only for the shell, which left the desktop - the part
        of the agent that fails in the most repetitive ways - learning nothing
        from a session to the next. :func:`describe_call` gives every tool an
        identity, so the rule is now simply "whatever ran".

        Deliberately never raises. Memory is an optimisation - a corrupt store
        or a missing model must degrade the run to "no recall", never fail it.
        """
        if self.memory is None:
            return
        if result.metadata.get("decision") or result.metadata.get("refused"):
            # Blocked by the safety layer, refused as a repeat, or turned away
            # by a tool before it acted. Nothing ran, so there is no failure to
            # remember and nothing a later success could be the fix for.
            return

        try:
            from ..rag.ingest import describe_call

            if call.name not in self.registry:
                return  # a name the model invented; the call never happened
            action = describe_call(
                call.name,
                call.arguments,
                mutating=self.registry.get(call.name).spec.mutating,
            )
            if action is None:
                return

            if self.watcher is not None:
                note = self.watcher.observe(
                    action, ok=result.ok, output=result.for_model()
                )
                if note:
                    self.trace.event("memory.capture", action=action.line, note=note)

            if result.ok:
                return

            recollection = self.memory.recall_for_error(result.for_model())
            if not recollection.found:
                return

            self.recalled += 1
            # Injected as a user turn rather than a system one: the system
            # prompt is the agent's standing instructions, and a note about one
            # error is not that. It also keeps the recall next to the failure
            # it refers to, instead of thousands of tokens earlier.
            self.messages.append({"role": "user", "content": recollection.for_model()})
            self.trace.event(
                "memory.recalled",
                action=action.line,
                score=round(recollection.best.score, 3) if recollection.best else 0.0,
                cost=0,
            )
        except Exception as exc:  # noqa: BLE001 - memory must never break a run
            self.trace.event("memory.error", detail=f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        """Release what this agent opened.

        The memory holds a SQLite connection, and nothing closed it: every
        ``victor do`` and every ``victor converse`` session leaked the handle.
        POSIX hides that - an open file can still be unlinked - so the suite was
        silent while the ResourceWarnings piled up unread. On Windows a held
        handle is exactly what stops a file being removed, which is the same
        shape as the two selftest gates that passed every assertion and then
        died in cleanup there.

        Only what it opened. A caller who passes ``memory=`` keeps the right to
        go on using it afterwards, which several tests do, so ownership is
        recorded at construction rather than guessed at here.
        """
        self.client.close()
        if not self.owns_memory or self.memory is None:
            return
        close = getattr(self.memory, "close", None)
        if callable(close):
            # A memory that will not shut cleanly must not turn a finished run
            # into a failed one - the answer is already in hand by this point.
            with contextlib.suppress(Exception):
                close()


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
    desktop: bool | None = None,
    app: str | None = None,
    memory: Any | None = None,
) -> Agent:
    """Wire an agent with the standard router, tools, client and safety layer."""
    from ..safety import (
        ActionJournal,
        SafetyInterceptor,
        build_adjudicator,
        build_confirmer,
        trash_for,
    )

    trace = trace or Trace.disabled()
    workdir = Path(cwd or Path.cwd())
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    router = Router(settings, ledger, on_select=trace.selection)

    if registry is None:
        use_desktop = desktop if desktop is not None else settings.desktop_control
        vision = None
        if use_desktop:
            from ..desktop.vision import VisionClient

            # Only built alongside the desktop tools: the vision fallback exists
            # to read surfaces the tree cannot, and there is nothing to fall
            # back from when the agent is not looking at a screen.
            vision = VisionClient(settings, router, trace=trace)

        journal = ActionJournal(settings.paths.journal_file, session=trace.session_id)
        trash = trash_for(settings.data_dir, trace.session_id)
        interceptor = SafetyInterceptor(
            confirmer=confirmer or build_confirmer(),
            kill_switch=kill_switch,
            journal=journal,
            trace=trace,
            dry_run=settings.dry_run,
            require_confirmation=settings.confirm_destructive,
            trash=trash,
            cwd=workdir,
            adjudicator=build_adjudicator(settings, ledger, trace=trace),
        )
        registry = build_registry(
            settings,
            cwd=workdir,
            interceptor=interceptor,
            kill_switch=kill_switch,
            trash=trash,
            desktop=use_desktop,
            app=app,
            vision=vision,
        )

    # Whoever opens the store closes it. Tracked rather than inferred, because
    # an injected memory belongs to the caller and closing it would pull the
    # floor out from under them.
    owns_memory = False
    if memory is None and settings.memory_enabled:
        from ..rag import build_memory

        try:
            memory = build_memory(settings, trace=trace)
            owns_memory = memory is not None
        except VictorError as exc:
            # A memory that will not open is a reason to run without one, not a
            # reason not to run. The commonest cause is a store built with a
            # different embedder, and it says how to fix itself.
            trace.event("memory.unavailable", detail=str(exc))
            memory = None

    return Agent(
        settings,
        ChatClient(settings, router, trace=trace),
        registry,
        trace=trace,
        cwd=workdir,
        max_steps=max_steps,
        voice=voice,
        kill_switch=kill_switch,
        memory=memory,
        owns_memory=owns_memory,
    )
