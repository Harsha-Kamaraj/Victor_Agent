from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from victor.agent import Agent, ChatClient, Outcome
from victor.agent.llm import _estimate_tokens, _parse
from victor.config import Settings
from victor.errors import ProviderError
from victor.providers import Router
from victor.providers.registry import GPT_OSS_120B, LLAMA_33_70B
from victor.quota import QuotaLedger
from victor.tools import ToolRegistry, ToolResult, ToolSpec
from victor.tracing import Trace, read_trace

# --- scripted provider ----------------------------------------------------


def assistant(content: str = "", tool_calls: list[dict] | None = None, **usage: int) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [
            {"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt", 100),
            "completion_tokens": usage.get("completion", 20),
        },
    }


def call(name: str, arguments: dict, call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class Script:
    """Serves canned completions in order and records what it was sent."""

    def __init__(self, *responses: dict | httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if not self.responses:
            return httpx.Response(200, json=assistant("done"))
        nxt = self.responses.pop(0)
        return nxt if isinstance(nxt, httpx.Response) else httpx.Response(200, json=nxt)


class Recorder:
    """A tool that records its calls and returns a fixed result."""

    def __init__(self, name: str = "shell", output: str = "ok", ok: bool = True) -> None:
        self.spec = ToolSpec(
            name,
            "test",
            {"type": "object", "properties": {"command": {"type": "string"}}},
            mutating=False,
        )
        self.calls: list[dict] = []
        self._output = output
        self._ok = ok

    def run(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(ok=self._ok, output=self._output)


def make_agent(
    settings: Settings,
    script: Script,
    *,
    tools: list[Any] | None = None,
    trace: Trace | None = None,
    max_steps: int = 8,
    token_budget: int = 20_000,
) -> tuple[Agent, QuotaLedger]:
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    trace = trace or Trace.disabled()
    router = Router(settings, ledger, on_select=trace.selection)
    client = ChatClient(
        settings, router, client=httpx.Client(transport=httpx.MockTransport(script)), trace=trace
    )
    registry = ToolRegistry()
    for tool in tools or [Recorder()]:
        registry.register(tool)
    return (
        Agent(
            settings,
            client,
            registry,
            trace=trace,
            max_steps=max_steps,
            token_budget=token_budget,
        ),
        ledger,
    )


# --- the loop -------------------------------------------------------------


def test_answers_without_tools(settings: Settings) -> None:
    agent, _ = make_agent(settings, Script(assistant("Two plus two is four.")))
    result = agent.run("what is two plus two")

    assert result.ok
    assert result.outcome is Outcome.ANSWERED
    assert result.answer == "Two plus two is four."
    assert len(result.steps) == 1


def test_calls_a_tool_then_answers(settings: Settings) -> None:
    tool = Recorder(output="On branch main")
    agent, _ = make_agent(
        settings,
        Script(
            assistant(tool_calls=[call("shell", {"command": "git status"})]),
            assistant("You are on branch main."),
        ),
        tools=[tool],
    )
    result = agent.run("what branch am I on")

    assert result.ok
    assert tool.calls == [{"command": "git status"}]
    assert len(result.steps) == 2
    assert result.steps[0].used_tools


def test_tool_output_is_fed_back_to_the_model(settings: Settings) -> None:
    script = Script(
        assistant(tool_calls=[call("shell", {"command": "ls"})]),
        assistant("There is one file."),
    )
    agent, _ = make_agent(settings, script, tools=[Recorder(output="file.txt")])
    agent.run("what is here")

    second_request = script.requests[1]["messages"]
    tool_message = next(m for m in second_request if m["role"] == "tool")
    assert "file.txt" in tool_message["content"]
    assert tool_message["tool_call_id"] == "c1"


def test_assistant_tool_calls_are_echoed_back_verbatim(settings: Settings) -> None:
    """The API rejects tool results whose originating call is missing."""
    script = Script(
        assistant(tool_calls=[call("shell", {"command": "ls"})]),
        assistant("done"),
    )
    agent, _ = make_agent(settings, script, tools=[Recorder()])
    agent.run("list files")

    messages = script.requests[1]["messages"]
    assistant_message = next(m for m in messages if m["role"] == "assistant")
    assert assistant_message["tool_calls"][0]["id"] == "c1"


def test_several_tool_calls_in_one_step(settings: Settings) -> None:
    tool = Recorder()
    agent, _ = make_agent(
        settings,
        Script(
            assistant(
                tool_calls=[
                    call("shell", {"command": "pwd"}, "a"),
                    call("shell", {"command": "ls"}, "b"),
                ]
            ),
            assistant("Both ran."),
        ),
        tools=[tool],
    )
    result = agent.run("where am I and what is here")

    assert len(tool.calls) == 2
    assert len(result.steps[0].calls) == 2


def test_step_limit_stops_the_loop(settings: Settings) -> None:
    """A model that never stops calling tools must not run forever."""
    script = Script(
        *[
            assistant(tool_calls=[call("shell", {"command": f"echo {i}"}, f"c{i}")])
            for i in range(20)
        ]
    )
    agent, _ = make_agent(settings, script, max_steps=3)
    result = agent.run("loop forever")

    assert result.outcome is Outcome.STEP_LIMIT
    assert len(result.steps) == 3


def test_repeated_identical_calls_are_refused(settings: Settings) -> None:
    """A stuck model would otherwise spend the whole day's budget."""
    tool = Recorder()
    script = Script(
        assistant(tool_calls=[call("shell", {"command": "ls"}, "a")]),
        assistant(tool_calls=[call("shell", {"command": "ls"}, "b")]),
        assistant("I already knew that."),
    )
    agent, _ = make_agent(settings, script, tools=[tool])
    result = agent.run("list twice")

    assert len(tool.calls) == 1  # the second was intercepted, not executed
    _, second_result = result.steps[1].calls[0]
    assert not second_result.ok
    assert "already made this exact call" in (second_result.error or "")


def test_token_budget_stops_the_loop(settings: Settings) -> None:
    script = Script(
        *[
            assistant(tool_calls=[call("shell", {"command": f"echo {i}"}, f"c{i}")], prompt=9_000)
            for i in range(10)
        ]
    )
    agent, _ = make_agent(settings, script, max_steps=10, token_budget=10_000)
    result = agent.run("expensive task")

    assert result.outcome is Outcome.TOKEN_LIMIT
    assert result.total_tokens >= 10_000


def test_unknown_tool_is_reported_to_the_model(settings: Settings) -> None:
    script = Script(
        assistant(tool_calls=[call("teleport", {}, "a")]),
        assistant("I cannot do that."),
    )
    agent, _ = make_agent(settings, script)
    result = agent.run("teleport")

    assert result.ok
    _, tool_result = result.steps[0].calls[0]
    assert "no such tool" in (tool_result.error or "")


def test_provider_failure_is_surfaced_not_raised(settings: Settings) -> None:
    agent, _ = make_agent(settings, Script(httpx.Response(500, text="server on fire")))
    result = agent.run("anything")

    assert result.outcome is Outcome.FAILED
    assert result.error is not None
    assert "could not reach a model" in result.answer


def test_conversation_persists_across_runs(settings: Settings) -> None:
    script = Script(assistant("First."), assistant("Second."))
    agent, _ = make_agent(settings, script)
    agent.run("one")
    agent.run("two")

    messages = script.requests[1]["messages"]
    assert [m["role"] for m in messages].count("user") == 2


def test_reset_clears_the_conversation(settings: Settings) -> None:
    script = Script(assistant("First."), assistant("Second."))
    agent, _ = make_agent(settings, script)
    agent.run("one")
    agent.reset()
    agent.run("two")

    assert [m["role"] for m in script.requests[1]["messages"]].count("user") == 1


def test_system_prompt_describes_the_machine(settings: Settings) -> None:
    script = Script(assistant("ok"))
    agent, _ = make_agent(settings, script)
    agent.run("hello")

    system = script.requests[0]["messages"][0]
    assert system["role"] == "system"
    assert "Victor" in system["content"]
    assert str(agent.cwd) in system["content"]


def test_tool_schemas_are_sent(settings: Settings) -> None:
    script = Script(assistant("ok"))
    agent, _ = make_agent(settings, script)
    agent.run("hello")

    tools = script.requests[0]["tools"]
    assert tools[0]["function"]["name"] == "shell"
    assert script.requests[0]["tool_choice"] == "auto"


# --- quota and routing ----------------------------------------------------


def test_tokens_are_reconciled_into_the_ledger(settings: Settings) -> None:
    agent, ledger = make_agent(
        settings, Script(assistant("done", prompt=300, completion=50))
    )
    agent.run("hello")

    requests, tokens, _ = ledger.usage(GPT_OSS_120B.key)
    assert requests == 1
    assert tokens == 350


def test_provider_429_falls_through_to_the_next_model(settings: Settings) -> None:
    """A 429 is authoritative even when the ledger believed there was room."""
    script = Script(httpx.Response(429, json={}), assistant("answered by the fallback"))
    agent, ledger = make_agent(settings, script)
    result = agent.run("hello")

    assert result.ok
    assert result.answer == "answered by the fallback"
    assert script.requests[0]["model"] == GPT_OSS_120B.model
    assert script.requests[1]["model"] == LLAMA_33_70B.model
    # The rate-limited model is marked spent so the next call skips it.
    assert not ledger.check(GPT_OSS_120B.key, GPT_OSS_120B.limits).allowed


def test_a_per_minute_429_does_not_cost_the_whole_day(settings: Settings) -> None:
    """Observed live: fifteen real calls made, 1,000 phantom ones written, and
    the best text model unavailable until tomorrow - because Groq said "try
    again in 8.5s" and that was read as the day's allowance running out."""
    script = Script(
        httpx.Response(429, json={}, headers={"retry-after": "8.5"}),
        assistant("answered by the fallback"),
    )
    agent, ledger = make_agent(settings, script)
    result = agent.run("hello")

    assert result.ok
    assert script.requests[1]["model"] == LLAMA_33_70B.model  # still fell through
    requests, _, _ = ledger.usage(GPT_OSS_120B.key)
    assert requests == 1, f"a short wait burned {requests} requests"
    assert ledger.check(GPT_OSS_120B.key, GPT_OSS_120B.limits).allowed


def test_a_retry_after_in_the_body_is_read_when_the_header_is_missing(
    settings: Settings,
) -> None:
    """Groq does not always send the header; it always says it in the message."""
    body = {
        "error": {
            "message": (
                "Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
                "minute (TPM): Limit 6000. Please try again in 7.482s."
            )
        }
    }
    agent, ledger = make_agent(
        settings, Script(httpx.Response(429, json=body), assistant("fallback"))
    )
    agent.run("hello")

    requests, _, _ = ledger.usage(GPT_OSS_120B.key)
    assert requests == 1


def test_a_long_429_still_stands_the_model_down_for_the_day(settings: Settings) -> None:
    """The original behaviour, kept: when the wait really is the day, the
    ledger should carry that into the next run rather than rediscovering it."""
    script = Script(
        httpx.Response(429, json={}, headers={"retry-after": "7200"}),
        assistant("answered by the fallback"),
    )
    agent, ledger = make_agent(settings, script)
    agent.run("hello")

    assert not ledger.check(GPT_OSS_120B.key, GPT_OSS_120B.limits).allowed


def test_a_429_with_no_wait_given_is_treated_as_the_day(settings: Settings) -> None:
    """Unknown must not mean "carry on" - that is how a real exhaustion turns
    into a loop of 429s against a provider that has already said no."""
    agent, ledger = make_agent(
        settings, Script(httpx.Response(429, json={}), assistant("fallback"))
    )
    agent.run("hello")

    assert not ledger.check(GPT_OSS_120B.key, GPT_OSS_120B.limits).allowed


def test_every_model_rate_limited_reports_clearly(settings: Settings) -> None:
    agent, _ = make_agent(
        settings, Script(*[httpx.Response(429, json={}) for _ in range(5)])
    )
    result = agent.run("hello")

    assert result.outcome is Outcome.FAILED
    assert "rate limited" in (result.error or "")


def test_rejected_key_is_not_retried_as_a_rate_limit(settings: Settings) -> None:
    agent, _ = make_agent(settings, Script(httpx.Response(401, json={})))
    result = agent.run("hello")

    assert result.outcome is Outcome.FAILED
    assert "rejected" in (result.error or "")


# --- parsing --------------------------------------------------------------


def test_parse_handles_malformed_tool_arguments() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "x", "function": {"name": "shell", "arguments": "{not json"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    reply = _parse(body, "groq:test", 10.0)

    assert reply.tool_calls[0].arguments == {}  # reported, not crashed


def test_parse_rejects_an_empty_response() -> None:
    with pytest.raises(ProviderError, match="no choices"):
        _parse({"choices": []}, "groq:test", 1.0)


def test_token_estimate_scales_with_input() -> None:
    small = _estimate_tokens([{"role": "user", "content": "hi"}], None)
    large = _estimate_tokens([{"role": "user", "content": "x" * 4_000}], None)

    assert large > small * 10


# --- tracing --------------------------------------------------------------


def test_a_run_is_traced_end_to_end(settings: Settings, tmp_path: Path) -> None:
    with Trace.open(settings.paths.ensure().traces_dir, label="agent") as trace:
        agent, _ = make_agent(
            settings,
            Script(
                assistant(tool_calls=[call("shell", {"command": "ls"})]),
                assistant("There is one file."),
            ),
            trace=trace,
        )
        agent.run("what is here")
        path = trace.path

    kinds = [e["kind"] for e in read_trace(path)]
    assert "router.select" in kinds
    assert "llm.reply" in kinds
    assert "tool.run" in kinds
    assert "agent.run" in kinds
    assert "agent.answer" in kinds


def test_a_long_session_does_not_grow_without_limit(settings: Settings) -> None:
    """Free tiers meter tokens per *minute*, so history is a rate limit.

    `converse` reuses one agent so a follow-up can say "the other one", and
    nothing ever dropped anything. A desktop turn appends a whole accessibility
    tree, so five turns reached 9,000 tokens and the session died on
    "rate limited (5744/6000 tok/min)" - a limit the user never spent anything
    to reach.
    """
    from victor.agent.loop import Agent

    agent = Agent.__new__(Agent)
    agent.history_budget = 500
    agent.messages = [{"role": "system", "content": "SYSTEM PROMPT"}]
    for turn in range(20):
        agent.messages.append({"role": "user", "content": f"turn {turn}"})
        agent.messages.append({"role": "assistant", "content": "x" * 200})

    agent._forget_old_turns()

    body = sum(len(str(m["content"])) for m in agent.messages[1:])
    assert body <= 500, f"history kept {body} characters"
    assert agent.messages[0]["content"] == "SYSTEM PROMPT", "the tool rules were dropped"
    assert agent.messages[-1]["content"] == "x" * 200, "the newest turn was dropped"


def test_trimming_never_orphans_a_tool_result(settings: Settings) -> None:
    """A tool message whose originating call has been dropped is a protocol
    error at every provider, so the pair has to go together."""
    from victor.agent.loop import Agent

    agent = Agent.__new__(Agent)
    agent.history_budget = 120
    agent.messages = [
        {"role": "system", "content": "S"},
        {"role": "assistant", "content": "a" * 100},
        {"role": "tool", "content": "t" * 100},
        {"role": "assistant", "content": "b" * 100},
    ]

    agent._forget_old_turns()

    assert agent.messages[1]["role"] != "tool", "a tool result outlived its call"
