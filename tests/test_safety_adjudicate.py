from __future__ import annotations

import httpx
import pytest

from victor.config import Settings
from victor.quota import QuotaLedger
from victor.safety import LLMAdjudicator, Risk, SafetyInterceptor, build_adjudicator
from victor.safety.classify import classify_shell
from victor.tools.base import Decision, ToolSpec

SHELL = ToolSpec("shell", "run", {"type": "object"}, mutating=True)


def adjudicator(answer: str | Exception) -> LLMAdjudicator:
    def complete(command: str) -> str:
        if isinstance(answer, Exception):
            raise answer
        return answer

    return LLMAdjudicator(complete=complete)


# --- what it is asked about -----------------------------------------------


def test_only_unmatched_commands_are_adjudicated() -> None:
    """Rules are deterministic and free; they do not get a second opinion."""
    asked: list[str] = []

    def complete(command: str) -> str:
        asked.append(command)
        return "SAFE"

    judge = LLMAdjudicator(complete=complete)

    judge.review("ls -la", classify_shell("ls -la"))  # rule says SAFE
    judge.review("rm notes.txt", classify_shell("rm notes.txt"))  # rule says CONFIRM
    judge.review("rm -rf /", classify_shell("rm -rf /"))  # rule says DENY

    assert asked == []


def test_an_unknown_command_is_adjudicated() -> None:
    asked: list[str] = []

    def complete(command: str) -> str:
        asked.append(command)
        return "SAFE"

    judge = LLMAdjudicator(complete=complete)
    verdict = judge.review("frobnicate --list", classify_shell("frobnicate --list"))

    assert asked == ["frobnicate --list"]
    assert verdict.risk is Risk.SAFE
    assert verdict.source == "llm"


# --- what it may decide ---------------------------------------------------


def test_it_can_clear_an_unknown_command() -> None:
    before = classify_shell("exotictool --version")
    assert before.risk is Risk.CONFIRM and before.unmatched

    after = adjudicator("SAFE").review("exotictool --version", before)
    assert after.risk is Risk.SAFE
    assert "read-only" in after.reason


def test_it_upholds_the_default_when_it_agrees() -> None:
    verdict = adjudicator("CONFIRM").review(
        "exotictool --wipe", classify_shell("exotictool --wipe")
    )

    assert verdict.risk is Risk.CONFIRM
    assert verdict.source == "llm"


def test_it_cannot_escalate_to_deny() -> None:
    """A non-deterministic refusal is worse than a consistent prompt."""
    verdict = adjudicator("DENY").review("exotictool --nuke", classify_shell("exotictool --nuke"))

    assert verdict.risk is Risk.CONFIRM  # not DENY


# --- failing closed -------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    ["", "I'm not sure about that", "maybe?", "```json\n{}\n```"],
)
def test_an_unparseable_answer_fails_closed(answer: str) -> None:
    verdict = adjudicator(answer).review("weirdtool run", classify_shell("weirdtool run"))

    assert verdict.risk is Risk.CONFIRM
    assert verdict.source == "llm-failed"
    assert "unclear" in verdict.reason


def test_a_network_error_fails_closed() -> None:
    verdict = adjudicator(httpx.ConnectError("down")).review(
        "weirdtool run", classify_shell("weirdtool run")
    )

    assert verdict.risk is Risk.CONFIRM
    assert verdict.source == "llm-failed"
    assert "unavailable" in verdict.reason


def test_a_timeout_fails_closed() -> None:
    verdict = adjudicator(TimeoutError("too slow")).review(
        "weirdtool run", classify_shell("weirdtool run")
    )
    assert verdict.risk is Risk.CONFIRM


def test_no_client_means_the_rules_stand() -> None:
    judge = LLMAdjudicator(complete=None)
    before = classify_shell("weirdtool run")

    assert judge.review("weirdtool run", before) is before
    assert not judge.available


# --- cost control ---------------------------------------------------------


def test_repeat_commands_are_cached() -> None:
    calls: list[str] = []

    def complete(command: str) -> str:
        calls.append(command)
        return "SAFE"

    judge = LLMAdjudicator(complete=complete)
    for _ in range(5):
        judge.review("frobnicate --list", classify_shell("frobnicate --list"))

    assert len(calls) == 1
    assert judge.stats.cached == 4


def test_whitespace_variants_share_a_cache_entry() -> None:
    calls: list[str] = []

    def complete(command: str) -> str:
        calls.append(command)
        return "SAFE"

    judge = LLMAdjudicator(complete=complete)
    judge.review("frobnicate  --list", classify_shell("frobnicate --list"))
    judge.review("frobnicate --list", classify_shell("frobnicate --list"))

    assert len(calls) == 1


def test_stats_track_outcomes() -> None:
    judge = adjudicator("SAFE")
    judge.review("a-tool x", classify_shell("a-tool x"))
    judge = adjudicator("CONFIRM")
    judge.review("b-tool x", classify_shell("b-tool x"))

    assert judge.stats.asked == 1
    assert judge.stats.upheld == 1


# --- through the interceptor ----------------------------------------------


def test_the_interceptor_skips_the_prompt_when_cleared() -> None:
    from victor.safety import AutoConfirmer

    confirmer = AutoConfirmer(True)
    gate = SafetyInterceptor(confirmer=confirmer, adjudicator=adjudicator("SAFE"))

    review = gate.review(SHELL, {"command": "frobnicate --list"})
    assert review.decision is Decision.ALLOW
    assert confirmer.requests == []  # never had to ask


def test_the_interceptor_still_prompts_when_upheld() -> None:
    from victor.safety import AutoConfirmer

    confirmer = AutoConfirmer(True)
    gate = SafetyInterceptor(confirmer=confirmer, adjudicator=adjudicator("CONFIRM"))

    gate.review(SHELL, {"command": "frobnicate --deploy"})
    assert len(confirmer.requests) == 1


def test_the_interceptor_prompts_when_adjudication_fails() -> None:
    from victor.safety import AutoConfirmer

    confirmer = AutoConfirmer(True)
    gate = SafetyInterceptor(
        confirmer=confirmer, adjudicator=adjudicator(httpx.ConnectError("down"))
    )

    gate.review(SHELL, {"command": "frobnicate --deploy"})
    assert len(confirmer.requests) == 1


def test_adjudication_never_softens_a_denial() -> None:
    from victor.safety import AutoConfirmer

    gate = SafetyInterceptor(
        confirmer=AutoConfirmer(True), adjudicator=adjudicator("SAFE")
    )
    assert gate.review(SHELL, {"command": "rm -rf /"}).decision is Decision.DENY


# --- construction ---------------------------------------------------------


def test_builder_is_disabled_without_a_key(tmp_path) -> None:
    settings = Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path))
    ledger = QuotaLedger(tmp_path / "quota.json")

    assert not build_adjudicator(settings, ledger).available


def test_builder_is_disabled_on_request(settings: Settings, tmp_path) -> None:
    ledger = QuotaLedger(tmp_path / "quota.json")
    assert not build_adjudicator(settings, ledger, enabled=False).available


def test_builder_pins_the_high_allowance_model(settings: Settings, tmp_path) -> None:
    """Adjudication must not compete with the agent for the scarce models."""
    from victor.providers.registry import LLAMA_31_8B

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SAFE"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1},
            },
        )

    ledger = QuotaLedger(tmp_path / "quota.json")
    judge = build_adjudicator(settings, ledger)
    # Swap in the mock transport behind the built client.
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    httpx.Client.__init__ = patched  # type: ignore[method-assign]
    try:
        judge = build_adjudicator(settings, ledger)
        judge.review("frobnicate --list", classify_shell("frobnicate --list"))
    finally:
        httpx.Client.__init__ = original  # type: ignore[method-assign]

    assert seen == [LLAMA_31_8B.model]
