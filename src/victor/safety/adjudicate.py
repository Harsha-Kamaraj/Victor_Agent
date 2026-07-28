"""The third classification layer: a cheap LLM opinion on unknown commands.

The plan: *"LLM adjudication (Groq, cheap) for anything unmatched — fails
closed to CONFIRM on error or timeout."*

The rules layer is deterministic, auditable and free, so it goes first and its
verdicts are final. This layer exists only for the residue — commands no rule
recognised, where the fail-closed default is CONFIRM. Left alone, that default
prompts on every unfamiliar binary a developer runs, and a safety layer that
cries wolf gets clicked through.

Three constraints shape the implementation:

* **It can only lower risk to SAFE, never raise it above what the rules said.**
  An LLM that could escalate to DENY would make refusals non-deterministic, and
  one that could downgrade a rule-matched CONFIRM would make the whole
  classifier only as trustworthy as its weakest layer.
* **Every failure means CONFIRM.** Timeout, network error, unparseable answer,
  no API key — all land on the same conservative default the rules already
  chose.
* **It is cached and cheap.** Pinned to the highest-allowance model, capped at
  a handful of tokens, and memoised per command so a retry loop cannot spend
  the day's budget asking the same question.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..tracing import Trace
from .classify import Classification, Risk

MAX_TOKENS = 4
TIMEOUT_SECONDS = 6.0
MAX_COMMAND_CHARS = 400

SYSTEM = (
    "You classify shell commands for a safety gate. Answer with exactly one "
    "word and nothing else.\n\n"
    "SAFE — the command only reads, inspects or reports. It creates nothing, "
    "changes nothing, deletes nothing, installs nothing and contacts no network.\n"
    "CONFIRM — anything else: it writes, deletes, installs, moves, changes "
    "permissions, starts or stops processes, or reaches the network.\n\n"
    "If you are unsure, answer CONFIRM."
)

_VERDICT = re.compile(r"\b(SAFE|CONFIRM|DENY)\b", re.IGNORECASE)


@dataclass
class AdjudicationStats:
    """What the layer did, for the session summary."""

    asked: int = 0
    downgraded: int = 0
    upheld: int = 0
    failed: int = 0
    cached: int = 0

    def summary(self) -> str:
        return (
            f"{self.asked} adjudicated, {self.downgraded} cleared, "
            f"{self.upheld} upheld, {self.failed} failed closed"
        )


@dataclass
class LLMAdjudicator:
    """Asks a model whether an unrecognised command is read-only.

    Takes a ``complete`` callable rather than a client so the safety layer does
    not import the agent, and so tests can drive it without a network.
    """

    complete: Callable[[str], str] | None = None
    trace: Trace = field(default_factory=Trace.disabled)
    stats: AdjudicationStats = field(default_factory=AdjudicationStats)
    _cache: dict[str, Risk] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.complete is not None

    def review(self, command: str, current: Classification) -> Classification:
        """Return a refined verdict, or ``current`` unchanged.

        Only ever returns SAFE or the verdict it was given. Never escalates.
        """
        if not current.unmatched or self.complete is None:
            return current

        key = " ".join(command.split())[:MAX_COMMAND_CHARS]
        if key in self._cache:
            self.stats.cached += 1
            return self._apply(self._cache[key], current, cached=True)

        self.stats.asked += 1
        try:
            answer = self.complete(key)
        except Exception as exc:
            # Any failure lands on the conservative default the rules chose.
            self.stats.failed += 1
            self.trace.event(
                "safety.adjudicate",
                command=key,
                outcome="failed-closed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return Classification(
                risk=current.risk,
                reason=f"{current.reason}; second opinion unavailable",
                trigger=current.trigger,
                source="llm-failed",
            )

        verdict = _parse(answer)
        if verdict is None:
            self.stats.failed += 1
            self.trace.event(
                "safety.adjudicate", command=key, outcome="unparseable", answer=answer[:80]
            )
            return Classification(
                risk=current.risk,
                reason=f"{current.reason}; second opinion was unclear",
                trigger=current.trigger,
                source="llm-failed",
            )

        self._cache[key] = verdict
        return self._apply(verdict, current)

    def _apply(
        self, verdict: Risk, current: Classification, *, cached: bool = False
    ) -> Classification:
        # Refuse to escalate. DENY is the rules layer's decision to make, and a
        # non-deterministic refusal is worse than a consistent prompt.
        if verdict is Risk.SAFE:
            if not cached:
                self.stats.downgraded += 1
            self.trace.event("safety.adjudicate", outcome="cleared", cached=cached or None)
            return Classification(
                risk=Risk.SAFE,
                reason="checked and found read-only",
                trigger=current.trigger,
                source="llm",
            )

        if not cached:
            self.stats.upheld += 1
        return Classification(
            risk=current.risk,
            reason=f"{current.reason}, and a check agreed it may have effects",
            trigger=current.trigger,
            source="llm",
        )


def _parse(answer: str) -> Risk | None:
    match = _VERDICT.search(answer or "")
    if match is None:
        return None
    word = match.group(1).upper()
    if word == "SAFE":
        return Risk.SAFE
    # DENY from the model is treated as CONFIRM: escalation stays with the rules.
    return Risk.CONFIRM


def build_adjudicator(
    settings, ledger, *, trace: Trace | None = None, enabled: bool = True
) -> LLMAdjudicator:
    """Wire an adjudicator onto the cheapest text model, or a disabled one.

    Deliberately pinned to the highest daily allowance rather than the best
    reasoner. This is a one-word classification on a command the rules already
    handled conservatively; it must never compete with the agent loop for the
    scarce models. The pin promotes that model to the front of the chain and
    leaves the rest as fallbacks, so exhausting it degrades rather than fails.

    The ledger is shared with the main router, so adjudication spends from the
    same visible budget as everything else.
    """
    trace = trace or Trace.disabled()
    if not enabled or not settings.has("groq_api_key"):
        return LLMAdjudicator(complete=None, trace=trace)

    from ..agent.llm import ChatClient
    from ..providers import Router
    from ..providers.registry import LLAMA_31_8B

    pinned = settings.model_copy(update={"text_model": LLAMA_31_8B.model})
    router = Router(pinned, ledger)
    client = ChatClient(pinned, router, trace=trace, timeout=TIMEOUT_SECONDS)

    def complete(command: str) -> str:
        reply = client.complete(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Command: {command}"},
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        return reply.content

    return LLMAdjudicator(complete=complete, trace=trace)
