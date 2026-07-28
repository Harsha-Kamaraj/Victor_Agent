"""Asking the user before doing something irreversible.

Every confirmer here fails closed. Silence is no, an unparseable answer is no,
a closed stdin is no, and a missing microphone is no. The only path to yes is
an affirmative the user actually gave.

That matters most in the spoken case, where the answer arrives through speech
recognition and may be wrong. "No" misheard as "go" would run a delete the user
just refused, so the affirmative set is small and explicit, and anything
outside it is a refusal rather than a guess.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .classify import Classification, Risk

AFFIRMATIVE = frozenset(
    {"yes", "yeah", "yep", "yup", "confirm", "confirmed", "go ahead", "do it", "proceed", "ok",
     "okay", "affirmative", "sure", "run it", "y"}
)

NEGATIVE = frozenset(
    {"no", "nope", "nah", "stop", "cancel", "don't", "dont", "do not", "negative", "abort",
     "never mind", "nevermind", "n"}
)


@dataclass(frozen=True, slots=True)
class ConfirmRequest:
    """What the user is being asked to approve."""

    tool: str
    summary: str
    """The exact command or operation, shown verbatim."""
    classification: Classification
    undo_hint: str = ""

    def spoken(self) -> str:
        """Phrasing for text-to-speech: short sentences, no punctuation soup."""
        lines = [f"I want to run {self.summary}.", f"This {self.classification.reason}."]
        if self.undo_hint:
            lines.append(self.undo_hint)
        lines.append("Say yes to continue, or no to stop.")
        return " ".join(lines)

    def written(self) -> str:
        return f"{self.summary}\n  {self.classification.reason}" + (
            f"\n  {self.undo_hint}" if self.undo_hint else ""
        )


@runtime_checkable
class Confirmer(Protocol):
    """Asks the user a yes/no question about a pending action."""

    def confirm(self, request: ConfirmRequest) -> bool: ...


def interpret(answer: str) -> bool | None:
    """Map a free-text answer to yes, no, or "could not tell".

    ``None`` is not "maybe" - callers treat it as a refusal. It is kept
    distinct so the user can be told their answer was not understood rather
    than being silently overruled.
    """
    cleaned = answer.strip().lower().rstrip(".!?").strip()
    if not cleaned:
        return None
    if cleaned in AFFIRMATIVE:
        return True
    if cleaned in NEGATIVE:
        return False

    # Allow a leading answer in a longer sentence: "yes, go ahead".
    first = cleaned.replace(",", " ").split()
    if first:
        if first[0] in AFFIRMATIVE:
            return True
        if first[0] in NEGATIVE:
            return False
    return None


class AutoConfirmer:
    """Always answers the same way. For tests and non-interactive runs."""

    def __init__(self, answer: bool, *, reason: str = "") -> None:
        self.answer = answer
        self.reason = reason
        self.requests: list[ConfirmRequest] = []

    def confirm(self, request: ConfirmRequest) -> bool:
        self.requests.append(request)
        return self.answer


class DenyingConfirmer(AutoConfirmer):
    """The default when nobody can be asked. Refuses everything."""

    def __init__(self) -> None:
        super().__init__(False, reason="no interactive confirmer is available")


class TypedConfirmer:
    """Terminal prompt. The fallback whenever voice is not in play."""

    def __init__(self, *, stream=None, output=None) -> None:
        self._in = stream or sys.stdin
        self._out = output or sys.stderr

    def confirm(self, request: ConfirmRequest) -> bool:
        if not self._in or not self._in.isatty():
            # Piped or absent stdin cannot answer, so the answer is no.
            print(
                f"\nrefusing (no terminal to confirm on): {request.summary}",
                file=self._out,
            )
            return False

        print(f"\nVictor wants to run:\n  {request.written()}", file=self._out)
        print("Continue? [y/N] ", end="", file=self._out, flush=True)
        try:
            answer = self._in.readline()
        except (OSError, ValueError):
            return False
        verdict = interpret(answer)
        if verdict is None:
            print("not understood - treating that as no", file=self._out)
            return False
        return verdict


class SpokenConfirmer:
    """Speaks the request and listens for the answer.

    Falls back to the terminal when the microphone yields nothing, so a failed
    capture asks again in writing rather than silently refusing something the
    user wanted.
    """

    def __init__(
        self,
        pipeline,
        *,
        fallback: Confirmer | None = None,
        attempts: int = 2,
    ) -> None:
        self._pipeline = pipeline
        self._fallback = fallback if fallback is not None else TypedConfirmer()
        self._attempts = attempts

    def confirm(self, request: ConfirmRequest) -> bool:
        from ..voice import ListenMode, NoSpeechDetected

        self._pipeline.speak(request.spoken())

        for attempt in range(self._attempts):
            try:
                turn = self._pipeline.listen(ListenMode.VAD)
            except (NoSpeechDetected, Exception):
                break

            verdict = interpret(turn.text)
            if verdict is not None:
                return verdict
            if attempt + 1 < self._attempts:
                self._pipeline.speak("I did not catch that. Please say yes or no.")

        return self._fallback.confirm(request)


def build_confirmer(
    *, voice_pipeline=None, interactive: bool = True
) -> Confirmer:
    """Pick a confirmer for the current session."""
    if voice_pipeline is not None:
        return SpokenConfirmer(voice_pipeline)
    if interactive and sys.stdin and sys.stdin.isatty():
        return TypedConfirmer()
    return DenyingConfirmer()


def summarise_call(tool: str, arguments: dict) -> str:
    """Render a tool call the way the user should see it before approving."""
    if tool == "shell":
        return " ".join(str(arguments.get("command", "")).split())
    if tool == "git":
        args = " ".join(str(a) for a in arguments.get("args") or [])
        return f"git {arguments.get('subcommand', '')} {args}".strip()
    rendered = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return f"{tool}({rendered})"


def risk_word(risk: Risk) -> str:
    return {Risk.SAFE: "safe", Risk.CONFIRM: "needs confirmation", Risk.DENY: "refused"}[risk]
