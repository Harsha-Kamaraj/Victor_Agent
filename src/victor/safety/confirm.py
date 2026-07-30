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
from .preview import spoken as spoken_preview

AFFIRMATIVE = frozenset(
    {"yes", "yeah", "yep", "yup", "confirm", "confirmed", "go ahead", "do it", "proceed", "ok",
     "okay", "affirmative", "sure", "run it", "y"}
)

#: Any of these anywhere in an answer makes it a refusal. "wait" and "hold on"
#: are in here because they are how a person interrupts out loud, and neither is
#: consent - previously they fell through to "not understood", which came to the
#: same thing but told the user the wrong reason.
NEGATIVE = frozenset(
    {"no", "nope", "nah", "not", "stop", "cancel", "don't", "dont", "do not", "negative",
     "abort", "never mind", "nevermind", "n", "wait", "hold on", "hang on"}
)

#: Words that turn an approval into approval of something else. "yes but not the
#: second one" is not consent to what the user was shown, so it is not a yes.
QUALIFIERS = frozenset({"but", "except", "unless", "although", "however"})

_NEGATIVE_PHRASES = tuple(phrase for phrase in NEGATIVE if " " in phrase)


@dataclass(frozen=True, slots=True)
class ConfirmRequest:
    """What the user is being asked to approve."""

    tool: str
    summary: str
    """The exact command or operation, shown verbatim."""
    classification: Classification
    undo_hint: str = ""
    preview: str = ""
    """What the command would actually do - matched files, sizes, overwrites.

    The command string alone is not enough to approve: `rm *.log` is exactly
    what the user typed, and the thing they cannot predict is what it matched.
    """

    def spoken(self) -> str:
        """Phrasing for text-to-speech: short sentences, no punctuation soup."""
        lines = [f"I want to run {self.summary}"]
        lines.append(
            spoken_preview(self.preview) if self.preview else f"This {self.classification.reason}"
        )
        if self.undo_hint:
            lines.append(self.undo_hint)
        lines.append("Say yes to continue, or no to stop")
        # Speech synthesis needs the full stops to pace the sentences, and
        # Piper emits one audio chunk per sentence - so the punctuation is what
        # makes a long prompt start playing before it has finished generating.
        # Backticks are for a terminal, not a speaker.
        spoken = " ".join(line.rstrip(". ") + "." for line in lines if line.strip())
        return spoken.replace("`", "")

    def written(self) -> str:
        parts = [self.summary, f"  {self.classification.reason}"]
        if self.preview:
            parts.append(f"  {self.preview}")
        if self.undo_hint:
            parts.append(f"  {self.undo_hint}")
        return "\n".join(parts)


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

    tokens = cleaned.replace(",", " ").split()

    # A refusal anywhere in the answer settles it, whatever came first. Only the
    # leading word used to be examined, so **"ok stop" returned True** - a word
    # from this module's own NEGATIVE set, in an answer that approved the delete
    # the user was interrupting. "yeah no" and "yes no wait" went the same way.
    # Out loud these are how people correct themselves mid-sentence, and the
    # correction is the part that matters.
    if _refuses(cleaned, tokens):
        return False

    # A qualified yes approves something other than what was described, so it is
    # not an answer to the question that was asked.
    if any(token in QUALIFIERS for token in tokens):
        return None

    # A leading answer in a longer sentence: "yes, go ahead".
    if tokens and tokens[0] in AFFIRMATIVE:
        return True
    return None


def _refuses(cleaned: str, tokens: list[str]) -> bool:
    """Whether a refusal appears anywhere in the answer."""
    if any(token in NEGATIVE for token in tokens):
        return True
    # Padded so a multi-word phrase matches on word boundaries: without it
    # "do not" is a substring of "do nothing", which is a refusal too but not
    # for the reason this would be claiming.
    padded = f" {cleaned} "
    return any(f" {phrase} " in padded for phrase in _NEGATIVE_PHRASES)


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
        from ..voice import ListenMode

        self._pipeline.speak(request.spoken())

        for attempt in range(self._attempts):
            try:
                turn = self._pipeline.listen(ListenMode.VAD)
            except Exception:
                # Deliberately everything: whatever went wrong with the
                # microphone, the answer is to ask in writing rather than to
                # guess. `NoSpeechDetected` is a VictorError and so already an
                # Exception - naming it alongside implied otherwise. Ctrl-C is a
                # BaseException and still gets out.
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
