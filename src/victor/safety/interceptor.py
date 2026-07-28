"""The safety interceptor.

Slots into the seam P2 left in :class:`victor.tools.ToolRegistry`. Every tool
call passes through :meth:`SafetyInterceptor.review` before it runs, in this
order:

1. **Kill switch** - if the run was stopped, nothing else matters.
2. **Classification** - safe, needs confirmation, or refused outright.
3. **Dry run** - show exactly what would have happened and stop.
4. **Confirmation** - ask, and treat anything but a clear yes as no.

Safe calls skip straight through with no prompt and no latency. That is not a
shortcut; it is what keeps the prompts meaningful when they do appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tools.base import Decision, Review, ToolSpec
from ..tracing import Trace
from .classify import Classification, Risk, classify
from .confirm import Confirmer, ConfirmRequest, DenyingConfirmer, summarise_call
from .journal import plan_undo
from .killswitch import KillSwitch
from .preview import preview


@dataclass
class SafetyStats:
    """What the interceptor did during a session."""

    reviewed: int = 0
    allowed: int = 0
    confirmed: int = 0
    refused: int = 0
    denied: int = 0
    dry_run: int = 0
    decisions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.reviewed} reviewed, {self.allowed} allowed, "
            f"{self.confirmed} confirmed, {self.refused} refused by you, "
            f"{self.denied} blocked"
        )


class SafetyInterceptor:
    """Classifies, gates and records every proposed action."""

    def __init__(
        self,
        *,
        confirmer: Confirmer | None = None,
        kill_switch: KillSwitch | None = None,
        journal: Any | None = None,
        trace: Trace | None = None,
        dry_run: bool = False,
        require_confirmation: bool = True,
        trash: Any | None = None,
        cwd: Any | None = None,
        adjudicator: Any | None = None,
    ) -> None:
        self.adjudicator = adjudicator
        self.confirmer = confirmer or DenyingConfirmer()
        self.kill_switch = kill_switch
        self.journal = journal
        self.trace = trace or Trace.disabled()
        self.dry_run = dry_run
        self.require_confirmation = require_confirmation
        self.trash = trash
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.stats = SafetyStats()
        #: Decisions already made this session, so the same command is not
        #: re-confirmed on every retry within one task.
        self._approved: set[str] = set()

    # -- the gate ----------------------------------------------------------

    def review(self, spec: ToolSpec, arguments: dict[str, Any]) -> Review:
        self.stats.reviewed += 1
        summary = summarise_call(spec.name, arguments)

        if self.kill_switch is not None and self.kill_switch.tripped:
            return self._record(spec, arguments, Decision.DENY, "run was stopped", None, summary)

        verdict = classify(spec.name, arguments, mutating=spec.mutating)

        # Third layer. Only consulted for commands no rule recognised, and it
        # can only clear them to SAFE - escalation stays with the rules.
        if self.adjudicator is not None and verdict.unmatched:
            verdict = self.adjudicator.review(summary, verdict)

        self.trace.event(
            "safety.classify",
            tool=spec.name,
            command=summary,
            risk=str(verdict.risk),
            reason=verdict.reason,
            source=verdict.source,
        )

        if verdict.risk is Risk.DENY:
            self.stats.denied += 1
            return self._record(
                spec,
                arguments,
                Decision.DENY,
                f"refused: {verdict.reason}. This is on the irreversible-damage list.",
                verdict,
                summary,
            )

        if verdict.risk is Risk.SAFE:
            self.stats.allowed += 1
            return Review(Decision.ALLOW, verdict.reason)

        # From here on the call needs confirmation.
        if self.dry_run:
            self.stats.dry_run += 1
            # Echoing the command back is not a preview - the user typed it.
            # What they cannot predict is what a glob matched.
            detail = (
                preview(str(arguments["command"]), self.cwd)
                if spec.name == "shell" and "command" in arguments
                else f"would run: {summary}"
            )
            return self._record(
                spec,
                arguments,
                Decision.DENY,
                f"dry run - {detail}",
                verdict,
                summary,
            )

        if not self.require_confirmation:
            self.stats.allowed += 1
            return Review(Decision.ALLOW, f"{verdict.reason} (confirmation disabled)")

        if summary in self._approved:
            self.stats.allowed += 1
            return Review(Decision.ALLOW, "already approved this session")

        request = ConfirmRequest(
            tool=spec.name,
            summary=summary,
            classification=verdict,
            undo_hint=self._undo_hint(spec.name, arguments),
            preview=(
                preview(str(arguments["command"]), self.cwd)
                if spec.name == "shell" and "command" in arguments
                else ""
            ),
        )
        with self.trace.span("safety.confirm", command=summary) as span:
            approved = self.confirmer.confirm(request)
            span["approved"] = approved

        if not approved:
            self.stats.refused += 1
            return self._record(
                spec,
                arguments,
                Decision.DENY,
                "you did not approve this. Tell the user you need permission, "
                "or do something else.",
                verdict,
                summary,
            )

        self.stats.confirmed += 1
        self._approved.add(summary)
        return Review(Decision.ALLOW, f"confirmed: {verdict.reason}")

    # -- helpers -----------------------------------------------------------

    def _undo_hint(self, tool: str, arguments: dict[str, Any]) -> str:
        """Tell the user up front whether this can be walked back.

        A delete that will be rerouted through the trash is recoverable, and
        saying so is the whole point of rerouting it - a user who believes a
        delete is permanent answers a different question than one who knows it
        is a move.
        """
        if self.trash is not None and tool == "shell":
            from .trash import parse_delete

            if parse_delete(str(arguments.get("command", "")), self.cwd) is not None:
                # Deliberately does not re-list the files: the preview above
                # already named them, and hearing the same list twice is how a
                # spoken prompt becomes something to sit through rather than
                # listen to.
                return "Nothing is deleted off the disk - victor undo puts it back."

        undo, why_not = plan_undo(tool, arguments, ok=True)
        if undo is not None:
            return f"If you change your mind I can {undo.description}."
        return f"This cannot be undone: {why_not}." if why_not else "This cannot be undone."

    def _record(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        decision: Decision,
        reason: str,
        verdict: Classification | None,
        summary: str,
    ) -> Review:
        """Journal a blocked call and return the review."""
        self.stats.decisions.append(f"{decision}: {summary}")
        if self.journal is not None:
            self.journal.record(
                spec.name,
                arguments,
                risk=verdict.risk if verdict else Risk.CONFIRM,
                decision=str(decision),
                reason=reason,
                ok=False,
            )
        return Review(decision, reason)

    def note_execution(
        self, spec: ToolSpec, arguments: dict[str, Any], *, result: Any
    ) -> None:
        """Record an action that actually ran. Called after execution.

        The result matters, not just its success: a delete that was rerouted
        through the trash carries the restore manifest in its metadata, and
        that is what makes the entry reversible.
        """
        if self.journal is None:
            return
        verdict = classify(spec.name, arguments, mutating=spec.mutating)
        if verdict.risk is Risk.SAFE:
            # Reads are not journalled; the journal is for things with effects.
            return
        entry = self.journal.record(
            spec.name,
            arguments,
            risk=verdict.risk,
            decision=str(Decision.ALLOW),
            reason=verdict.reason,
            ok=getattr(result, "ok", True),
            metadata=getattr(result, "metadata", None),
        )
        self.trace.event(
            "safety.journal",
            entry=entry.id,
            tool=spec.name,
            reversible=entry.reversible,
        )
