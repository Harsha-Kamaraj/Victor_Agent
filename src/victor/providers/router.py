"""Split-brain routing.

The router is the only component that decides *who* runs a piece of work. It
answers one question - "given this workload, what can I afford right now?" -
by walking the workload's chain in preference order and taking the first model
that has a credential configured and free allowance remaining.

It deliberately does not make API calls. Keeping selection pure means the
routing policy is unit-testable without a network, and the provider clients
built in later phases stay thin.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from ..errors import NoProviderAvailable
from ..quota import QuotaLedger, QuotaStatus
from .base import ModelSpec, Selection, Workload
from .registry import ROUTING_TABLE, spec_by_id

#: Settings field holding a user override, per workload.
_OVERRIDE_FIELD: dict[Workload, str] = {
    Workload.TEXT: "text_model",
    Workload.VISION: "vision_model",
    Workload.STT: "stt_model",
}


class Router:
    """Chooses a model per workload, subject to credentials and quota."""

    def __init__(
        self,
        settings: Settings,
        ledger: QuotaLedger,
        table: dict[Workload, tuple[ModelSpec, ...]] | None = None,
        *,
        on_select: Callable[[Selection], None] | None = None,
    ) -> None:
        self._settings = settings
        self._ledger = ledger
        self._table = table if table is not None else ROUTING_TABLE
        self._on_select = on_select

    # -- chain construction ------------------------------------------------

    def chain(self, workload: Workload) -> tuple[ModelSpec, ...]:
        """Candidates for ``workload``, best first, honouring any override.

        An override does not remove the rest of the chain - it promotes the
        named model to the front. A user who pins a model still gets a working
        agent when that model's daily allowance runs out.
        """
        specs = self._table.get(workload, ())
        field = _OVERRIDE_FIELD.get(workload)
        if field is None:
            return specs
        override = getattr(self._settings, field, None)
        if not override:
            return specs
        pinned = spec_by_id(override)
        if pinned is None or workload not in pinned.workloads:
            # Unknown or mismatched override: ignore rather than fail hard, so
            # a typo in .env degrades to default behaviour.
            return specs
        return (pinned, *(s for s in specs if s.key != pinned.key))

    # -- availability ------------------------------------------------------

    def _availability(
        self, spec: ModelSpec, *, tokens: int, audio_seconds: float
    ) -> QuotaStatus:
        """Why this model can or cannot serve a call of the given size."""
        if spec.local:
            return QuotaStatus(key=spec.key, allowed=True)
        if spec.credential and not self._settings.has(spec.credential):
            env_name = spec.credential.upper()
            return QuotaStatus(key=spec.key, allowed=False, reason=f"{env_name} not set")
        status = self._ledger.check(
            spec.key, spec.limits, tokens=tokens, audio_seconds=audio_seconds
        )
        if not status.allowed and not self._settings.strict_free_tier:
            # The user has explicitly opted out of the free-tier guarantee, so
            # an exhausted allowance becomes a warning rather than a veto.
            return QuotaStatus(
                key=spec.key,
                allowed=True,
                reason=f"over free tier ({status.reason}) - billing may apply",
                requests_remaining=status.requests_remaining,
                tokens_remaining=status.tokens_remaining,
                audio_seconds_remaining=status.audio_seconds_remaining,
            )
        return status

    def candidates(
        self, workload: Workload, *, tokens: int = 0, audio_seconds: float = 0.0
    ) -> list[tuple[ModelSpec, QuotaStatus]]:
        """Every candidate with its availability. Used by ``victor quota``."""
        return [
            (spec, self._availability(spec, tokens=tokens, audio_seconds=audio_seconds))
            for spec in self.chain(workload)
        ]

    # -- selection ---------------------------------------------------------

    def select(
        self, workload: Workload, *, tokens: int = 0, audio_seconds: float = 0.0
    ) -> Selection:
        """Pick a model, or raise :class:`NoProviderAvailable`.

        ``tokens``/``audio_seconds`` are an estimate of the call's size; they
        let the router skip a model that has requests left but not enough
        token headroom this minute.
        """
        rejected: list[tuple[str, str]] = []
        for spec in self.chain(workload):
            status = self._availability(spec, tokens=tokens, audio_seconds=audio_seconds)
            if status.allowed:
                selection = Selection(
                    workload=workload,
                    spec=spec,
                    status=status,
                    rejected=tuple(rejected),
                )
                if self._on_select is not None:
                    self._on_select(selection)
                return selection
            rejected.append((spec.key, status.reason or "unavailable"))

        raise NoProviderAvailable(
            workload.value, [f"{key}: {why}" for key, why in rejected] or ["no candidates"]
        )

    # -- accounting --------------------------------------------------------

    def record(
        self,
        selection: Selection,
        *,
        requests: int = 1,
        tokens: int = 0,
        audio_seconds: float = 0.0,
    ) -> None:
        """Charge a call against the selected model's allowance.

        Local models are free by definition and are not tracked - keeping them
        out of the ledger means the ledger file only ever contains things that
        can actually run out.
        """
        if selection.spec.local:
            return
        self._ledger.record(
            selection.spec.key,
            selection.spec.limits,
            requests=requests,
            tokens=tokens,
            audio_seconds=audio_seconds,
        )

    def reconcile(self, selection: Selection, *, tokens: int) -> None:
        """Fold real token usage into a request already recorded."""
        if selection.spec.local or not tokens:
            return
        self._ledger.record(
            selection.spec.key, selection.spec.limits, requests=0, tokens=tokens
        )
