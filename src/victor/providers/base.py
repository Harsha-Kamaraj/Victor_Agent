"""Core types for the routing layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..quota import QuotaLimits, QuotaStatus


class Workload(StrEnum):
    """A kind of work, independent of who ends up doing it.

    Callers ask for a workload; the router decides the provider. That
    indirection is the whole point - vision is scarce, text is not, and the
    agent loop should not have to know which is which.
    """

    TEXT = "text"
    """ReAct reasoning and tool selection. The hot path."""

    VISION = "vision"
    """Screenshot understanding. Fallback only - the UIA tree comes first."""

    STT = "stt"
    """Speech to text."""

    TTS = "tts"
    """Text to speech."""

    EMBEDDING = "embedding"
    """Vectors for the memory index."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One concrete model, its free allowance, and what it needs to run."""

    provider: str
    model: str
    workloads: tuple[Workload, ...]
    limits: QuotaLimits
    credential: str | None = None
    """Name of the :class:`~victor.config.Settings` field holding its key."""
    local: bool = False
    """Runs on this machine. No key, no network, no quota."""
    notes: str = ""

    @property
    def key(self) -> str:
        """Stable identifier, also the quota ledger key: ``groq:model-name``."""
        return f"{self.provider}:{self.model}"

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True, slots=True)
class Selection:
    """The router's decision, plus why the better options were passed over.

    ``rejected`` is what makes a "why did it use the slow model?" question
    answerable after the fact - it lands verbatim in the session trace.
    """

    workload: Workload
    spec: ModelSpec
    status: QuotaStatus
    rejected: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        return self.spec.key

    def explain(self) -> str:
        if not self.rejected:
            return f"{self.workload} -> {self.key}"
        skipped = "; ".join(f"{k} ({why})" for k, why in self.rejected)
        return f"{self.workload} -> {self.key} (skipped: {skipped})"


__all__ = ["ModelSpec", "QuotaLimits", "Selection", "Workload"]
