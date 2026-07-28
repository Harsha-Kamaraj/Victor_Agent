"""Filesystem layout for Victor's local state.

One place that knows where things live, so no other module has to hardcode a
path. Everything sits under a single data directory (``~/.victor`` by default)
which can be relocated with ``VICTOR_DATA_DIR`` - handy for tests, which point
it at a tmpdir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved locations of Victor's on-disk state."""

    root: Path

    @property
    def quota_file(self) -> Path:
        """Rolling record of free-tier consumption per provider/model."""
        return self.root / "quota.json"

    @property
    def limits_file(self) -> Path:
        """Per-model free-tier overrides.

        The plan requires every limit to be config rather than hardcoded,
        because providers change them without notice. The built-in table is a
        conservative default; this file wins where it disagrees.
        """
        return self.root / "limits.json"

    @property
    def trash_dir(self) -> Path:
        """Where deleted files wait, so `victor undo` can put them back."""
        return self.root / "trash"

    @property
    def traces_dir(self) -> Path:
        """One JSONL file per session, appended as the agent runs."""
        return self.root / "traces"

    @property
    def journal_file(self) -> Path:
        """P3: append-only log of executed actions and their undo recipes."""
        return self.root / "journal.jsonl"

    @property
    def memory_dir(self) -> Path:
        """P6: FAISS index plus the payload store it points into."""
        return self.root / "memory"

    @property
    def models_dir(self) -> Path:
        """Downloaded ONNX artefacts (Piper voices, fastembed encoders)."""
        return self.root / "models"

    def ensure(self) -> Paths:
        """Create every directory Victor writes to. Idempotent."""
        for d in (self.root, self.traces_dir, self.memory_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
