"""Session tracing.

Every run writes one JSONL file. Each line is a self-contained event, so a
trace stays readable even if the process is killed mid-task - which, for an
agent with a kill switch, is a supported way to end a session rather than an
edge case.

The format is deliberately boring: append-only, one JSON object per line, no
nesting beyond the payload. `victor trace show` renders it; `jq` reads it; a
future eval harness can diff two runs without a parser.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(slots=True)
class Event:
    """One line of a trace."""

    seq: int
    ts: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"seq": self.seq, "ts": self.ts, "kind": self.kind}
        if self.duration_ms is not None:
            out["duration_ms"] = round(self.duration_ms, 2)
        if self.payload:
            out["payload"] = self.payload
        return out


class Trace:
    """Append-only event log for a single session.

    Use as a context manager so ``session.end`` is always written::

        with Trace.open(paths, label="voice") as trace:
            trace.event("heard", text="open notepad")
    """

    def __init__(self, session_id: str, path: Path | None, *, label: str = "") -> None:
        self.session_id = session_id
        self.path = path
        self.label = label
        self._seq = 0
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    # -- construction ------------------------------------------------------

    @classmethod
    def open(cls, traces_dir: Path, *, label: str = "") -> Trace:
        """Start a trace in ``traces_dir`` with a sortable, unique filename."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        session_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        trace = cls(session_id, traces_dir / f"{session_id}.jsonl", label=label)
        trace.event("session.start", label=label, pid=os.getpid())
        return trace

    @classmethod
    def disabled(cls) -> Trace:
        """A trace that records nothing. Keeps call sites free of `if trace:`."""
        return cls(session_id="disabled", path=None)

    # -- writing -----------------------------------------------------------

    def event(self, kind: str, /, duration_ms: float | None = None, **payload: Any) -> Event:
        """Record one event. Returns it, mostly so tests can assert on it."""
        with self._lock:
            self._seq += 1
            evt = Event(
                seq=self._seq,
                ts=_now_iso(),
                kind=kind,
                payload={k: v for k, v in payload.items() if v is not None},
                duration_ms=duration_ms,
            )
            if self._fh is not None:
                self._fh.write(json.dumps(evt.to_json(), default=str) + "\n")
                self._fh.flush()
            return evt

    @contextmanager
    def span(self, kind: str, /, **payload: Any) -> Iterator[dict[str, Any]]:
        """Time a block and emit ``kind`` once it finishes, pass or fail.

        The yielded dict is merged into the final event, so a block can attach
        results it only learns about partway through::

            with trace.span("llm.call", model=spec.key) as sp:
                reply = client.chat(...)
                sp["tokens"] = reply.usage.total_tokens
        """
        extra: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            yield extra
        except BaseException as exc:
            self.event(
                kind,
                duration_ms=(time.perf_counter() - started) * 1000,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                **{**payload, **extra},
            )
            raise
        else:
            self.event(
                kind,
                duration_ms=(time.perf_counter() - started) * 1000,
                status="ok",
                **{**payload, **extra},
            )

    def selection(self, selection: Any) -> None:
        """Record a router decision, including what it passed over."""
        self.event(
            "router.select",
            workload=str(selection.workload),
            model=selection.key,
            rejected=[{"model": k, "reason": why} for k, why in selection.rejected] or None,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self, status: str = "ok", **payload: Any) -> None:
        if self._fh is None:
            return
        self.event(
            "session.end",
            duration_ms=(time.perf_counter() - self._started) * 1000,
            status=status,
            **payload,
        )
        self._fh.close()
        self._fh = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc_type is None:
            self.close("ok")
        else:
            self.close("error", error=f"{exc_type.__name__}: {exc}")


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Parse a trace file, skipping any line a crash left half-written."""
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def list_traces(traces_dir: Path, limit: int = 20) -> list[Path]:
    """Most recent traces first."""
    if not traces_dir.exists():
        return []
    return sorted(traces_dir.glob("*.jsonl"), reverse=True)[:limit]
