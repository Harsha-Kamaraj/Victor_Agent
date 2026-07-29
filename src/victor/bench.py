"""Turning recorded sessions into the numbers the README quotes.

The plan's rule for P8 is that every published figure is backed by a
measurement, and this is how that stays true as the code changes: the table is
regenerated from session traces rather than typed in. A number in the README
that nobody can reproduce is a claim; a number that falls out of
``victor bench --traces`` is a measurement.

**Percentiles, not averages.** An average latency hides the shape of the thing
it summarises - one eight-second tree walk and nineteen fast ones average to
something that never happened. p50 says what a run feels like and p95 says what
it feels like when it is bad, and those are the two facts worth publishing.

**Sample counts travel with the numbers.** A p95 over three observations is not
a p95, and a table that does not say so invites its reader to believe it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .tracing import list_traces, read_trace

#: Trace event kinds worth publishing, and what to call them in the table.
#: An allowlist rather than everything, because a table of forty rows is not a
#: benchmark, it is a log.
STAGES: dict[str, str] = {
    "stt.transcribe": "speech to text",
    "tts.synthesize": "speech synthesis",
    "llm.complete": "model reply",
    "tool.run": "tool call",
    "uia.snapshot": "read the screen",
    "vision.locate": "vision fallback",
    "memory.recall": "recall a past fix",
    "safety.classify": "safety check",
    "agent.run": "whole task",
}


@dataclass(frozen=True, slots=True)
class Stage:
    """One measured stage across every session that recorded it."""

    name: str
    samples: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def p50(self) -> float:
        return median(self.samples)

    @property
    def p95(self) -> float:
        """The 95th percentile, or the worst sample when there are too few.

        Reported honestly rather than interpolated: with six observations there
        is no 95th percentile, and the largest one is the most defensible thing
        to print beside a count that says how much to trust it.
        """
        ordered = sorted(self.samples)
        if len(ordered) < 20:
            return ordered[-1]
        return ordered[int(len(ordered) * 0.95) - 1]

    @property
    def trustworthy(self) -> bool:
        return self.count >= 20


@dataclass(frozen=True, slots=True)
class TraceReport:
    """What a set of sessions measured, and what they cost."""

    stages: tuple[Stage, ...]
    sessions: int
    api_calls: int
    free_tool_calls: int
    billed_tool_calls: int

    @property
    def zero_cost_ratio(self) -> float:
        total = self.free_tool_calls + self.billed_tool_calls
        return 1.0 if not total else self.free_tool_calls / total

    def summary(self) -> str:
        if not self.sessions:
            return "no sessions recorded yet - run `victor do` or `victor uia` first"
        total = self.free_tool_calls + self.billed_tool_calls
        return (
            f"{self.sessions} sessions, {self.api_calls} API calls, "
            f"{self.free_tool_calls}/{total} tool calls free "
            f"({self.zero_cost_ratio * 100:.0f}%)"
        )


def collect(traces_dir: Path, *, limit: int = 200) -> TraceReport:
    """Read recorded sessions and summarise every stage they timed."""
    samples: dict[str, list[float]] = defaultdict(list)
    sessions = api_calls = free_calls = billed_calls = 0

    for path in list_traces(traces_dir, limit=limit):
        try:
            events = read_trace(path)
        except OSError:
            continue
        sessions += 1
        for event in events:
            kind = str(event.get("kind", ""))
            payload = event.get("payload") or {}
            duration = event.get("duration_ms")

            label = _label_for(kind)
            if label and isinstance(duration, int | float):
                samples[label].append(float(duration))

            if kind.startswith(("llm.complete", "stt.transcribe", "vision.locate")):
                api_calls += 1
            if kind.startswith("tool.run"):
                cost = payload.get("cost")
                if cost:
                    billed_calls += 1
                else:
                    free_calls += 1

    stages = tuple(
        Stage(name, tuple(values))
        for name, values in sorted(samples.items(), key=lambda kv: -len(kv[1]))
        if values
    )
    return TraceReport(stages, sessions, api_calls, free_calls, billed_calls)


def _label_for(kind: str) -> str:
    for prefix, label in STAGES.items():
        if kind.startswith(prefix):
            return label
    return ""
