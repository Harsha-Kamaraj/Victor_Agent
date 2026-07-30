"""Latency benchmarks.

The README commits to publishing measured numbers rather than aspirational
ones, so this reports percentiles from real runs on the machine it is invoked
on, and refuses to report a stage it could not actually measure.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings
from ..providers import Router
from ..quota import QuotaLedger
from .audio import DEFAULT_FORMAT, AudioFormat, Segment, silence, tone
from .sources import ArraySource
from .stt import Transcriber
from .tts import NullSynthesizer, PiperSynthesizer, Player, Synthesizer, speak
from .vad import EndpointConfig, Endpointer, EnergyVad

SAMPLE_TEXT = "Opening Settings and switching to dark mode."
MULTI_SENTENCE_TEXT = "Opening Settings. Switching to dark mode. Done."


@dataclass
class Measurement:
    """Timings for one stage, summarised as percentiles."""

    stage: str
    samples: list[float] = field(default_factory=list)
    unit: str = "ms"
    note: str = ""

    def add(self, value: float) -> None:
        self.samples.append(value)

    @property
    def runs(self) -> int:
        return len(self.samples)

    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def p95(self) -> float:
        if not self.samples:
            return float("nan")
        if len(self.samples) < 3:
            return max(self.samples)
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples) if self.samples else float("nan")


def bench_vad(runs: int = 5, seconds: float = 3.0, frame_ms: int = 20) -> Measurement:
    """How long endpointing takes per second of audio.

    Pure CPU, no I/O. Anything but a tiny number here means the VAD would be
    stealing time from the network calls that actually dominate the loop.
    """
    fmt = DEFAULT_FORMAT
    audio = np.concatenate(
        [silence(fmt, 0.4), tone(fmt, seconds, 220.0, 0.35), silence(fmt, 1.0)]
    )
    measurement = Measurement("vad endpointing", unit="ms/s audio")

    for _ in range(runs):
        endpointer = Endpointer(EnergyVad(fmt), EndpointConfig(), fmt, frame_ms)
        started = time.perf_counter()
        for frame, ts in ArraySource(audio, fmt, frame_ms).stream():
            if endpointer.feed(frame, ts) is not None:
                break
        elapsed_ms = (time.perf_counter() - started) * 1000
        measurement.add(elapsed_ms / (len(audio) / fmt.sample_rate))
    return measurement


def bench_tts(
    synthesizer: Synthesizer,
    *,
    runs: int = 5,
    text: str = SAMPLE_TEXT,
    label: str = "",
    playback: bool = False,
) -> tuple[Measurement, Measurement, Measurement]:
    """Time to first audio, full synthesis, and real-time factor.

    Piper emits one chunk per sentence, so for a single-sentence reply
    time-to-first-audio *is* the full synthesis time - streaming buys nothing.
    Benchmarking both a one- and a three-sentence line is what makes that
    visible instead of averaging it away.
    """
    player = Player(enabled=playback)
    suffix = f" ({label})" if label else ""
    ttfa = Measurement(f"tts time-to-first-audio{suffix}")
    synth = Measurement(f"tts synthesis{suffix}")
    rtf = Measurement(f"tts realtime factor{suffix}", unit="x")

    warm = getattr(synthesizer, "warm", None)
    if callable(warm):
        warm()  # exclude the one-time model load from every sample

    for _ in range(runs):
        stats = speak(synthesizer, player, text)
        ttfa.add(stats.ttfa_ms)
        synth.add(stats.synth_ms)
        rtf.add(stats.realtime_factor)
    return ttfa, synth, rtf


def bench_tts_warmup(synthesizer: Synthesizer) -> Measurement:
    """One-time model load. Paid once per session, so reported separately."""
    measurement = Measurement("tts model load (once)")
    warm = getattr(synthesizer, "warm", None)
    if callable(warm):
        measurement.add(warm() * 1000)
    else:
        measurement.note = "backend has no load step"
    return measurement


def bench_stt(
    settings: Settings,
    *,
    runs: int = 3,
    seconds: float = 3.0,
    segment: Segment | None = None,
) -> Measurement:
    """Round-trip latency to the STT provider.

    Uses a synthetic tone unless a real recording is supplied, so the number
    measures the network and the model's fixed cost, not recognition accuracy.
    Spends real audio-seconds from the free tier - hence the low default.
    """
    fmt = AudioFormat()
    audio = segment or Segment(tone(fmt, seconds, 180.0, 0.25), fmt)
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    router = Router(settings, ledger)
    transcriber = Transcriber(settings, router)
    measurement = Measurement("stt round trip")
    try:
        for _ in range(runs):
            transcript = transcriber.transcribe(audio)
            measurement.add(transcript.latency_ms)
    finally:
        transcriber.close()
    return measurement


def bench_pipeline(
    settings: Settings,
    *,
    runs: int = 3,
    synthesizer: Synthesizer | None = None,
    stt: bool = False,
) -> list[Measurement]:
    """Everything measurable on this machine, in one pass."""
    results = [bench_vad(runs=max(runs, 3))]

    # auto_download=False on purpose: a benchmark that fetches 63 MB is timing
    # the network, not the machine, and the download has no timeout to fall back
    # from - a stalled one hangs here indefinitely rather than raising.
    synth = synthesizer or PiperSynthesizer(settings.paths.models_dir, auto_download=False)
    try:
        results.append(bench_tts_warmup(synth))
        results.extend(bench_tts(synth, runs=runs, text=SAMPLE_TEXT, label="1 sentence"))
        results.extend(
            bench_tts(synth, runs=runs, text=MULTI_SENTENCE_TEXT, label="3 sentences")
        )
    except Exception as exc:
        fallback = NullSynthesizer()
        # The message, not just the type: VoiceModelMissing already names the
        # command that fixes it, and "(VoiceModelMissing)" alone does not.
        note = f"piper unavailable - {exc} - measured with the null backend"
        for m in bench_tts(fallback, runs=runs):
            m.note = note
            results.append(m)

    if stt:
        try:
            results.append(bench_stt(settings, runs=min(runs, 3)))
        except Exception as exc:
            results.append(
                Measurement("stt round trip", note=f"skipped: {type(exc).__name__}: {exc}")
            )
    else:
        results.append(
            Measurement("stt round trip", note="skipped - pass --stt to spend real audio quota")
        )
    return results


def summarise(measurements: Sequence[Measurement]) -> str:
    """Plain-text table. Used by the CLI and pasted into the README."""
    width = max([28, *(len(m.stage) for m in measurements)])
    rows = [f"{'stage':<{width}} {'runs':>5} {'p50':>10} {'p95':>10}  unit"]
    for m in measurements:
        if not m.samples:
            rows.append(f"{m.stage:<{width}} {'-':>5} {'-':>10} {'-':>10}  {m.note}")
            continue
        # A realtime factor near 0.03 rounds to nothing at one decimal place.
        places = 1 if m.unit.startswith("ms") else 3
        suffix = f"  {m.unit}" + (f"  ({m.note})" if m.note else "")
        rows.append(
            f"{m.stage:<{width}} {m.runs:>5} {m.p50:>10.{places}f} "
            f"{m.p95:>10.{places}f}{suffix}"
        )
    return "\n".join(rows)


def timed(fn: Callable[[], object]) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000
