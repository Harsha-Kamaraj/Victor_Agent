"""The voice loop: microphone in, speech out.

P1 stops at the edges of the agent. :meth:`VoicePipeline.listen` returns a
transcript and :meth:`VoicePipeline.speak` says a string; what goes between
them is P2's job. Keeping that seam explicit means the voice stack can be
benchmarked and demoed on its own, before there is any agent to blame.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ..config import Settings
from ..errors import VictorError
from ..providers import Router
from ..quota import QuotaLedger
from ..tracing import Trace
from .audio import DEFAULT_FORMAT, AudioFormat, Segment
from .sources import AudioSource, MicrophoneSource
from .stt import Transcriber, Transcript
from .tts import Player, SpeechStats, Synthesizer, build_synthesizer, speak
from .vad import Endpoint, EndpointConfig, Endpointer, EnergyVad


class ListenMode(StrEnum):
    VAD = "vad"
    """Record until the endpointer decides you stopped talking."""

    PTT = "ptt"
    """Record between two Enter presses. Immune to a bad noise floor."""

    FIXED = "fixed"
    """Record a fixed number of seconds. For scripted benchmarks."""


@dataclass(frozen=True, slots=True)
class Turn:
    """One round trip: what was heard, and what it cost."""

    transcript: Transcript
    endpoint: Endpoint
    capture_ms: float

    @property
    def text(self) -> str:
        return self.transcript.text


class NoSpeechDetected(VictorError):
    """The endpointer never heard an utterance."""

    exit_code = 4


class VoicePipeline:
    """Composes capture, endpointing, transcription and speech."""

    def __init__(
        self,
        settings: Settings,
        router: Router,
        *,
        transcriber: Transcriber | None = None,
        synthesizer: Synthesizer | None = None,
        player: Player | None = None,
        source_factory: Callable[[], AudioSource] | None = None,
        fmt: AudioFormat = DEFAULT_FORMAT,
        frame_ms: int = 20,
        endpoint_config: EndpointConfig | None = None,
        trace: Trace | None = None,
    ) -> None:
        self.settings = settings
        self.router = router
        self.fmt = fmt
        self.frame_ms = frame_ms
        self.endpoint_config = endpoint_config or EndpointConfig()
        self.trace = trace or Trace.disabled()
        # The mic stays open a little past the longest allowed utterance, so a
        # hard stop always comes from the endpointer rather than the device.
        mic_seconds = self.endpoint_config.max_utterance_ms / 1000 + 10
        self._source_factory = source_factory or (
            lambda: MicrophoneSource(fmt, frame_ms, max_seconds=mic_seconds)
        )
        self.transcriber = transcriber or Transcriber(settings, router, trace=self.trace)
        self.synthesizer = synthesizer or build_synthesizer(settings.paths.models_dir)
        self.player = player or Player()

    # -- listening ---------------------------------------------------------

    def capture(
        self,
        mode: ListenMode = ListenMode.VAD,
        *,
        seconds: float = 5.0,
        on_level: Callable[[float, bool], None] | None = None,
    ) -> Endpoint:
        """Record one utterance and return it, without transcribing."""
        source = self._source_factory()
        started = time.perf_counter()

        with self.trace.span("voice.capture", mode=str(mode)) as span:
            if mode is ListenMode.PTT:
                endpoint = self._capture_ptt(source, on_level=on_level)
            elif mode is ListenMode.FIXED:
                endpoint = self._capture_fixed(source, seconds, on_level=on_level)
            else:
                endpoint = self._capture_vad(source, on_level=on_level)
            span["reason"] = endpoint.reason
            span["audio_seconds"] = round(endpoint.segment.duration, 2)

        self._last_capture_ms = (time.perf_counter() - started) * 1000
        return endpoint

    def _capture_vad(
        self, source: AudioSource, *, on_level: Callable[[float, bool], None] | None
    ) -> Endpoint:
        endpointer = Endpointer(
            EnergyVad(self.fmt), self.endpoint_config, self.fmt, self.frame_ms
        )
        last_ts = 0.0
        for frame, ts in source.stream():
            last_ts = ts
            result = endpointer.feed(frame, ts)
            if on_level is not None:
                on_level(_level(frame), endpointer.state.value == "speaking")
            if result is not None:
                return result
        result = endpointer.flush(last_ts)
        if result is None:
            raise NoSpeechDetected("no speech detected")
        return result

    def _capture_fixed(
        self,
        source: AudioSource,
        seconds: float,
        *,
        on_level: Callable[[float, bool], None] | None,
    ) -> Endpoint:
        collected: list[np.ndarray] = []
        wanted = int(seconds * 1000 / self.frame_ms)
        first_ts = last_ts = 0.0
        for i, (frame, ts) in enumerate(source.stream()):
            if i == 0:
                first_ts = ts
            last_ts = ts
            collected.append(frame)
            if on_level is not None:
                on_level(_level(frame), True)
            if len(collected) >= wanted:
                break
        return _endpoint_from(collected, self.fmt, first_ts, last_ts, "fixed")

    def _capture_ptt(
        self, source: AudioSource, *, on_level: Callable[[float, bool], None] | None
    ) -> Endpoint:
        """Record until Enter is pressed.

        A terminal key, not a global hotkey: a system-wide push-to-talk binding
        needs an OS-level hook and belongs with the HUD in P8. This is the
        honest version that works today.
        """
        import threading

        stop = threading.Event()

        def wait_for_key() -> None:
            # A closed or redirected stdin just means "stop now", not an error.
            with contextlib.suppress(OSError, ValueError):
                sys.stdin.readline()
            stop.set()

        threading.Thread(target=wait_for_key, daemon=True).start()

        collected: list[np.ndarray] = []
        first_ts = last_ts = 0.0
        for i, (frame, ts) in enumerate(source.stream()):
            if i == 0:
                first_ts = ts
            last_ts = ts
            collected.append(frame)
            if on_level is not None:
                on_level(_level(frame), True)
            if stop.is_set():
                break
        return _endpoint_from(collected, self.fmt, first_ts, last_ts, "key-release")

    def listen(
        self,
        mode: ListenMode = ListenMode.VAD,
        *,
        seconds: float = 5.0,
        language: str | None = "en",
        prompt: str | None = None,
        on_level: Callable[[float, bool], None] | None = None,
    ) -> Turn:
        """Capture one utterance and transcribe it."""
        endpoint = self.capture(mode, seconds=seconds, on_level=on_level)
        transcript = self.transcriber.transcribe(
            endpoint.segment, language=language, prompt=prompt
        )
        self.trace.event(
            "voice.heard",
            text=transcript.text,
            audio_seconds=round(transcript.audio_seconds, 2),
            stt_ms=round(transcript.latency_ms),
        )
        return Turn(
            transcript=transcript,
            endpoint=endpoint,
            capture_ms=getattr(self, "_last_capture_ms", 0.0),
        )

    # -- speaking ----------------------------------------------------------

    def speak(self, text: str) -> SpeechStats:
        """Say something out loud."""
        with self.trace.span("voice.speak", chars=len(text)) as span:
            stats = speak(self.synthesizer, self.player, text)
            span["ttfa_ms"] = round(stats.ttfa_ms)
            span["backend"] = stats.backend
        return stats

    def warm(self) -> float:
        """Pre-load the synthesis model so the first reply is not the slowest."""
        warm = getattr(self.synthesizer, "warm", None)
        return warm() if callable(warm) else 0.0

    def close(self) -> None:
        self.transcriber.close()


def _level(frame: np.ndarray) -> float:
    from .audio import dbfs

    return dbfs(frame)


def _endpoint_from(
    collected: list[np.ndarray],
    fmt: AudioFormat,
    started_at: float,
    ended_at: float,
    reason: str,
) -> Endpoint:
    samples = np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)
    segment = Segment(samples, fmt, started_at, ended_at)
    return Endpoint(segment=segment, reason=reason, speech_ms=segment.duration * 1000)


def build_pipeline(
    settings: Settings,
    *,
    trace: Trace | None = None,
    tts_backend: str = "piper",
    playback: bool = True,
    source_factory: Callable[[], AudioSource] | None = None,
) -> VoicePipeline:
    """Wire a pipeline with the standard components."""
    trace = trace or Trace.disabled()
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    router = Router(settings, ledger, on_select=trace.selection)
    return VoicePipeline(
        settings,
        router,
        synthesizer=build_synthesizer(settings.paths.models_dir, prefer=tts_backend),
        player=Player(enabled=playback),
        source_factory=source_factory,
        trace=trace,
    )


def frames_of(segment: Segment, frame_ms: int) -> Iterator[np.ndarray]:
    """Re-frame a segment. Convenience for feeding a recording to a VAD."""
    from .audio import frames

    yield from frames(segment.samples, segment.fmt.frame_samples(frame_ms))
