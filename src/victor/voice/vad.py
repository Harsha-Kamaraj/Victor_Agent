"""Voice activity detection and endpointing.

Two separable jobs, kept separate:

* a **VAD backend** answers "is this 20 ms frame speech?" - noisy, per-frame,
  wrong maybe 5% of the time;
* an **endpointer** turns that noisy stream into one clean utterance, using
  hysteresis so a single misclassified frame never starts or ends a recording.

The default backend is an adaptive energy detector written here rather than
``webrtcvad``. That package is unmaintained, imports ``pkg_resources``, and so
fails outright on setuptools 81+, which is not a dependency worth putting on
the critical path of a voice agent. WebRTC is still available as an opt-in
backend when it is installed and working.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from .audio import DEFAULT_FORMAT, AudioFormat, Segment, dbfs


@runtime_checkable
class VadBackend(Protocol):
    """Per-frame speech decision."""

    def is_speech(self, frame: np.ndarray) -> bool: ...

    def reset(self) -> None: ...


class EnergyVad:
    """Adaptive-threshold energy detector.

    Speech is declared when a frame sits ``threshold_db`` above the running
    estimate of the room's noise floor. The floor only adapts on frames judged
    *not* to be speech, so a long utterance cannot drag the threshold up over
    itself and cut the speaker off mid-sentence.

    ``absolute_floor_dbfs`` is the backstop: in a very quiet room the adaptive
    floor can sit near -80 dBFS, where fan noise clears the relative threshold.
    Nothing below the absolute floor is ever speech.
    """

    def __init__(
        self,
        fmt: AudioFormat = DEFAULT_FORMAT,
        *,
        threshold_db: float = 10.0,
        absolute_floor_dbfs: float = -48.0,
        max_noise_floor_dbfs: float = -35.0,
        adapt_rate: float = 0.05,
        calibration_frames: int = 8,
    ) -> None:
        self.fmt = fmt
        self.threshold_db = threshold_db
        self.absolute_floor_dbfs = absolute_floor_dbfs
        self.max_noise_floor_dbfs = max_noise_floor_dbfs
        self.adapt_rate = adapt_rate
        self.calibration_frames = calibration_frames
        self.noise_floor_dbfs: float | None = None
        self._seen = 0

    def reset(self) -> None:
        self.noise_floor_dbfs = None
        self._seen = 0

    @property
    def trigger_dbfs(self) -> float:
        """Level a frame must clear right now to count as speech."""
        floor = self.noise_floor_dbfs if self.noise_floor_dbfs is not None else -60.0
        # Clamp the floor: if calibration happened to hear speech - someone
        # talking as the mic opened, common in push-to-talk - an uncapped floor
        # sits at speech level and the VAD never triggers again. Deafness is a
        # far worse failure than a slightly trigger-happy threshold.
        floor = min(floor, self.max_noise_floor_dbfs)
        return max(floor + self.threshold_db, self.absolute_floor_dbfs)

    def is_speech(self, frame: np.ndarray) -> bool:
        level = dbfs(frame)
        self._seen += 1

        # Treat the opening frames as room tone regardless of content. Someone
        # who starts talking before the mic opens loses a few frames; someone
        # in a noisy room gets a threshold that fits the room.
        if self._seen <= self.calibration_frames:
            self.noise_floor_dbfs = (
                level
                if self.noise_floor_dbfs is None
                else min(self.noise_floor_dbfs, level)
            )
            return False

        speech = level > self.trigger_dbfs
        if not speech:
            assert self.noise_floor_dbfs is not None
            self.noise_floor_dbfs = (
                1 - self.adapt_rate
            ) * self.noise_floor_dbfs + self.adapt_rate * level
        return speech


class WebRtcVad:
    """Opt-in wrapper around Google's WebRTC VAD.

    Better than the energy detector in babble noise, worse at being installable.
    Requires 10/20/30 ms frames at 8, 16, 32 or 48 kHz.
    """

    VALID_RATES = (8_000, 16_000, 32_000, 48_000)
    VALID_FRAME_MS = (10, 20, 30)

    def __init__(
        self, fmt: AudioFormat = DEFAULT_FORMAT, frame_ms: int = 20, aggressiveness: int = 2
    ) -> None:
        if fmt.sample_rate not in self.VALID_RATES:
            raise ValueError(f"webrtcvad needs one of {self.VALID_RATES} Hz, got {fmt.sample_rate}")
        if frame_ms not in self.VALID_FRAME_MS:
            raise ValueError(f"webrtcvad needs a frame of {self.VALID_FRAME_MS} ms, got {frame_ms}")
        try:
            import webrtcvad
        except Exception as exc:  # ImportError, or pkg_resources blowing up
            raise RuntimeError(
                f"webrtcvad unavailable ({exc}). The energy backend is the default for "
                "exactly this reason."
            ) from exc
        self._vad = webrtcvad.Vad(aggressiveness)
        self.fmt = fmt
        self.frame_ms = frame_ms

    def reset(self) -> None:
        return None

    def is_speech(self, frame: np.ndarray) -> bool:
        return self._vad.is_speech(frame.astype(np.int16).tobytes(), self.fmt.sample_rate)


class State(StrEnum):
    WAITING = "waiting"
    SPEAKING = "speaking"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """Hysteresis and guard rails around one utterance."""

    start_ms: int = 150
    """Consecutive speech needed to open a recording. Rejects clicks and taps."""

    end_ms: int = 700
    """Trailing silence that ends it. Long enough to survive a mid-sentence pause."""

    preroll_ms: int = 300
    """Audio retained from *before* the trigger, so the first phoneme survives."""

    min_utterance_ms: int = 250
    """Anything shorter is discarded as a false trigger."""

    keep_tail_ms: int = 200
    """Trailing silence retained when the turn ends.

    Ending a turn requires ``end_ms`` of silence, but uploading all of it would
    bill the audio-seconds budget for the pause that proved the user stopped
    talking. A short tail is kept so the final consonant is not clipped.
    """

    max_utterance_ms: int = 20_000
    """Hard stop, so a stuck-open mic cannot bill the whole audio budget."""

    max_silence_ms: int = 8_000
    """Give up waiting for speech that never comes."""


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A completed utterance and how it ended."""

    segment: Segment
    reason: str
    speech_ms: float

    @property
    def truncated(self) -> bool:
        return self.reason == "max-duration"


class Endpointer:
    """Turns a frame stream into a single utterance.

    A pure state machine: feed it frames, get back an :class:`Endpoint` when the
    utterance closes. No audio hardware, no clock of its own - the caller
    supplies timestamps, which is what makes it testable frame by frame.
    """

    def __init__(
        self,
        backend: VadBackend | None = None,
        config: EndpointConfig | None = None,
        fmt: AudioFormat = DEFAULT_FORMAT,
        frame_ms: int = 20,
    ) -> None:
        self.fmt = fmt
        self.frame_ms = frame_ms
        self.config = config or EndpointConfig()
        self.backend = backend or EnergyVad(fmt)
        self.state = State.WAITING
        self._preroll: deque[tuple[np.ndarray, float]] = deque(
            maxlen=max(1, self.config.preroll_ms // frame_ms)
        )
        self._captured: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self._started_at: float | None = None
        self._waiting_frames = 0

    # -- helpers -----------------------------------------------------------

    def _frames_for(self, ms: int) -> int:
        return max(1, ms // self.frame_ms)

    def _finish(self, reason: str, ended_at: float) -> Endpoint | None:
        self.state = State.DONE
        captured = self._captured
        if reason == "silence" and self.config.end_ms > self.config.keep_tail_ms:
            # Drop the proof-of-silence tail; it is pure cost at the STT step.
            drop = self._frames_for(self.config.end_ms - self.config.keep_tail_ms)
            if len(captured) > drop:
                captured = captured[:-drop]
        samples = np.concatenate(captured) if captured else np.zeros(0, dtype=np.int16)
        duration_ms = self.fmt.duration_of(len(samples)) * 1000
        if duration_ms < self.config.min_utterance_ms:
            return None
        return Endpoint(
            segment=Segment(
                samples=samples,
                fmt=self.fmt,
                started_at=self._started_at or ended_at,
                ended_at=ended_at,
            ),
            reason=reason,
            speech_ms=self._speech_frames * self.frame_ms,
        )

    # -- the state machine -------------------------------------------------

    def feed(self, frame: np.ndarray, timestamp: float) -> Endpoint | None:
        """Consume one frame. Returns the utterance once it is complete."""
        if self.state is State.DONE:
            return None

        speech = self.backend.is_speech(frame)

        if self.state is State.WAITING:
            self._preroll.append((frame, timestamp))
            self._waiting_frames += 1
            self._speech_run = self._speech_run + 1 if speech else 0

            if self._speech_run >= self._frames_for(self.config.start_ms):
                self.state = State.SPEAKING
                # Rewind into the pre-roll so the utterance keeps its onset.
                self._captured = [f for f, _ in self._preroll]
                self._started_at = self._preroll[0][1]
                self._speech_frames = self._speech_run
                self._silence_run = 0
                self._preroll.clear()
            elif self._waiting_frames >= self._frames_for(self.config.max_silence_ms):
                return self._finish("no-speech", timestamp)
            return None

        # SPEAKING
        self._captured.append(frame)
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._frames_for(self.config.end_ms):
                return self._finish("silence", timestamp)

        captured_ms = len(self._captured) * self.frame_ms
        if captured_ms >= self.config.max_utterance_ms:
            return self._finish("max-duration", timestamp)
        return None

    def flush(self, timestamp: float) -> Endpoint | None:
        """Close out whatever has been captured. For when the source ends."""
        if self.state is State.SPEAKING:
            return self._finish("source-ended", timestamp)
        self.state = State.DONE
        return None
