"""Audio primitives.

Everything in the voice stack moves int16 mono PCM around at a declared sample
rate. Keeping one representation end to end avoids the class of bug where a
float array meant for playback is handed to an encoder expecting int16 and
comes out as noise.
"""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

import numpy as np

#: Whisper resamples everything to 16 kHz mono internally, so sending it
#: anything else just wastes upload bandwidth against the audio-seconds budget.
STT_SAMPLE_RATE = 16_000

FULL_SCALE = 32_768.0
SILENCE_DBFS = -100.0


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Shape of a PCM stream."""

    sample_rate: int = STT_SAMPLE_RATE
    channels: int = 1
    sample_width: int = 2  # int16

    def frame_samples(self, frame_ms: int) -> int:
        """Samples in a frame of ``frame_ms`` milliseconds."""
        return int(self.sample_rate * frame_ms / 1000)

    def duration_of(self, samples: int) -> float:
        return samples / self.sample_rate


DEFAULT_FORMAT = AudioFormat()


@dataclass(frozen=True, slots=True)
class Segment:
    """A captured stretch of audio, with the wall-clock window it came from."""

    samples: np.ndarray
    fmt: AudioFormat = DEFAULT_FORMAT
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def duration(self) -> float:
        return self.fmt.duration_of(len(self.samples))

    @property
    def peak_dbfs(self) -> float:
        return dbfs(self.samples)

    def to_wav(self) -> bytes:
        """Encode as a WAV container - the format every STT endpoint accepts."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.fmt.channels)
            wav.setsampwidth(self.fmt.sample_width)
            wav.setframerate(self.fmt.sample_rate)
            wav.writeframes(as_int16(self.samples).tobytes())
        return buffer.getvalue()

    def resampled(self, sample_rate: int) -> Segment:
        if sample_rate == self.fmt.sample_rate:
            return self
        return Segment(
            samples=resample(self.samples, self.fmt.sample_rate, sample_rate),
            fmt=AudioFormat(sample_rate, self.fmt.channels, self.fmt.sample_width),
            started_at=self.started_at,
            ended_at=self.ended_at,
        )

    @classmethod
    def from_wav(cls, data: bytes, *, started_at: float = 0.0) -> Segment:
        with wave.open(io.BytesIO(data), "rb") as wav:
            fmt = AudioFormat(wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
            raw = wav.readframes(wav.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        if fmt.channels > 1:
            samples = samples.reshape(-1, fmt.channels).mean(axis=1).astype(np.int16)
            fmt = AudioFormat(fmt.sample_rate, 1, fmt.sample_width)
        return cls(samples, fmt, started_at, started_at + fmt.duration_of(len(samples)))


def as_int16(samples: np.ndarray) -> np.ndarray:
    """Coerce to int16, scaling if handed the float form used for playback."""
    if samples.dtype == np.int16:
        return samples
    if np.issubdtype(samples.dtype, np.floating):
        return np.clip(samples * FULL_SCALE, -FULL_SCALE, FULL_SCALE - 1).astype(np.int16)
    return samples.astype(np.int16)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average interleaved channels down to one."""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1).astype(samples.dtype)


def dbfs(samples: np.ndarray) -> float:
    """RMS level in dBFS. Returns a floor value for digital silence."""
    if samples.size == 0:
        return SILENCE_DBFS
    audio = as_int16(samples).astype(np.float64)
    rms = math.sqrt(float(np.mean(audio * audio)))
    if rms <= 0.0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(rms / FULL_SCALE))


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resample.

    Deliberately not a windowed-sinc filter. Speech recognition is unbothered
    by the aliasing this introduces, and the alternative is a scipy dependency
    that would be the single largest thing in the install.
    """
    if source_rate == target_rate or samples.size == 0:
        return as_int16(samples)
    audio = as_int16(samples).astype(np.float64)
    count = int(round(len(audio) * target_rate / source_rate))
    if count <= 0:
        return np.zeros(0, dtype=np.int16)
    source_x = np.arange(len(audio), dtype=np.float64)
    target_x = np.linspace(0, len(audio) - 1, count, dtype=np.float64)
    return np.interp(target_x, source_x, audio).astype(np.int16)


def frames(samples: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    """Split into fixed-size frames, discarding any short tail."""
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    usable = (len(samples) // frame_samples) * frame_samples
    if usable == 0:
        return []
    return list(samples[:usable].reshape(-1, frame_samples))


def silence(fmt: AudioFormat, seconds: float) -> np.ndarray:
    return np.zeros(int(fmt.sample_rate * seconds), dtype=np.int16)


def tone(fmt: AudioFormat, seconds: float, hz: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    """A sine tone. Used by benchmarks and tests that need non-silent audio."""
    t = np.arange(int(fmt.sample_rate * seconds), dtype=np.float64) / fmt.sample_rate
    return as_int16(np.sin(2 * np.pi * hz * t) * amplitude)
