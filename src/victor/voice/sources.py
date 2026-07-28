"""Where captured audio comes from.

The pipeline consumes an iterator of fixed-size frames and does not care
whether they arrive from a microphone, a WAV file or an array built in a test.
That is the whole reason endpointing and transcription can be tested on a
machine with no audio hardware.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import VictorError
from .audio import DEFAULT_FORMAT, AudioFormat, as_int16, frames, to_mono


class AudioDeviceError(VictorError):
    """The microphone could not be opened."""


@runtime_checkable
class AudioSource(Protocol):
    """A stream of fixed-size int16 mono frames."""

    fmt: AudioFormat
    frame_ms: int

    def stream(self) -> Iterator[tuple[np.ndarray, float]]:
        """Yield ``(frame, timestamp)`` until the source is exhausted."""
        ...


class ArraySource:
    """Replays a fixed array. The test and benchmark workhorse."""

    def __init__(
        self,
        samples: np.ndarray,
        fmt: AudioFormat = DEFAULT_FORMAT,
        frame_ms: int = 20,
        *,
        realtime: bool = False,
        start_time: float = 0.0,
    ) -> None:
        self.fmt = fmt
        self.frame_ms = frame_ms
        self._samples = as_int16(samples)
        self._realtime = realtime
        self._start = start_time

    def stream(self) -> Iterator[tuple[np.ndarray, float]]:
        size = self.fmt.frame_samples(self.frame_ms)
        step = self.frame_ms / 1000
        for i, frame in enumerate(frames(self._samples, size)):
            if self._realtime:
                time.sleep(step)
            yield frame, self._start + i * step


class MicrophoneSource:
    """Live capture via PortAudio.

    Frames are pulled off a queue filled by PortAudio's own thread, so a slow
    consumer shows up as a reported overflow rather than silently dropped
    audio in the middle of an utterance.
    """

    def __init__(
        self,
        fmt: AudioFormat = DEFAULT_FORMAT,
        frame_ms: int = 20,
        *,
        device: int | str | None = None,
        max_seconds: float = 60.0,
    ) -> None:
        self.fmt = fmt
        self.frame_ms = frame_ms
        self.device = device
        self.max_seconds = max_seconds
        self.overflows = 0

    def stream(self) -> Iterator[tuple[np.ndarray, float]]:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:  # OSError: PortAudio missing
            raise AudioDeviceError(
                f"audio capture unavailable ({exc}). Install with: pip install -e '.[voice]'"
            ) from exc

        size = self.fmt.frame_samples(self.frame_ms)
        pending: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time, status) -> None:
            if status and status.input_overflow:
                self.overflows += 1
            pending.put(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=self.fmt.sample_rate,
                blocksize=size,
                device=self.device,
                channels=self.fmt.channels,
                dtype="int16",
                callback=callback,
            )
        except Exception as exc:
            raise AudioDeviceError(f"could not open microphone: {exc}") from exc

        deadline = time.monotonic() + self.max_seconds
        with stream:
            while time.monotonic() < deadline:
                try:
                    block = pending.get(timeout=0.5)
                except queue.Empty:
                    continue
                yield to_mono(np.asarray(block, dtype=np.int16)), time.monotonic()


def list_devices() -> list[dict[str, object]]:
    """Available audio devices, or an empty list if PortAudio is unavailable."""
    try:
        import sounddevice as sd

        default_in, default_out = sd.default.device
        return [
            {
                "index": i,
                "name": d["name"],
                "inputs": d["max_input_channels"],
                "outputs": d["max_output_channels"],
                "default_input": i == default_in,
                "default_output": i == default_out,
            }
            for i, d in enumerate(sd.query_devices())
        ]
    except Exception:
        return []


def has_input_device() -> bool:
    return any(d["inputs"] for d in list_devices())


def has_output_device() -> bool:
    return any(d["outputs"] for d in list_devices())
