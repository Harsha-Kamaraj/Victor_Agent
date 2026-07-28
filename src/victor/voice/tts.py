"""Text to speech, and getting it out of the speakers.

Synthesis is streamed rather than buffered. Piper emits audio in chunks as it
works through a sentence, and the player starts writing the first chunk to the
output device while later ones are still being generated. On a 3-second reply
that is the difference between ~450 ms and ~150 ms before the user hears
anything, which is most of the perceived responsiveness of the whole agent.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import VictorError
from .audio import AudioFormat, Segment, as_int16, silence

DEFAULT_VOICE = "en_US-lessac-medium"


class VoiceModelMissing(VictorError):
    """The Piper voice has not been downloaded yet."""

    exit_code = 2


@runtime_checkable
class Synthesizer(Protocol):
    """Turns text into a stream of int16 PCM chunks."""

    sample_rate: int
    name: str

    def stream(self, text: str) -> Iterator[np.ndarray]: ...


class PiperSynthesizer:
    """Local ONNX synthesis. No network, no quota, no per-call cost.

    The voice model is ~63 MB and downloaded once. Loading it takes about a
    second, so the instance is meant to be created once and reused - the
    pipeline holds onto it for the life of the session.
    """

    name = "piper"

    def __init__(
        self,
        models_dir: Path,
        voice: str = DEFAULT_VOICE,
        *,
        auto_download: bool = True,
    ) -> None:
        self.models_dir = models_dir
        self.voice = voice
        self.auto_download = auto_download
        self._voice_obj = None
        self.sample_rate = 22_050  # corrected from the model on first synthesis

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.voice}.onnx"

    @property
    def installed(self) -> bool:
        return self.model_path.exists() and self.model_path.with_suffix(".onnx.json").exists()

    def ensure_installed(self, *, force: bool = False) -> Path:
        """Download the voice if it is not already on disk."""
        if self.installed and not force:
            return self.model_path
        try:
            from piper.download_voices import download_voice
        except ImportError as exc:
            raise VoiceModelMissing(
                f"piper-tts is not installed ({exc}). pip install -e '.[voice]'"
            ) from exc
        self.models_dir.mkdir(parents=True, exist_ok=True)
        download_voice(self.voice, self.models_dir, force_redownload=force)
        return self.model_path

    def _load(self):
        if self._voice_obj is not None:
            return self._voice_obj
        if not self.installed:
            if not self.auto_download:
                raise VoiceModelMissing(
                    f"voice {self.voice!r} not downloaded. Run: victor voice install"
                )
            self.ensure_installed()
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise VoiceModelMissing(
                f"piper-tts is not installed ({exc}). pip install -e '.[voice]'"
            ) from exc
        self._voice_obj = PiperVoice.load(self.model_path)
        return self._voice_obj

    def warm(self) -> float:
        """Pay the model-load cost up front. Returns seconds spent."""
        started = time.perf_counter()
        self._load()
        return time.perf_counter() - started

    def stream(self, text: str) -> Iterator[np.ndarray]:
        voice = self._load()
        for chunk in voice.synthesize(text):
            self.sample_rate = chunk.sample_rate
            yield as_int16(chunk.audio_int16_array)


class SystemSynthesizer:
    """The OS's built-in voice - macOS ``say``, Windows SAPI.

    A development convenience for machines without the Piper model, and a
    genuine fallback if the download fails. Not the shipped path: quality is
    worse and it cannot stream, so time-to-first-audio is the full synthesis
    time.
    """

    name = "system"

    def __init__(self, sample_rate: int = 22_050) -> None:
        self.sample_rate = sample_rate

    @staticmethod
    def available() -> bool:
        system = platform.system()
        if system == "Darwin":
            return shutil.which("say") is not None
        if system == "Windows":
            return shutil.which("powershell") is not None
        return False

    def stream(self, text: str) -> Iterator[np.ndarray]:
        system = platform.system()
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "speech.wav"
            if system == "Darwin":
                subprocess.run(
                    ["say", "-o", str(wav_path), "--data-format=LEI16@22050", text],
                    check=True,
                    capture_output=True,
                )
            elif system == "Windows":
                script = (
                    "Add-Type -AssemblyName System.Speech;"
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                    f"$s.SetOutputToWaveFile('{wav_path}');"
                    f"$s.Speak([Console]::In.ReadToEnd());$s.Dispose()"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    input=text.encode(),
                    check=True,
                    capture_output=True,
                )
            else:
                raise VictorError(f"no system voice on {system}")
            segment = Segment.from_wav(wav_path.read_bytes())
        self.sample_rate = segment.fmt.sample_rate
        yield segment.samples


class NullSynthesizer:
    """Silence, sized to the text. Lets tests exercise timing without audio."""

    name = "null"

    def __init__(self, sample_rate: int = 22_050, chars_per_second: float = 15.0) -> None:
        self.sample_rate = sample_rate
        self.chars_per_second = chars_per_second
        self.spoken: list[str] = []

    def stream(self, text: str) -> Iterator[np.ndarray]:
        self.spoken.append(text)
        seconds = max(0.1, len(text) / self.chars_per_second)
        fmt = AudioFormat(self.sample_rate)
        # Two chunks, so streaming playback is exercised rather than bypassed.
        yield silence(fmt, seconds / 2)
        yield silence(fmt, seconds / 2)


@dataclass(frozen=True, slots=True)
class SpeechStats:
    """What a single spoken reply cost, in time."""

    text: str
    backend: str
    ttfa_ms: float
    """Time to first audio: request to first sample hitting the device."""
    synth_ms: float
    total_ms: float
    audio_seconds: float

    @property
    def realtime_factor(self) -> float:
        """Synthesis time over audio duration. Below 1.0 is faster than real time."""
        return self.synth_ms / 1000 / self.audio_seconds if self.audio_seconds else 0.0


class Player:
    """Writes PCM chunks to the output device as they arrive."""

    def __init__(self, *, device: int | str | None = None, enabled: bool = True) -> None:
        self.device = device
        self.enabled = enabled

    def play(self, chunks: Iterator[np.ndarray], sample_rate: int) -> tuple[float, float, int]:
        """Play a chunk stream.

        Returns ``(ttfa_ms, total_ms, samples)``. With playback disabled the
        stream is still fully consumed, so synthesis timings stay comparable.
        """
        started = time.perf_counter()
        first_audio_at: float | None = None
        total = 0

        if not self.enabled:
            for chunk in chunks:
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                total += len(chunk)
            elapsed = (time.perf_counter() - started) * 1000
            ttfa = ((first_audio_at or time.perf_counter()) - started) * 1000
            return ttfa, elapsed, total

        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise VictorError(f"playback unavailable ({exc}). pip install -e '.[voice]'") from exc

        stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", device=self.device
        )
        with stream:
            for chunk in chunks:
                data = as_int16(chunk)
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                stream.write(data)
                total += len(data)
        elapsed = (time.perf_counter() - started) * 1000
        ttfa = ((first_audio_at or time.perf_counter()) - started) * 1000
        return ttfa, elapsed, total


class _TimedStream:
    """Wraps a chunk generator and accumulates time spent producing chunks.

    Synthesis and playback interleave, so wall-clock time around the whole
    operation cannot be decomposed arithmetically. Timing each ``next()`` call
    measures synthesis exactly, whatever the player is doing between pulls.
    """

    def __init__(self, source: Iterator[np.ndarray]) -> None:
        self._source = source
        self.elapsed_ms = 0.0

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            started = time.perf_counter()
            try:
                chunk = next(self._source)
            except StopIteration:
                self.elapsed_ms += (time.perf_counter() - started) * 1000
                return
            self.elapsed_ms += (time.perf_counter() - started) * 1000
            yield chunk


def speak(synthesizer: Synthesizer, player: Player, text: str) -> SpeechStats:
    """Synthesize and play, measuring synthesis and latency separately."""
    timed = _TimedStream(synthesizer.stream(text))
    ttfa_ms, total_ms, samples = player.play(iter(timed), synthesizer.sample_rate)
    audio_seconds = samples / synthesizer.sample_rate if samples else 0.0
    return SpeechStats(
        text=text,
        backend=synthesizer.name,
        ttfa_ms=ttfa_ms,
        synth_ms=timed.elapsed_ms,
        total_ms=total_ms,
        audio_seconds=audio_seconds,
    )


def build_synthesizer(
    models_dir: Path,
    voice: str = DEFAULT_VOICE,
    *,
    prefer: str = "piper",
    auto_download: bool = True,
) -> Synthesizer:
    """Pick a synthesis backend, preferring the local ONNX one."""
    if prefer == "null":
        return NullSynthesizer()
    if prefer == "system":
        return SystemSynthesizer()

    piper = PiperSynthesizer(models_dir, voice, auto_download=auto_download)
    if piper.installed or auto_download:
        return piper
    if SystemSynthesizer.available():
        return SystemSynthesizer()
    return piper
