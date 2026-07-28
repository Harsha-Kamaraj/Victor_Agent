"""Voice I/O: microphone in, speech out.

Imports here pull in numpy and (indirectly) PortAudio, so the rest of Victor
must not import this package at module scope. The CLI imports it inside the
commands that need it, which keeps ``victor doctor`` working on an install
without the ``voice`` extra.
"""

from .audio import STT_SAMPLE_RATE, AudioFormat, Segment
from .pipeline import ListenMode, NoSpeechDetected, Turn, VoicePipeline, build_pipeline
from .sources import ArraySource, AudioSource, MicrophoneSource, list_devices
from .stt import Transcriber, Transcript
from .tts import (
    DEFAULT_VOICE,
    NullSynthesizer,
    PiperSynthesizer,
    Player,
    SpeechStats,
    Synthesizer,
    SystemSynthesizer,
    VoiceModelMissing,
    build_synthesizer,
    speak,
)
from .vad import Endpoint, EndpointConfig, Endpointer, EnergyVad, WebRtcVad

__all__ = [
    "ArraySource",
    "AudioFormat",
    "AudioSource",
    "DEFAULT_VOICE",
    "Endpoint",
    "EndpointConfig",
    "Endpointer",
    "EnergyVad",
    "ListenMode",
    "MicrophoneSource",
    "NoSpeechDetected",
    "NullSynthesizer",
    "PiperSynthesizer",
    "Player",
    "STT_SAMPLE_RATE",
    "Segment",
    "SpeechStats",
    "Synthesizer",
    "SystemSynthesizer",
    "Transcriber",
    "Transcript",
    "Turn",
    "VoiceModelMissing",
    "VoicePipeline",
    "WebRtcVad",
    "build_pipeline",
    "build_synthesizer",
    "list_devices",
    "speak",
]
