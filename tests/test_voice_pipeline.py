from __future__ import annotations

import httpx
import numpy as np
import pytest

from victor.config import Settings
from victor.providers import Router
from victor.quota import QuotaLedger
from victor.tracing import Trace, read_trace
from victor.voice.audio import AudioFormat, silence, tone
from victor.voice.pipeline import ListenMode, NoSpeechDetected, VoicePipeline
from victor.voice.sources import ArraySource
from victor.voice.stt import Transcriber
from victor.voice.tts import NullSynthesizer, Player
from victor.voice.vad import EndpointConfig

FMT = AudioFormat(16_000)


def spoken_audio(lead: float = 0.4, speech: float = 1.2, trail: float = 1.2) -> np.ndarray:
    return np.concatenate(
        [silence(FMT, lead), tone(FMT, speech, 220.0, 0.4), silence(FMT, trail)]
    )


def build(
    settings: Settings,
    audio: np.ndarray,
    *,
    reply: str = "open notepad",
    trace: Trace | None = None,
) -> VoicePipeline:
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    trace = trace or Trace.disabled()
    router = Router(settings, ledger, on_select=trace.selection)
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"text": reply}))
    )
    return VoicePipeline(
        settings,
        router,
        transcriber=Transcriber(settings, router, client=client, trace=trace),
        synthesizer=NullSynthesizer(),
        player=Player(enabled=False),
        source_factory=lambda: ArraySource(audio, FMT, 20),
        fmt=FMT,
        trace=trace,
    )


def test_listen_returns_a_transcript(settings: Settings) -> None:
    pipeline = build(settings, spoken_audio())
    turn = pipeline.listen(ListenMode.VAD)

    assert turn.text == "open notepad"
    assert turn.endpoint.reason == "silence"
    assert turn.transcript.audio_seconds > 0


def test_fixed_mode_records_the_requested_length(settings: Settings) -> None:
    pipeline = build(settings, tone(FMT, 5.0, 220.0, 0.4))
    endpoint = pipeline.capture(ListenMode.FIXED, seconds=2.0)

    assert endpoint.segment.duration == pytest.approx(2.0, abs=0.05)
    assert endpoint.reason == "fixed"


def test_silence_raises_rather_than_transcribing_nothing(settings: Settings) -> None:
    pipeline = build(settings, silence(FMT, 3.0))
    with pytest.raises(NoSpeechDetected):
        pipeline.listen(ListenMode.VAD)


def test_speak_reports_stats(settings: Settings) -> None:
    pipeline = build(settings, spoken_audio())
    stats = pipeline.speak("acknowledged")

    assert stats.backend == "null"
    assert stats.audio_seconds > 0


def test_level_callback_sees_every_frame(settings: Settings) -> None:
    pipeline = build(settings, spoken_audio())
    levels: list[float] = []
    pipeline.capture(ListenMode.VAD, on_level=lambda db, speaking: levels.append(db))

    assert len(levels) > 10
    assert max(levels) > min(levels)  # the tone is louder than the lead-in silence


def test_a_full_turn_is_traced_end_to_end(settings: Settings) -> None:
    with Trace.open(settings.paths.ensure().traces_dir, label="test") as trace:
        pipeline = build(settings, spoken_audio(), trace=trace)
        pipeline.listen(ListenMode.VAD)
        pipeline.speak("done")
        path = trace.path

    kinds = [e["kind"] for e in read_trace(path)]
    assert "voice.capture" in kinds
    assert "router.select" in kinds
    assert "stt.transcribe" in kinds
    assert "voice.heard" in kinds
    assert "voice.speak" in kinds


def test_endpoint_config_is_honoured(settings: Settings) -> None:
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    router = Router(settings, ledger)
    pipeline = VoicePipeline(
        settings,
        router,
        transcriber=Transcriber(
            settings,
            router,
            client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"text": "x"}))
            ),
        ),
        synthesizer=NullSynthesizer(),
        player=Player(enabled=False),
        source_factory=lambda: ArraySource(tone(FMT, 6.0, 220.0, 0.4), FMT, 20),
        fmt=FMT,
        endpoint_config=EndpointConfig(max_utterance_ms=1_000, end_ms=5_000),
    )
    endpoint = pipeline.capture(ListenMode.VAD)

    assert endpoint.truncated
    assert endpoint.segment.duration <= 1.3
