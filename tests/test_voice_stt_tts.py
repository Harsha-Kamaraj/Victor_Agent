from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from victor.config import Settings
from victor.errors import ProviderError, QuotaExhausted
from victor.providers import Router
from victor.providers.registry import WHISPER_TURBO
from victor.quota import QuotaLedger
from victor.voice.audio import AudioFormat, Segment, tone
from victor.voice.stt import Transcriber
from victor.voice.tts import NullSynthesizer, Player, SpeechStats, speak

FMT = AudioFormat(16_000)


def segment(seconds: float = 1.0, rate: int = 16_000) -> Segment:
    fmt = AudioFormat(rate)
    return Segment(tone(fmt, seconds), fmt)


def transcriber_with(
    handler, settings: Settings, ledger: QuotaLedger | None = None
) -> tuple[Transcriber, QuotaLedger]:
    led = ledger or QuotaLedger(settings.paths.ensure().quota_file)
    router = Router(settings, led)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Transcriber(settings, router, client=client), led


# --- STT ------------------------------------------------------------------


def test_transcribe_returns_text(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-groq"
        assert b"whisper-large-v3-turbo" in request.content
        return httpx.Response(200, json={"text": "  open notepad  "})

    stt, _ = transcriber_with(handler, settings)
    result = stt.transcribe(segment(1.0))

    assert result.text == "open notepad"
    assert result.model == WHISPER_TURBO.key
    assert result.audio_seconds == pytest.approx(1.0, abs=0.01)


def test_transcribe_charges_audio_seconds_to_the_ledger(settings: Settings) -> None:
    stt, ledger = transcriber_with(
        lambda r: httpx.Response(200, json={"text": "hi"}), settings
    )
    stt.transcribe(segment(2.5))

    requests, _, audio = ledger.usage(WHISPER_TURBO.key)
    assert requests == 1
    assert audio == pytest.approx(2.5, abs=0.01)


def test_audio_is_charged_even_when_the_call_fails(settings: Settings) -> None:
    """Whisper bills by duration, so a 500 still costs what it consumed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")

    stt, ledger = transcriber_with(handler, settings)
    with pytest.raises(ProviderError):
        stt.transcribe(segment(3.0))

    assert ledger.usage(WHISPER_TURBO.key)[2] == pytest.approx(3.0, abs=0.01)


def test_audio_is_resampled_to_16k_before_upload(settings: Settings) -> None:
    seen: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # WAV layout from the "WAVE" tag: +4 "fmt ", +8 chunk size, +12 format,
        # +14 channels, +16 sample rate.
        body = request.content
        offset = body.find(b"WAVE")
        seen["rate"] = int.from_bytes(body[offset + 16 : offset + 20], "little")
        return httpx.Response(200, json={"text": "ok"})

    stt, _ = transcriber_with(handler, settings)
    stt.transcribe(segment(0.5, rate=48_000))

    assert seen["rate"] == 16_000


def test_rate_limit_raises_quota_exhausted(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "30"}, json={})

    stt, _ = transcriber_with(handler, settings)
    with pytest.raises(QuotaExhausted) as excinfo:
        stt.transcribe(segment(1.0))
    assert excinfo.value.retry_after == 30


def test_rejected_key_is_reported_clearly(settings: Settings) -> None:
    stt, _ = transcriber_with(lambda r: httpx.Response(401, json={}), settings)
    with pytest.raises(ProviderError, match="GROQ_API_KEY rejected"):
        stt.transcribe(segment(1.0))


def test_network_failure_is_wrapped(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    stt, _ = transcriber_with(handler, settings)
    with pytest.raises(ProviderError, match="ConnectError"):
        stt.transcribe(segment(1.0))


def test_prompt_and_language_are_forwarded(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b"git status" in request.content
        assert b"name=\"language\"" in request.content
        return httpx.Response(200, json={"text": "ok"})

    stt, _ = transcriber_with(handler, settings)
    stt.transcribe(segment(0.5), language="en", prompt="git status")


def test_no_stt_credential_raises_before_any_request(tmp_path: Path) -> None:
    from victor.errors import NoProviderAvailable

    bare = Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path))
    stt, _ = transcriber_with(lambda r: httpx.Response(200, json={}), bare)
    with pytest.raises(NoProviderAvailable):
        stt.transcribe(segment(1.0))


# --- TTS ------------------------------------------------------------------


def test_speak_measures_streaming_latency() -> None:
    synth = NullSynthesizer()
    stats = speak(synth, Player(enabled=False), "hello there")

    assert isinstance(stats, SpeechStats)
    assert stats.backend == "null"
    assert stats.audio_seconds > 0
    assert stats.ttfa_ms >= 0
    assert synth.spoken == ["hello there"]


def test_realtime_factor_is_derived_from_audio_length() -> None:
    stats = speak(NullSynthesizer(), Player(enabled=False), "a" * 150)

    assert stats.audio_seconds == pytest.approx(10.0, abs=0.5)
    assert stats.realtime_factor < 1.0  # silence generation is far faster than real time


def test_disabled_player_still_consumes_every_chunk() -> None:
    """Benchmarks with playback off must measure the same synthesis work."""
    synth = NullSynthesizer()
    ttfa, total, samples = Player(enabled=False).play(synth.stream("hello"), synth.sample_rate)

    assert samples > 0
    assert total >= ttfa


def test_piper_reports_missing_voice_instead_of_silently_downloading(
    tmp_path: Path,
) -> None:
    from victor.voice.tts import PiperSynthesizer, VoiceModelMissing

    synth = PiperSynthesizer(tmp_path / "models", auto_download=False)
    assert not synth.installed
    with pytest.raises(VoiceModelMissing, match="victor voice install"):
        list(synth.stream("hello"))
