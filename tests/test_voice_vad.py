from __future__ import annotations

import pytest

# Audio is numpy arrays end to end, so these tests need it. Skipping is the
# honest outcome on an install without the voice extra; importing at module
# level would fail collection and take the other 700 tests down with it.
pytest.importorskip("numpy")

import numpy as np

from victor.voice.audio import AudioFormat, silence, tone
from victor.voice.sources import ArraySource
from victor.voice.vad import EndpointConfig, Endpointer, EnergyVad, State

FMT = AudioFormat(16_000)
FRAME_MS = 20


def utterance(lead: float, speech: float, trail: float, amplitude: float = 0.4) -> np.ndarray:
    """Silence, then a tone standing in for speech, then silence."""
    return np.concatenate(
        [silence(FMT, lead), tone(FMT, speech, 220.0, amplitude), silence(FMT, trail)]
    )


def run(audio: np.ndarray, config: EndpointConfig | None = None):
    endpointer = Endpointer(EnergyVad(FMT), config or EndpointConfig(), FMT, FRAME_MS)
    last = 0.0
    for frame, ts in ArraySource(audio, FMT, FRAME_MS).stream():
        last = ts
        result = endpointer.feed(frame, ts)
        if result is not None:
            return result, endpointer
    return endpointer.flush(last), endpointer


# --- the energy backend ---------------------------------------------------


def test_energy_vad_calibrates_on_the_opening_frames() -> None:
    vad = EnergyVad(FMT, calibration_frames=5)
    quiet = silence(FMT, 0.02)

    assert [vad.is_speech(quiet) for _ in range(5)] == [False] * 5
    assert vad.noise_floor_dbfs is not None


def test_energy_vad_separates_tone_from_silence() -> None:
    vad = EnergyVad(FMT, calibration_frames=4)
    quiet = silence(FMT, 0.02)
    loud = tone(FMT, 0.02, 220.0, 0.4)

    for _ in range(6):
        vad.is_speech(quiet)
    assert vad.is_speech(loud)
    assert not vad.is_speech(quiet)


def test_energy_vad_ignores_hum_below_the_absolute_floor() -> None:
    """A very quiet room must not make faint noise look like speech."""
    vad = EnergyVad(FMT, calibration_frames=4, absolute_floor_dbfs=-48.0)
    near_silent = tone(FMT, 0.02, 100.0, 0.00005)  # about -86 dBFS

    for _ in range(10):
        vad.is_speech(near_silent)
    hum = tone(FMT, 0.02, 100.0, 0.0015)  # about -56 dBFS: over the adaptive
    assert not vad.is_speech(hum)  # threshold, under the absolute floor


def test_energy_vad_floor_does_not_drift_up_during_speech() -> None:
    """The bug this guards: a long utterance raising its own threshold."""
    vad = EnergyVad(FMT, calibration_frames=4)
    quiet = silence(FMT, 0.02)
    loud = tone(FMT, 0.02, 220.0, 0.4)

    for _ in range(6):
        vad.is_speech(quiet)
    before = vad.trigger_dbfs
    for _ in range(200):
        vad.is_speech(loud)
    assert vad.trigger_dbfs == pytest.approx(before)


def test_energy_vad_reset_clears_calibration() -> None:
    vad = EnergyVad(FMT)
    for _ in range(20):
        vad.is_speech(silence(FMT, 0.02))
    vad.reset()
    assert vad.noise_floor_dbfs is None


# --- the endpointer -------------------------------------------------------


def test_endpointer_captures_a_full_utterance() -> None:
    endpoint, _ = run(utterance(0.5, 1.5, 1.2))

    assert endpoint is not None
    assert endpoint.reason == "silence"
    # 1.5s of speech, plus 300ms pre-roll and the 200ms retained tail.
    assert endpoint.segment.duration == pytest.approx(2.0, abs=0.25)


def test_endpointer_trims_the_proof_of_silence_tail() -> None:
    """The 700ms pause that ends a turn must not be billed to the STT budget."""
    long_pause = EndpointConfig(end_ms=1_500, keep_tail_ms=100)
    short_pause = EndpointConfig(end_ms=300, keep_tail_ms=100)

    long_endpoint, _ = run(utterance(0.4, 1.0, 2.5), long_pause)
    short_endpoint, _ = run(utterance(0.4, 1.0, 2.5), short_pause)

    assert long_endpoint is not None and short_endpoint is not None
    # A longer end-of-turn pause must not produce a longer upload.
    assert long_endpoint.segment.duration == pytest.approx(
        short_endpoint.segment.duration, abs=0.15
    )


def test_endpointer_keeps_preroll_so_the_onset_survives() -> None:
    """Without pre-roll the first phoneme is lost to the start-trigger delay."""
    config = EndpointConfig(preroll_ms=300, start_ms=100)
    endpoint, _ = run(utterance(0.5, 1.0, 1.0), config)

    assert endpoint is not None
    # Captured audio must exceed the speech itself by roughly the pre-roll.
    assert endpoint.segment.duration > 1.0


def test_endpointer_ignores_a_click() -> None:
    audio = np.concatenate(
        [silence(FMT, 0.4), tone(FMT, 0.02, 900.0, 0.9), silence(FMT, 1.5)]
    )
    endpoint, endpointer = run(audio)

    assert endpoint is None
    assert endpointer.state in (State.WAITING, State.DONE)


def test_endpointer_survives_a_mid_sentence_pause() -> None:
    """A 300 ms breath must not end the turn when end_ms is 700."""
    audio = np.concatenate(
        [
            silence(FMT, 0.4),
            tone(FMT, 0.8, 220.0, 0.4),
            silence(FMT, 0.3),
            tone(FMT, 0.8, 240.0, 0.4),
            silence(FMT, 1.2),
        ]
    )
    endpoint, _ = run(audio, EndpointConfig(end_ms=700))

    assert endpoint is not None
    assert endpoint.segment.duration > 1.6  # both halves, not just the first


def test_endpointer_stops_at_max_duration() -> None:
    config = EndpointConfig(max_utterance_ms=1_000, end_ms=5_000)
    endpoint, _ = run(utterance(0.2, 4.0, 0.2), config)

    assert endpoint is not None
    assert endpoint.truncated
    assert endpoint.reason == "max-duration"
    assert endpoint.segment.duration <= 1.3


def test_endpointer_gives_up_when_nobody_speaks() -> None:
    config = EndpointConfig(max_silence_ms=500)
    endpoint, endpointer = run(silence(FMT, 3.0), config)

    assert endpoint is None
    assert endpointer.state is State.DONE


def test_endpointer_flush_closes_an_open_utterance() -> None:
    """A source that ends mid-speech must still yield what it captured."""
    endpoint, _ = run(utterance(0.4, 2.0, 0.0))

    assert endpoint is not None
    assert endpoint.reason == "source-ended"


def test_endpointer_is_inert_once_done() -> None:
    endpoint, endpointer = run(utterance(0.4, 1.0, 1.2))

    assert endpoint is not None
    assert endpointer.feed(tone(FMT, 0.02, 220.0, 0.5), 99.0) is None


def test_endpoint_records_speech_duration() -> None:
    endpoint, _ = run(utterance(0.4, 1.5, 1.2))

    assert endpoint is not None
    assert endpoint.speech_ms > 500
