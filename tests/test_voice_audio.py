from __future__ import annotations

import pytest

# Audio is numpy arrays end to end, so these tests need it. Skipping is the
# honest outcome on an install without the voice extra; importing at module
# level would fail collection and take the other 700 tests down with it.
pytest.importorskip("numpy")

import numpy as np

from victor.voice.audio import (
    AudioFormat,
    Segment,
    as_int16,
    dbfs,
    frames,
    resample,
    silence,
    to_mono,
    tone,
)

FMT = AudioFormat(16_000)


def test_wav_round_trip_preserves_samples() -> None:
    original = Segment(tone(FMT, 0.5), FMT)
    restored = Segment.from_wav(original.to_wav())

    assert restored.fmt == original.fmt
    np.testing.assert_array_equal(restored.samples, original.samples)


def test_wav_round_trip_downmixes_stereo() -> None:
    import io
    import wave

    buffer = io.BytesIO()
    stereo = np.repeat(tone(FMT, 0.2), 2)  # identical L/R
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(stereo.tobytes())

    segment = Segment.from_wav(buffer.getvalue())
    assert segment.fmt.channels == 1
    assert len(segment.samples) == len(stereo) // 2


def test_resample_changes_length_proportionally() -> None:
    source = tone(AudioFormat(48_000), 1.0)
    out = resample(source, 48_000, 16_000)

    assert out.dtype == np.int16
    assert abs(len(out) - len(source) // 3) <= 1


def test_resample_is_a_noop_at_the_same_rate() -> None:
    source = tone(FMT, 0.1)
    assert resample(source, 16_000, 16_000) is not None
    np.testing.assert_array_equal(resample(source, 16_000, 16_000), source)


def test_segment_resampled_updates_format_and_keeps_duration() -> None:
    segment = Segment(tone(AudioFormat(48_000), 1.0), AudioFormat(48_000))
    out = segment.resampled(16_000)

    assert out.fmt.sample_rate == 16_000
    assert out.duration == pytest.approx(segment.duration, abs=0.01)


def test_dbfs_orders_silence_below_speech() -> None:
    assert dbfs(silence(FMT, 0.1)) == -100.0
    assert dbfs(tone(FMT, 0.1, amplitude=0.01)) < dbfs(tone(FMT, 0.1, amplitude=0.5))
    assert dbfs(tone(FMT, 0.1, amplitude=1.0)) == pytest.approx(-3.0, abs=0.5)


def test_dbfs_of_empty_is_the_floor() -> None:
    assert dbfs(np.zeros(0, dtype=np.int16)) == -100.0


def test_as_int16_scales_floats() -> None:
    out = as_int16(np.array([0.0, 0.5, -1.0], dtype=np.float32))
    assert out.dtype == np.int16
    assert out[0] == 0 and out[1] > 16_000 and out[2] == -32_768


def test_frames_drops_the_short_tail() -> None:
    assert len(frames(np.zeros(1050, dtype=np.int16), 320)) == 3
    assert frames(np.zeros(100, dtype=np.int16), 320) == []


def test_frames_rejects_a_bad_size() -> None:
    with pytest.raises(ValueError):
        frames(np.zeros(10, dtype=np.int16), 0)


def test_to_mono_averages_channels() -> None:
    stereo = np.array([[100, 300], [0, 0]], dtype=np.int16)
    np.testing.assert_array_equal(to_mono(stereo), np.array([200, 0], dtype=np.int16))
