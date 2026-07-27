from __future__ import annotations

from pathlib import Path

from victor.quota import UNMETERED, QuotaLedger, QuotaLimits

LIMITS = QuotaLimits(requests_per_minute=3, requests_per_day=5, tokens_per_minute=1_000)
KEY = "groq:test-model"


def make(tmp_path: Path, clock) -> QuotaLedger:
    return QuotaLedger(tmp_path / "quota.json", clock=clock)


def test_unmetered_is_always_allowed(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    for _ in range(1_000):
        ledger.record("piper:local", UNMETERED)
    assert ledger.check("piper:local", UNMETERED).allowed


def test_daily_request_limit(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    for _ in range(5):
        ledger.record(KEY, LIMITS)
        clock.advance(30)  # stay clear of the per-minute cap

    status = ledger.check(KEY, LIMITS)
    assert not status.allowed
    assert "daily request limit" in (status.reason or "")
    assert status.requests_remaining == 0


def test_per_minute_limit_recovers(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    for _ in range(3):
        ledger.record(KEY, LIMITS)

    status = ledger.check(KEY, LIMITS)
    assert not status.allowed
    assert "req/min" in (status.reason or "")
    # Daily budget is untouched - this is a transient block, not an exhausted one.
    assert status.requests_remaining == 2
    assert status.retry_after is not None and 0 < status.retry_after <= 60

    clock.advance(61)
    assert ledger.check(KEY, LIMITS).allowed


def test_token_per_minute_limit_considers_estimate(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    ledger.record(KEY, LIMITS, tokens=900)

    assert not ledger.check(KEY, LIMITS, tokens=200).allowed
    assert ledger.check(KEY, LIMITS, tokens=50).allowed


def test_daily_window_rolls_over(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    for _ in range(5):
        ledger.record(KEY, LIMITS)
        clock.advance(30)
    assert not ledger.check(KEY, LIMITS).allowed

    clock.advance(24 * 3600)
    status = ledger.check(KEY, LIMITS)
    assert status.allowed
    assert status.requests_remaining == 5


def test_audio_seconds_budget(tmp_path: Path, clock) -> None:
    limits = QuotaLimits(audio_seconds_per_day=100.0)
    ledger = make(tmp_path, clock)
    ledger.record("groq:whisper", limits, audio_seconds=95.0)

    assert not ledger.check("groq:whisper", limits, audio_seconds=10.0).allowed
    assert ledger.check("groq:whisper", limits, audio_seconds=4.0).allowed


def test_state_survives_reload(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    ledger.record(KEY, LIMITS, tokens=42)

    reloaded = QuotaLedger(tmp_path / "quota.json", clock=clock)
    assert reloaded.usage(KEY) == (1, 42, 0.0)


def test_corrupt_ledger_does_not_block_startup(tmp_path: Path, clock) -> None:
    path = tmp_path / "quota.json"
    path.write_text("{ this is not json", encoding="utf-8")

    ledger = QuotaLedger(path, clock=clock)
    assert ledger.usage(KEY) == (0, 0, 0.0)
    ledger.record(KEY, LIMITS)  # and it recovers by overwriting
    assert QuotaLedger(path, clock=clock).usage(KEY)[0] == 1


def test_reconciling_tokens_does_not_double_count_requests(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    ledger.record(KEY, LIMITS, tokens=0)
    ledger.record(KEY, LIMITS, requests=0, tokens=120)

    requests, tokens, _ = ledger.usage(KEY)
    assert (requests, tokens) == (1, 120)


def test_reset(tmp_path: Path, clock) -> None:
    ledger = make(tmp_path, clock)
    ledger.record(KEY, LIMITS)
    ledger.reset(KEY)
    assert ledger.usage(KEY) == (0, 0, 0.0)
