from __future__ import annotations

from pathlib import Path

import pytest

from victor.config import Settings
from victor.errors import NoProviderAvailable
from victor.providers import Router, Workload
from victor.providers.registry import GEMINI_25_FLASH, GPT_OSS_120B, LLAMA_4_SCOUT
from victor.quota import QuotaLedger


def ledger_for(tmp_path: Path, clock) -> QuotaLedger:
    return QuotaLedger(tmp_path / "quota.json", clock=clock)


def test_prefers_the_head_of_the_chain(settings: Settings, tmp_path: Path, clock) -> None:
    router = Router(settings, ledger_for(tmp_path, clock))
    assert router.select(Workload.TEXT).key == GPT_OSS_120B.key
    assert router.select(Workload.VISION).key == GEMINI_25_FLASH.key


def test_falls_through_when_daily_allowance_is_spent(
    settings: Settings, tmp_path: Path, clock
) -> None:
    ledger = ledger_for(tmp_path, clock)
    router = Router(settings, ledger)

    # Burn Gemini's whole day, spacing calls out past its per-minute window.
    for _ in range(GEMINI_25_FLASH.limits.requests_per_day or 0):
        ledger.record(GEMINI_25_FLASH.key, GEMINI_25_FLASH.limits)
        clock.advance(10)

    selection = router.select(Workload.VISION)
    assert selection.key == LLAMA_4_SCOUT.key
    assert selection.rejected
    assert "daily request limit" in selection.rejected[0][1]


def test_missing_credential_skips_a_provider(tmp_path: Path, clock) -> None:
    settings = Settings(
        _env_file=None, GROQ_API_KEY="test-groq", VICTOR_DATA_DIR=str(tmp_path)
    )
    router = Router(settings, ledger_for(tmp_path, clock))

    selection = router.select(Workload.VISION)
    assert selection.key == LLAMA_4_SCOUT.key
    assert selection.rejected[0][1] == "GEMINI_API_KEY not set"


def test_raises_when_nothing_can_serve(tmp_path: Path, clock) -> None:
    settings = Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path))
    router = Router(settings, ledger_for(tmp_path, clock))

    with pytest.raises(NoProviderAvailable) as excinfo:
        router.select(Workload.TEXT)
    assert "GROQ_API_KEY not set" in str(excinfo.value)


def test_local_models_need_no_credential(tmp_path: Path, clock) -> None:
    settings = Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path))
    router = Router(settings, ledger_for(tmp_path, clock))

    assert router.select(Workload.TTS).spec.local
    assert router.select(Workload.EMBEDDING).spec.local


def test_override_promotes_without_removing_the_chain(
    tmp_path: Path, clock
) -> None:
    settings = Settings(
        _env_file=None,
        GROQ_API_KEY="test-groq",
        VICTOR_DATA_DIR=str(tmp_path),
        VICTOR_TEXT_MODEL="llama-3.1-8b-instant",
    )
    router = Router(settings, ledger_for(tmp_path, clock))

    chain = router.chain(Workload.TEXT)
    assert chain[0].model == "llama-3.1-8b-instant"
    assert GPT_OSS_120B in chain  # still there as a fallback


def test_unknown_override_is_ignored(tmp_path: Path, clock) -> None:
    settings = Settings(
        _env_file=None,
        GROQ_API_KEY="test-groq",
        VICTOR_DATA_DIR=str(tmp_path),
        VICTOR_TEXT_MODEL="does-not-exist",
    )
    router = Router(settings, ledger_for(tmp_path, clock))
    assert router.select(Workload.TEXT).key == GPT_OSS_120B.key


def test_strict_free_tier_off_allows_overspend(tmp_path: Path, clock) -> None:
    settings = Settings(
        _env_file=None,
        GEMINI_API_KEY="test-gemini",
        VICTOR_DATA_DIR=str(tmp_path),
        VICTOR_STRICT_FREE_TIER=False,
    )
    ledger = ledger_for(tmp_path, clock)
    router = Router(settings, ledger)

    for _ in range(GEMINI_25_FLASH.limits.requests_per_day or 0):
        ledger.record(GEMINI_25_FLASH.key, GEMINI_25_FLASH.limits)
        clock.advance(10)

    selection = router.select(Workload.VISION)
    assert selection.key == GEMINI_25_FLASH.key
    assert "billing may apply" in (selection.status.reason or "")


def test_record_skips_local_models(settings: Settings, tmp_path: Path, clock) -> None:
    ledger = ledger_for(tmp_path, clock)
    router = Router(settings, ledger)

    selection = router.select(Workload.TTS)
    router.record(selection)
    assert ledger.usage(selection.key) == (0, 0, 0.0)


def test_on_select_hook_fires(settings: Settings, tmp_path: Path, clock) -> None:
    seen = []
    router = Router(settings, ledger_for(tmp_path, clock), on_select=seen.append)
    router.select(Workload.TEXT)
    assert len(seen) == 1 and seen[0].workload is Workload.TEXT
