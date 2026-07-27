from __future__ import annotations

from pathlib import Path

import pytest

from victor.config import Settings, reset_settings

#: Env vars that would otherwise leak a developer's real keys into tests.
_LEAKY = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "VICTOR_DATA_DIR",
    "VICTOR_STRICT_FREE_TIER",
    "VICTOR_TEXT_MODEL",
    "VICTOR_VISION_MODEL",
    "VICTOR_STT_MODEL",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LEAKY:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with both keys present and state pointed at a tmpdir."""
    return Settings(
        _env_file=None,
        GROQ_API_KEY="test-groq",
        GEMINI_API_KEY="test-gemini",
        VICTOR_DATA_DIR=str(tmp_path / "state"),
    )


class FakeClock:
    """Manually advanced clock, so quota windows can be tested without sleeping."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
