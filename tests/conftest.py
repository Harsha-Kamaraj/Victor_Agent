from __future__ import annotations

import socket
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


#: Loopback is fine - a test may bind its own server. The internet is not.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", ""})


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Refuse real network access, so "the suite is offline" is a property.

    Every network layer is tested through ``httpx.MockTransport``, which made
    offline a *convention* - and a convention holds only as long as the last
    person who remembered it. ``victor bench --voice`` did not: it built a
    synthesizer with ``auto_download=True`` against a fresh tmp models_dir, so
    a unit test fetched a 63 MB voice model from HuggingFace on every run. It
    passed, so nothing said so; the cost was a suite whose runtime depended on
    the network, and one that hung for 53 minutes on a stalled TLS read. The
    fallback around that download catches ``Exception`` - and a stall is not an
    exception, so nothing caught it.

    Mark a test ``@pytest.mark.network`` if it genuinely needs the internet.
    Nothing does today: live checks belong in ``victor selftest --live``, which
    is a command a person runs deliberately, not something a push triggers.
    """
    if request.node.get_closest_marker("network"):
        return

    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def _permitted(sock: socket.socket, address: object) -> bool:
        if sock.family == getattr(socket, "AF_UNIX", object()):
            return True
        host = address[0] if isinstance(address, tuple) else address
        return host in _LOOPBACK

    def _refuse(sock: socket.socket, address: object) -> None:
        host = address[0] if isinstance(address, tuple) else address
        raise OSError(
            f"the test suite is offline and refused a connection to {host!r}. "
            "Route it through httpx.MockTransport, or mark the test "
            "@pytest.mark.network if it truly needs the internet."
        )

    def guarded_connect(self: socket.socket, address: object) -> None:
        if _permitted(self, address):
            return connect(self, address)
        _refuse(self, address)

    def guarded_connect_ex(self: socket.socket, address: object) -> int:
        if _permitted(self, address):
            return connect_ex(self, address)
        _refuse(self, address)
        return 1  # unreachable; keeps the signature honest

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


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
