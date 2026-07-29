from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from victor.cli import app
from victor.config import Settings
from victor.doctor import Status, run_checks

runner = CliRunner()


def test_blank_env_values_mean_absent(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None, GROQ_API_KEY="   ", VICTOR_DATA_DIR=str(tmp_path)
    )
    assert settings.groq_api_key is None
    assert not settings.has("groq_api_key")


def test_blank_data_dir_falls_back_to_default() -> None:
    settings = Settings(_env_file=None, VICTOR_DATA_DIR="")
    assert settings.data_dir == (Path.home() / ".victor").resolve()


def test_secrets_are_not_printed(settings: Settings) -> None:
    assert "test-groq" not in repr(settings)
    assert settings.secret("groq_api_key") == "test-groq"


def test_paths_are_created_under_data_dir(settings: Settings) -> None:
    paths = settings.paths.ensure()
    assert paths.traces_dir.is_dir()
    assert paths.memory_dir.is_dir()
    assert paths.quota_file.parent == paths.root


def test_doctor_fails_without_the_required_key(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path))
    checks = run_checks(settings, network=False)

    failed = [c for c in checks if c.status is Status.FAIL]
    assert any("GROQ_API_KEY" in c.name for c in failed)


def test_nothing_is_reported_as_unbuilt_now_that_every_phase_shipped(
    settings: Settings,
) -> None:
    """The PENDING convention retires when the last phase lands.

    It existed so a green tick could never stand for a pipeline that did not
    exist. Every phase now has one, so an empty PENDING set is the correct
    state - and a check that still claimed "not implemented" would be the lie
    the convention was guarding against.
    """
    checks = run_checks(settings, network=False)

    assert {c.name for c in checks if c.status is Status.PENDING} == set()
    assert not any("not implemented" in c.detail for c in checks)


def test_a_transient_machine_state_does_not_block_the_exit_code(
    settings: Settings,
) -> None:
    """A locked screen made `victor doctor` exit non-zero. False alarms are how
    a preflight check teaches people to ignore it."""
    checks = run_checks(settings, network=False)
    blocking = [c for c in checks if c.blocking]

    assert not any("locked" in c.detail for c in blocking)


def test_doctor_checks_memory_for_real_now_that_p6_exists(settings: Settings) -> None:
    """P6 shipped, so the store must be opened rather than reported PENDING."""
    names = {c.name: c for c in run_checks(settings, network=False)}

    assert names["memory"].status is not Status.PENDING
    assert names["memory index"].status is not Status.PENDING
    # Which embedder answered is part of the claim, so it has to be stated.
    assert "semantic" in names["memory"].detail or "repeated text" in names["memory"].detail


def test_doctor_checks_voice_for_real_now_that_p1_exists(settings: Settings) -> None:
    """P1 shipped, so mic and TTS must be probed rather than reported PENDING."""
    names = {c.name: c for c in run_checks(settings, network=False)}

    assert "microphone" in names
    assert names["microphone"].status is not Status.PENDING
    assert names["tts (piper)"].status is not Status.PENDING


@pytest.mark.parametrize("command", [["models"], ["--version"]])
def test_cli_commands_exit_cleanly(command: list[str]) -> None:
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output


def test_cli_doctor_exits_nonzero_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VICTOR_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # no .env to pick up

    result = runner.invoke(app, ["doctor", "--no-network"])
    assert result.exit_code == 1
    assert "not ready" in result.output


def test_cli_quota_and_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["quota"]).exit_code == 0
    result = runner.invoke(app, ["route", "text"])
    assert result.exit_code == 0
    assert "gpt-oss-120b" in result.output
