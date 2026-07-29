"""The gates that check the gates.

Mostly this file is guarding one property: a self-test that cannot run a gate
must say so, and must never report a pass it did not earn. That is the failure
this command exists to prevent, so it is the failure worth testing for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from victor.cli import app
from victor.config import Settings
from victor.doctor import Status
from victor.selftest import Report, live_cost, selftest

runner = CliRunner()


def test_the_free_run_spends_no_quota(settings: Settings) -> None:
    """The point of the default run: safe to put in a loop, or in CI."""
    report = selftest(settings)

    assert report.cost == 0
    assert all(gate.cost == 0 for gate in report.gates)


def test_every_phase_is_represented(settings: Settings) -> None:
    """A gate quietly dropped is a phase nobody is checking any more."""
    phases = {gate.phase for gate in selftest(settings).gates}
    assert {"P0", "P1", "P3", "P4", "P5", "P6", "P8"} <= phases


def test_the_gates_that_do_not_need_a_machine_pass(settings: Settings) -> None:
    """P0, P3, P6 and P8 need no screen, microphone or key. If any of those
    fails, something is actually broken rather than merely unavailable."""
    portable = {"P0", "P3", "P6", "P8"}
    failed = [
        f"{g.phase}: {g.claim} -> {g.detail}"
        for g in selftest(settings).gates
        if g.phase in portable and g.status is Status.FAIL
    ]
    assert not failed, failed


def test_a_gate_that_raises_becomes_a_failure_not_a_crash(settings: Settings) -> None:
    """One broken gate must not take the rest of the run down with it - a
    self-test that dies halfway has told you less than one that reports."""
    from victor import selftest as module

    @module._gate("PX", "a claim that cannot be checked")
    def exploding(_: Settings):
        raise RuntimeError("the machinery is on fire")

    gate = exploding(settings)
    assert gate.status is Status.FAIL
    assert "the machinery is on fire" in gate.detail
    assert "RuntimeError" in gate.detail


def test_an_unavailable_capability_skips_rather_than_passes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this whole command is written against: reporting success for
    something that was never run."""
    from victor import selftest as module

    monkeypatch.setattr(
        module,
        "_speak",
        lambda *a, **k: None,  # no voice model on disk
    )
    gate = module._p1_tts(settings)

    assert gate.status is Status.SKIP
    # Two ways to be unavailable - no extra installed, no voice downloaded -
    # and which one this machine hits depends on the machine. Both have to name
    # the remedy, which is the part that matters; pinning either message would
    # make this test fail on half the installs it is meant to protect.
    assert "install" in gate.detail


def test_a_missing_key_skips_rather_than_fails(tmp_path: Path) -> None:
    """CI has no keys and a fresh clone has no keys. Both used to report FAIL
    on the routing and voice gates - "not configured here" dressed up as
    "broken", which is the exact confusion this command exists to prevent."""
    bare = Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path))
    failed = [
        f"{g.phase}: {g.claim} -> {g.detail}"
        for g in selftest(bare).gates
        if g.status is Status.FAIL
    ]
    assert not failed, failed


def test_routing_is_checked_without_needing_a_key(tmp_path: Path) -> None:
    """Which model the chain picks is arithmetic, not credentials - so it is
    worth protecting on every push, including the pushes CI runs with no keys
    at all. Standing in a placeholder spends nothing and calls nothing."""
    from victor import selftest as module

    bare = Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path))
    gate = module._p0_routing(bare)

    assert gate.status is Status.OK, gate.detail
    assert "fell through" in gate.detail
    assert gate.cost == 0


def test_live_gates_are_not_run_by_default(settings: Settings) -> None:
    phases = [(g.phase, g.claim) for g in selftest(settings).gates]
    assert not any("Whisper" in claim for _, claim in phases)
    assert not any("the model chooses a tool" in claim for _, claim in phases)


def test_the_live_cost_is_stated_before_it_is_spent() -> None:
    """The number the CLI puts in front of the user, so it cannot drift."""
    assert live_cost() > 0


def test_a_failing_gate_makes_the_report_fail() -> None:
    from victor.selftest import Gate

    passing = Gate("P0", "claim", Status.OK, "fine")
    failing = Gate("P1", "claim", Status.FAIL, "broken")

    assert not Report([passing]).failed
    assert Report([passing, failing]).failed == [failing]
    assert "1 failed" in Report([passing, failing]).summary()


def test_a_skip_does_not_fail_the_command() -> None:
    """A locked screen is not a broken install - the same distinction doctor
    learned the hard way."""
    from victor.selftest import Gate

    report = Report([Gate("P4", "claim", Status.SKIP, "the screen is locked")])
    assert not report.failed
    assert "1 skipped" in report.summary()


def test_cli_selftest_runs_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VICTOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["selftest"])
    assert "passed" in result.output
    # Exit code depends on the machine (a locked screen skips, it does not
    # fail), so the assertion is that it reported rather than crashed.
    assert result.exit_code in (0, 1), result.output


def test_cli_selftest_asks_before_spending_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VICTOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["selftest", "--live"], input="n\n")
    assert "spends up to" in result.output
    assert result.exit_code == 1  # aborted, nothing spent
