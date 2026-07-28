from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from victor.cli import app
from victor.config import Settings
from victor.doctor import Status, run_checks
from victor.providers import Workload
from victor.safety import ActionJournal, Risk

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VICTOR_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def scripted(*replies: dict) -> httpx.MockTransport:
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        body = queue.pop(0) if queue else {"content": "done"}
        message = {"role": "assistant", "content": body.get("content")}
        if "tool_calls" in body:
            message["tool_calls"] = body["tool_calls"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            },
        )

    return httpx.MockTransport(handler)


def patch_transport(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)


def shell_call(command: str, call_id: str = "c1") -> dict:
    return {
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": command})},
            }
        ]
    }


# --- victor check ---------------------------------------------------------


def test_check_reports_safe(workspace: Path) -> None:
    result = runner.invoke(app, ["check", "ls -la"])
    assert result.exit_code == 0
    assert "safe" in result.output


def test_check_reports_confirm_and_undo_status(workspace: Path) -> None:
    result = runner.invoke(app, ["check", "rm notes.txt"])

    assert "confirm" in result.output
    assert "cannot be undone" in result.output


def test_check_reports_denial(workspace: Path) -> None:
    result = runner.invoke(app, ["check", "rm -rf /"])
    assert "deny" in result.output


def test_check_shows_available_undo(workspace: Path) -> None:
    result = runner.invoke(app, ["check", "mkdir build"])
    assert "undo available" in result.output


# --- the gate, through the CLI --------------------------------------------


def test_destructive_action_is_refused_without_confirmation(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No terminal to ask on means the answer is no."""
    (workspace / "target.txt").write_text("data", encoding="utf-8")
    patch_transport(
        monkeypatch,
        scripted(shell_call("rm target.txt"), {"content": "I could not remove it."}),
    )

    result = runner.invoke(app, ["do", "delete target.txt"])

    assert result.exit_code == 0
    assert (workspace / "target.txt").exists()  # the file survived


def test_yes_flag_approves_and_the_action_runs(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "target.txt").write_text("data", encoding="utf-8")
    patch_transport(
        monkeypatch, scripted(shell_call("rm target.txt"), {"content": "Removed it."})
    )

    result = runner.invoke(app, ["do", "delete target.txt", "--yes"])

    assert result.exit_code == 0
    assert not (workspace / "target.txt").exists()
    assert "every action is pre-approved" in result.output


def test_dry_run_previews_without_executing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "target.txt").write_text("data", encoding="utf-8")
    patch_transport(
        monkeypatch, scripted(shell_call("rm target.txt"), {"content": "It would be removed."})
    )

    result = runner.invoke(app, ["do", "delete target.txt", "--dry-run"])

    assert "dry run" in result.output
    assert (workspace / "target.txt").exists()


def test_catastrophic_command_is_refused_even_with_yes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--yes approves confirmations; it does not unlock the denylist."""
    patch_transport(
        monkeypatch, scripted(shell_call("rm -rf /"), {"content": "I will not do that."})
    )

    result = runner.invoke(app, ["do", "wipe the disk", "--yes"])
    assert result.exit_code == 0
    assert "I will not do that." in result.output


def test_safe_commands_run_without_a_prompt(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "marker.txt").write_text("x", encoding="utf-8")
    patch_transport(
        monkeypatch, scripted(shell_call("ls"), {"content": "There is one file."})
    )

    result = runner.invoke(app, ["do", "what is here"])
    assert result.exit_code == 0
    assert "marker.txt" in result.output or "There is one file." in result.output


# --- journal --------------------------------------------------------------


def test_executed_actions_are_journalled(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_transport(
        monkeypatch, scripted(shell_call("mkdir made"), {"content": "Created it."})
    )
    runner.invoke(app, ["do", "make a directory", "--yes"])

    journal = ActionJournal(Settings().paths.journal_file)
    entries = [e for e in journal if e.tool == "shell"]
    assert entries
    assert any("mkdir made" in str(e.arguments) for e in entries)


def test_reads_are_not_journalled(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The journal is for things with effects, not for every `ls`."""
    patch_transport(monkeypatch, scripted(shell_call("ls"), {"content": "done"}))
    runner.invoke(app, ["do", "list files", "--yes"])

    assert list(ActionJournal(Settings().paths.journal_file)) == []


def test_journal_list_shows_entries(workspace: Path) -> None:
    settings = Settings()
    journal = ActionJournal(settings.paths.ensure().journal_file)
    journal.record("shell", {"command": "rm x"}, risk=Risk.CONFIRM, decision="allow")
    journal.record("shell", {"command": "mkdir y"}, risk=Risk.CONFIRM, decision="allow")

    result = runner.invoke(app, ["journal", "list"])
    assert result.exit_code == 0
    assert "rm x" in result.output
    assert "mkdir y" in result.output


def test_journal_list_distinguishes_reversible_from_not(workspace: Path) -> None:
    settings = Settings()
    journal = ActionJournal(settings.paths.ensure().journal_file)
    journal.record("shell", {"command": "rm x"}, risk=Risk.CONFIRM, decision="allow")

    result = runner.invoke(app, ["journal", "list"])
    # The table wraps, so compare on collapsed whitespace.
    assert "cannot be restored" in " ".join(result.output.split())


def test_journal_list_is_empty_before_anything_happens(workspace: Path) -> None:
    result = runner.invoke(app, ["journal", "list"])
    assert "no actions recorded" in result.output


# --- undo -----------------------------------------------------------------


def test_undo_reverses_a_mkdir(workspace: Path) -> None:
    settings = Settings()
    journal = ActionJournal(settings.paths.ensure().journal_file)
    (workspace / "made").mkdir()
    journal.record("shell", {"command": "mkdir made"}, risk=Risk.CONFIRM, decision="allow")

    result = runner.invoke(app, ["journal", "undo", "last"])
    assert result.exit_code == 0, result.output
    assert not (workspace / "made").exists()


def test_undo_refuses_an_irreversible_action(workspace: Path) -> None:
    settings = Settings()
    journal = ActionJournal(settings.paths.ensure().journal_file)
    entry = journal.record(
        "shell", {"command": "rm gone.txt"}, risk=Risk.CONFIRM, decision="allow"
    )

    result = runner.invoke(app, ["journal", "undo", entry.id])
    assert result.exit_code == 1
    assert "cannot be restored" in result.output


def test_undo_last_with_nothing_reversible_explains_why(workspace: Path) -> None:
    settings = Settings()
    ActionJournal(settings.paths.ensure().journal_file).record(
        "shell", {"command": "rm x"}, risk=Risk.CONFIRM, decision="allow"
    )

    result = runner.invoke(app, ["journal", "undo", "last"])
    assert result.exit_code == 1
    assert "no inverse" in result.output or "nothing recent" in result.output


def test_undo_reverses_a_git_add(workspace: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=workspace, check=True)
    (workspace / "a.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=workspace, check=True)

    settings = Settings()
    ActionJournal(settings.paths.ensure().journal_file).record(
        "git", {"subcommand": "add", "args": ["a.txt"]}, risk=Risk.CONFIRM, decision="allow"
    )

    result = runner.invoke(app, ["journal", "undo", "last"])
    assert result.exit_code == 0, result.output

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert "a.txt" not in staged.stdout


# --- doctor ---------------------------------------------------------------


def test_doctor_reports_the_safety_layer_as_built(workspace: Path) -> None:
    names = {c.name: c for c in run_checks(Settings(), network=False)}

    assert "safety interceptor" not in names
    assert names["safety mode"].status is Status.OK
    assert names["action journal"].status is Status.OK


def test_doctor_warns_when_confirmation_is_disabled(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VICTOR_CONFIRM_DESTRUCTIVE", "false")
    names = {c.name: c for c in run_checks(Settings(), network=False)}

    assert names["safety mode"].status is Status.WARN
    assert "without asking" in names["safety mode"].detail


def test_tools_no_longer_claims_the_gate_is_missing(workspace: Path) -> None:
    result = runner.invoke(app, ["tools"])

    assert result.exit_code == 0
    assert "not built yet" not in result.output
    assert "classified before they run" in result.output


# --- the CLI surface the plan names ---------------------------------------


def test_undo_last_n_reverses_several(workspace: Path) -> None:
    settings = Settings()
    journal = ActionJournal(settings.paths.ensure().journal_file)
    for name in ("one", "two"):
        (workspace / name).mkdir()
        journal.record(
            "shell", {"command": f"mkdir {name}"}, risk=Risk.CONFIRM, decision="allow"
        )

    result = runner.invoke(app, ["undo", "--last", "2"])
    assert result.exit_code == 0, result.output
    assert not (workspace / "one").exists()
    assert not (workspace / "two").exists()


def test_undo_explains_when_nothing_is_reversible(workspace: Path) -> None:
    result = runner.invoke(app, ["undo"])
    assert result.exit_code == 1
    assert "nothing recent" in result.output


def test_sessions_and_replay_are_aliases(workspace: Path) -> None:
    from victor.tracing import Trace

    settings = Settings()
    with Trace.open(settings.paths.ensure().traces_dir, label="t") as trace:
        trace.event("hello", text="world")

    listed = runner.invoke(app, ["sessions"])
    assert listed.exit_code == 0
    assert "ok" in listed.output

    shown = runner.invoke(app, ["replay", "last"])
    assert shown.exit_code == 0
    assert "hello" in shown.output


def test_install_shim_writes_a_launcher(workspace: Path) -> None:
    target = workspace / "bin"
    result = runner.invoke(app, ["install-shim", "--dir", str(target)])

    assert result.exit_code == 0, result.output
    shim = next(target.iterdir())
    assert "-m victor" in shim.read_text()


def test_run_with_text_executes_one_task(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_transport(monkeypatch, scripted({"content": "On branch main."}))

    result = runner.invoke(app, ["run", "--text", "what branch am I on"])
    assert result.exit_code == 0, result.output
    assert "On branch main." in result.output


def test_bench_takes_a_voice_flag(workspace: Path) -> None:
    result = runner.invoke(app, ["bench", "--voice", "--runs", "1"])
    assert result.exit_code == 0, result.output
    assert "vad endpointing" in result.output


def test_limits_are_config_not_constants(workspace: Path) -> None:
    """The plan: every free-tier limit is config, never hardcoded."""
    import json

    from victor.providers import Router
    from victor.providers.registry import GEMINI_25_FLASH
    from victor.quota import QuotaLedger

    settings = Settings()
    settings.paths.ensure().limits_file.write_text(
        json.dumps({GEMINI_25_FLASH.key: {"requests_per_day": 9999}}), encoding="utf-8"
    )

    router = Router(settings, QuotaLedger(settings.paths.quota_file))
    spec = router.chain(Workload.VISION)[0]
    assert spec.limits.requests_per_day == 9999


def test_a_broken_limits_file_does_not_stop_startup(workspace: Path) -> None:
    from victor.providers import Router
    from victor.providers.registry import GEMINI_25_FLASH
    from victor.quota import QuotaLedger

    settings = Settings()
    settings.paths.ensure().limits_file.write_text("{ not json", encoding="utf-8")

    router = Router(settings, QuotaLedger(settings.paths.quota_file))
    spec = router.chain(Workload.VISION)[0]
    assert spec.limits.requests_per_day == GEMINI_25_FLASH.limits.requests_per_day
