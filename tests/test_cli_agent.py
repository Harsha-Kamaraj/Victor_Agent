from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from victor.cli import app
from victor.config import Settings
from victor.doctor import Status, run_checks

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
    """Force every httpx.Client built inside the CLI onto the mock."""
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)


def test_do_prints_the_answer(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_transport(monkeypatch, scripted({"content": "You are on branch main."}))

    result = runner.invoke(app, ["do", "what branch am I on"])
    assert result.exit_code == 0, result.output
    assert "You are on branch main." in result.output


def test_do_runs_a_tool_and_shows_it(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (workspace / "marker.txt").write_text("x", encoding="utf-8")
    patch_transport(
        monkeypatch,
        scripted(
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "marker.txt"}),
                        },
                    }
                ]
            },
            {"content": "The file contains x."},
        ),
    )

    result = runner.invoke(app, ["do", "read marker.txt"])
    assert result.exit_code == 0, result.output
    assert "read_file" in result.output
    assert "The file contains x." in result.output


def test_do_exits_nonzero_when_the_provider_fails(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_transport(
        monkeypatch, httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    )

    result = runner.invoke(app, ["do", "anything"])
    assert result.exit_code == 1
    assert "could not reach a model" in result.output


def test_do_announces_dry_run(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_DRY_RUN", "true")
    patch_transport(monkeypatch, scripted({"content": "ok"}))

    result = runner.invoke(app, ["do", "delete everything"])
    assert "dry run" in result.output


def test_tools_command_lists_tools(workspace: Path) -> None:
    result = runner.invoke(app, ["tools"])

    assert result.exit_code == 0
    assert "shell" in result.output
    assert "git" in result.output


def test_doctor_reports_the_agent_as_built(workspace: Path) -> None:
    settings = Settings()
    names = {c.name: c for c in run_checks(settings, network=False)}

    assert names["agent loop"].status is not Status.PENDING
    assert names["agent tools"].status is Status.OK


def test_doctor_no_longer_flags_a_missing_safety_layer(workspace: Path) -> None:
    """P3 shipped: the PENDING placeholder must be gone, not merely quieter."""
    checks = {c.name: c for c in run_checks(Settings(), network=False)}

    assert "safety interceptor" not in checks
    assert checks["safety mode"].status is Status.OK
