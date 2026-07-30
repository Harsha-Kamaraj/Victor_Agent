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


def test_the_agent_closes_the_memory_it_opened(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Agent.close()` shut the model client and forgot the SQLite connection.

    Asserting that close is *called*, not that cleanup succeeded: on POSIX an
    open file unlinks happily, so a test of the outcome passes here whatever the
    code does. Windows is where a held handle stops the file being removed, and
    this test has to be able to fail on the machine where that cannot happen.
    """
    from victor.rag.store import VectorStore

    closed: list[str] = []
    original = VectorStore.close

    def spy(self, *args, **kwargs):
        closed.append(str(self.db_path))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VectorStore, "close", spy)
    patch_transport(monkeypatch, scripted({"content": "You are on branch main."}))

    result = runner.invoke(app, ["do", "what branch am I on"])

    assert result.exit_code == 0, result.output
    assert closed, "victor do left the memory's SQLite connection open"


def test_an_injected_memory_is_left_open_for_its_owner(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing what you were handed is as wrong as leaking what you opened - the
    caller may still be using it, and several tests are."""
    from victor.agent import build_agent
    from victor.rag import Memory, VectorStore
    from victor.rag.embed import HashEmbedder

    embedder = HashEmbedder()
    store = VectorStore(
        workspace / "borrowed", embedder_name=embedder.name, dimensions=embedder.dimensions
    )
    memory = Memory(store, embedder)
    try:
        agent = build_agent(Settings(_env_file=None, GROQ_API_KEY="k"), memory=memory)
        agent.close()

        # Still usable: the owner has not been shut down under them.
        assert memory.recall_for_error("anything") is not None
    finally:
        store.close()
