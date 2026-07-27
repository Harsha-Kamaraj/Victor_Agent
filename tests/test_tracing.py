from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.tracing import Trace, list_traces, read_trace


def test_session_writes_start_and_end(tmp_path: Path) -> None:
    with Trace.open(tmp_path, label="test") as trace:
        trace.event("heard", text="open notepad")
        path = trace.path

    assert path is not None
    events = read_trace(path)
    assert [e["kind"] for e in events] == ["session.start", "heard", "session.end"]
    assert events[-1]["payload"]["status"] == "ok"


def test_span_records_duration_and_extra(tmp_path: Path) -> None:
    with Trace.open(tmp_path) as trace:
        with trace.span("llm.call", model="groq:x") as span:
            span["tokens"] = 128
        path = trace.path

    call = next(e for e in read_trace(path) if e["kind"] == "llm.call")
    assert call["payload"]["model"] == "groq:x"
    assert call["payload"]["tokens"] == 128
    assert call["duration_ms"] >= 0


def test_span_records_failure_and_reraises(tmp_path: Path) -> None:
    trace = Trace.open(tmp_path)
    with pytest.raises(ValueError), trace.span("tool.run", tool="shell"):
        raise ValueError("boom")
    trace.close()

    event = next(e for e in read_trace(trace.path) if e["kind"] == "tool.run")
    assert event["payload"]["status"] == "error"
    assert "boom" in event["payload"]["error"]


def test_session_end_records_the_exception(tmp_path: Path) -> None:
    trace = Trace.open(tmp_path)
    with pytest.raises(RuntimeError), trace:
        raise RuntimeError("kill switch")

    end = read_trace(trace.path)[-1]
    assert end["payload"]["status"] == "error"
    assert "kill switch" in end["payload"]["error"]


def test_truncated_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps({"seq": 1, "ts": "t", "kind": "a"}) + "\n{\"seq\": 2, \"ki",
        encoding="utf-8",
    )
    assert [e["kind"] for e in read_trace(path)] == ["a"]


def test_disabled_trace_writes_nothing(tmp_path: Path) -> None:
    trace = Trace.disabled()
    trace.event("noop")
    trace.close()
    assert list_traces(tmp_path) == []


def test_list_traces_is_newest_first(tmp_path: Path) -> None:
    for name in ("20250101T000000-aaaaaaaa", "20250102T000000-bbbbbbbb"):
        (tmp_path / f"{name}.jsonl").write_text("", encoding="utf-8")
    assert [p.stem[:15] for p in list_traces(tmp_path)] == [
        "20250102T000000",
        "20250101T000000",
    ]
