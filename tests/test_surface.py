"""P8: the status strip and the trace-derived benchmark table.

Neither needs a display. The HUD's monitor is a file reader, and the benchmark
table is a fold over recorded events, so the parts with decisions in them are
tested directly and only the tkinter drawing is left unexercised.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from victor.bench import Stage, collect
from victor.ui.hud import Monitor, Snapshot, _today_keys


def write_trace(directory: Path, name: str, events: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return path


def write_quota(path: Path, buckets: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "buckets": buckets}), encoding="utf-8")


def today(zone: str = "UTC") -> str:
    return datetime.now(ZoneInfo(zone)).strftime("%Y-%m-%d")


# --- the status strip ------------------------------------------------------


def test_an_empty_state_directory_reads_as_idle(tmp_path: Path):
    monitor = Monitor(tmp_path / "quota.json", tmp_path / "traces")
    snapshot = monitor.read()
    assert snapshot.state == "idle"
    assert snapshot.spent == 0
    assert snapshot.cost_line == "0 API calls today"


def test_the_strip_names_what_victor_is_doing(tmp_path: Path):
    write_trace(
        tmp_path / "traces",
        "s1",
        [
            {"seq": 1, "kind": "agent.run", "payload": {}},
            {"seq": 2, "kind": "tool.run", "payload": {"tool": "shell"}},
        ],
    )
    snapshot = Monitor(tmp_path / "quota.json", tmp_path / "traces").read()
    assert snapshot.state == "acting"
    assert snapshot.detail == "shell"


def test_the_most_recent_event_wins(tmp_path: Path):
    write_trace(
        tmp_path / "traces",
        "s1",
        [
            {"seq": 1, "kind": "tool.run", "payload": {"tool": "shell"}},
            {"seq": 2, "kind": "agent.answer", "payload": {"answer": "done"}},
        ],
    )
    assert Monitor(tmp_path / "quota.json", tmp_path / "traces").read().state == "idle"


def test_unrecognised_events_do_not_blank_the_strip(tmp_path: Path):
    """A new event kind must not make the HUD forget what it was showing."""
    write_trace(
        tmp_path / "traces",
        "s1",
        [
            {"seq": 1, "kind": "tool.run", "payload": {"tool": "git"}},
            {"seq": 2, "kind": "something.brand.new", "payload": {}},
        ],
    )
    assert Monitor(tmp_path / "quota.json", tmp_path / "traces").read().state == "acting"


def test_a_half_written_trace_line_is_skipped(tmp_path: Path):
    """Traces are appended live, so the last line can be mid-write."""
    directory = tmp_path / "traces"
    directory.mkdir()
    (directory / "s1.jsonl").write_text(
        json.dumps({"seq": 1, "kind": "tool.run", "payload": {"tool": "shell"}})
        + '\n{"seq": 2, "kind": "tool.ru',
        encoding="utf-8",
    )
    assert Monitor(tmp_path / "quota.json", directory).read().state == "acting"


def test_the_newest_session_is_the_one_shown(tmp_path: Path):
    import os
    import time

    directory = tmp_path / "traces"
    old = write_trace(directory, "old", [{"seq": 1, "kind": "tool.run", "payload": {}}])
    time.sleep(0.01)
    new = write_trace(directory, "new", [{"seq": 1, "kind": "memory.recall", "payload": {}}])
    os.utime(old, (time.time() - 100, time.time() - 100))

    assert Monitor(tmp_path / "quota.json", directory).read().state == "remembering"
    assert new.exists()


def test_the_counter_sums_todays_requests(tmp_path: Path):
    quota = tmp_path / "quota.json"
    write_quota(
        quota,
        {
            "groq:openai/gpt-oss-120b": {"day": today(), "requests": 7},
            "gemini:gemini-2.5-flash": {"day": today("America/Los_Angeles"), "requests": 2},
        },
    )
    snapshot = Monitor(quota, tmp_path / "traces").read()
    assert snapshot.spent == 9
    assert "9 API calls today" in snapshot.cost_line


def test_yesterdays_spending_is_not_todays(tmp_path: Path):
    """The ledger keeps a bucket's day, so summing everything shows history."""
    quota = tmp_path / "quota.json"
    write_quota(
        quota,
        {
            "groq:a": {"day": "2020-01-01", "requests": 500},
            "groq:b": {"day": today(), "requests": 3},
        },
    )
    assert Monitor(quota, tmp_path / "traces").read().spent == 3


def test_both_provider_timezones_count_as_today():
    """Groq rolls at UTC midnight and Google at Pacific; for several hours a
    day those disagree, and comparing against one date would read as zero."""
    keys = _today_keys()
    assert today("UTC") in keys
    assert today("America/Los_Angeles") in keys


def test_a_corrupt_quota_file_reads_as_zero_not_a_crash(tmp_path: Path):
    quota = tmp_path / "quota.json"
    quota.write_text("{not json", encoding="utf-8")
    assert Monitor(quota, tmp_path / "traces").read().spent == 0


def test_zero_is_the_headline(tmp_path: Path):
    """The counter staying at zero is the whole claim of the project."""
    assert Snapshot(spent=0).cost_line == "0 API calls today"
    assert "12 API calls" in Snapshot(spent=12).cost_line


# --- the benchmark table ---------------------------------------------------


def test_no_sessions_says_so(tmp_path: Path):
    report = collect(tmp_path / "traces")
    assert report.stages == ()
    assert "no sessions recorded yet" in report.summary()


def test_stages_are_collected_across_sessions(tmp_path: Path):
    directory = tmp_path / "traces"
    for session in ("a", "b"):
        write_trace(
            directory,
            session,
            [
                {"seq": 1, "kind": "tool.run", "duration_ms": 10, "payload": {}},
                {"seq": 2, "kind": "tool.run", "duration_ms": 20, "payload": {}},
            ],
        )
    report = collect(directory)
    assert report.sessions == 2
    stage = next(s for s in report.stages if s.name == "tool call")
    assert stage.count == 4
    assert stage.p50 == 15


def test_events_without_a_duration_are_not_samples(tmp_path: Path):
    write_trace(
        tmp_path / "traces",
        "a",
        [
            {"seq": 1, "kind": "tool.run", "payload": {}},
            {"seq": 2, "kind": "tool.run", "duration_ms": 5, "payload": {}},
        ],
    )
    assert collect(tmp_path / "traces").stages[0].count == 1


def test_unlisted_event_kinds_stay_out_of_the_table():
    """A table of forty rows is not a benchmark, it is a log."""
    from victor.bench import _label_for

    assert _label_for("tool.run") == "tool call"
    assert _label_for("safety.journal") == ""


def test_a_thin_sample_is_labelled_rather_than_interpolated():
    """A p95 over three observations is not a p95."""
    thin = Stage("x", (10.0, 20.0, 90.0))
    assert not thin.trustworthy
    assert thin.p95 == 90.0  # the worst sample, honestly

    thick = Stage("y", tuple(float(i) for i in range(100)))
    assert thick.trustworthy
    assert thick.p95 == pytest.approx(94.0)


def test_the_zero_cost_ratio_comes_from_what_tools_reported(tmp_path: Path):
    write_trace(
        tmp_path / "traces",
        "a",
        [
            {"seq": 1, "kind": "tool.run", "duration_ms": 1, "payload": {"cost": 0}},
            {"seq": 2, "kind": "tool.run", "duration_ms": 1, "payload": {"cost": 0}},
            {"seq": 3, "kind": "tool.run", "duration_ms": 1, "payload": {"cost": 1}},
        ],
    )
    report = collect(tmp_path / "traces")
    assert report.free_tool_calls == 2
    assert report.billed_tool_calls == 1
    assert report.zero_cost_ratio == pytest.approx(2 / 3)
    assert "2/3 tool calls free" in report.summary()


def test_api_calls_are_counted_from_the_events_that_spend_them(tmp_path: Path):
    write_trace(
        tmp_path / "traces",
        "a",
        [
            {"seq": 1, "kind": "llm.complete", "duration_ms": 400, "payload": {}},
            {"seq": 2, "kind": "stt.transcribe", "duration_ms": 300, "payload": {}},
            {"seq": 3, "kind": "vision.locate", "duration_ms": 900, "payload": {}},
            {"seq": 4, "kind": "memory.recall", "duration_ms": 2, "payload": {}},
        ],
    )
    report = collect(tmp_path / "traces")
    assert report.api_calls == 3, "recall is local and must not count"


def test_a_session_with_no_api_calls_reports_zero(tmp_path: Path):
    """The figure the README quotes for a desktop task."""
    write_trace(
        tmp_path / "traces",
        "a",
        [
            {"seq": i, "kind": "tool.run", "duration_ms": 3, "payload": {"cost": 0}}
            for i in range(5)
        ],
    )
    report = collect(tmp_path / "traces")
    assert report.api_calls == 0
    assert report.zero_cost_ratio == 1.0


def test_an_unreadable_trace_does_not_stop_the_table(tmp_path: Path):
    directory = tmp_path / "traces"
    write_trace(directory, "good", [{"seq": 1, "kind": "tool.run", "duration_ms": 4}])
    (directory / "bad.jsonl").write_text("\x00\x01 not text", encoding="utf-8")
    assert collect(directory).stages[0].count >= 1


def test_stages_are_ordered_by_how_much_evidence_they_have(tmp_path: Path):
    write_trace(
        tmp_path / "traces",
        "a",
        [{"seq": 1, "kind": "vision.locate", "duration_ms": 900}]
        + [{"seq": i, "kind": "tool.run", "duration_ms": 4} for i in range(2, 8)],
    )
    names = [s.name for s in collect(tmp_path / "traces").stages]
    assert names[0] == "tool call"


def test_utc_is_always_a_valid_today():
    assert datetime.now(UTC).strftime("%Y-%m-%d") in _today_keys()
