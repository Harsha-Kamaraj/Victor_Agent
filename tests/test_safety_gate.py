from __future__ import annotations

import io
import platform
import shlex
import sys
import threading
import time
from pathlib import Path

import pytest

from victor.config import Settings
from victor.safety import (
    ActionJournal,
    AutoConfirmer,
    DenyingConfirmer,
    KillSwitch,
    Risk,
    SafetyInterceptor,
    TypedConfirmer,
    interpret,
    plan_undo,
    undo_entry,
    undo_last,
)
from victor.safety.classify import Classification
from victor.safety.confirm import ConfirmRequest, SpokenConfirmer, build_confirmer
from victor.tools import ShellTool, build_registry
from victor.tools.base import Decision, ToolSpec


def _shell_invocation(*argv: str) -> str:
    """Quote a program and its arguments for whichever shell the tool uses.

    PowerShell needs the call operator in front of a quoted program path, or it
    treats the quoted string as a literal to print rather than a command to run.
    """
    if platform.system() == "Windows":
        return "& " + " ".join(f'"{part}"' for part in argv)
    return " ".join(shlex.quote(part) for part in argv)


SHELL = ToolSpec("shell", "run", {"type": "object"}, mutating=True)
READER = ToolSpec("read_file", "read", {"type": "object"}, mutating=False)


def interceptor(**kwargs) -> SafetyInterceptor:
    kwargs.setdefault("confirmer", AutoConfirmer(True))
    return SafetyInterceptor(**kwargs)


# --- the gate -------------------------------------------------------------


def test_safe_calls_are_not_confirmed() -> None:
    """Prompting for `ls` is how a safety layer becomes noise."""
    confirmer = AutoConfirmer(True)
    gate = interceptor(confirmer=confirmer)

    review = gate.review(SHELL, {"command": "ls -la"})
    assert review.decision is Decision.ALLOW
    assert confirmer.requests == []


def test_destructive_calls_are_confirmed() -> None:
    confirmer = AutoConfirmer(True)
    gate = interceptor(confirmer=confirmer)

    review = gate.review(SHELL, {"command": "rm build.log"})
    assert review.decision is Decision.ALLOW
    assert len(confirmer.requests) == 1
    assert "rm build.log" in confirmer.requests[0].summary


def test_refusing_blocks_the_call() -> None:
    gate = interceptor(confirmer=AutoConfirmer(False))

    review = gate.review(SHELL, {"command": "rm -r build"})
    assert review.decision is Decision.DENY
    assert "did not approve" in review.reason


def test_catastrophic_calls_are_denied_without_asking() -> None:
    """`rm -rf /` must not even offer a yes button."""
    confirmer = AutoConfirmer(True)
    gate = interceptor(confirmer=confirmer)

    review = gate.review(SHELL, {"command": "rm -rf /"})
    assert review.decision is Decision.DENY
    assert "irreversible-damage list" in review.reason
    assert confirmer.requests == []


def test_dry_run_previews_what_a_glob_matched(tmp_path: Path) -> None:
    """Echoing the command back is not a preview - the user typed it."""
    for name in ("a.log", "b.log", "c.log"):
        (tmp_path / name).write_text("x" * 500, encoding="utf-8")

    gate = interceptor(dry_run=True, cwd=tmp_path)
    review = gate.review(SHELL, {"command": "rm *.log"})

    assert review.decision is Decision.DENY
    assert "would delete 3 items" in review.reason
    assert "a.log" in review.reason


# --- the preview has to be true, not merely present ------------------------
#
# A preview is what the user approves, so a wrong one is worse than none: it
# buys consent for something that will not happen. These three were all found by
# running the branches nobody had run - the redirect and move previews had no
# test coverage at all - and all three erred in the same direction, understating
# or misnaming the change.


def test_a_quoted_redirect_does_not_hide_the_real_target(tmp_path: Path) -> None:
    """`echo "a > b" > out.txt` previewed as "would create b". The user would be
    approving a file the command never touches, while out.txt is the one it
    actually writes."""
    from victor.safety.preview import preview

    assert preview('echo "a > b" > out.txt', tmp_path) == "would create out.txt"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("echo hi >out.txt", "would create out.txt"),  # no space
        ("echo boom >&2", "would run: echo boom >&2"),  # not a file
        ("ls 2>err.txt", "would run: ls 2>err.txt"),  # stderr, not a truncation
    ],
)
def test_redirect_forms_that_are_not_a_file_write(
    tmp_path: Path, command: str, expected: str
) -> None:
    from victor.safety.preview import preview

    assert preview(command, tmp_path) == expected


def test_every_source_of_a_move_is_named(tmp_path: Path) -> None:
    """It reported the first operand only, so `mv a b c dir/` was previewed as
    moving one file. Approving a third of an action is not approving it."""
    from victor.safety.preview import preview

    (tmp_path / "dir").mkdir()
    rendered = preview("mv a.txt b.txt c.txt dir/", tmp_path)

    assert "3 files" in rendered
    for name in ("a.txt", "b.txt", "c.txt"):
        assert name in rendered


def test_moving_into_a_directory_says_what_it_replaces(tmp_path: Path) -> None:
    """The commonest way a move quietly destroys something: the destination is a
    directory that already holds a file of that name. Nothing in the command
    says so, and the preview did not either."""
    from victor.safety.preview import preview

    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "notes.txt").write_text("precious", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("new", encoding="utf-8")

    rendered = preview("mv notes.txt dir/", tmp_path)
    assert "replacing what is already there" in rendered
    assert "notes.txt" in rendered


def test_moving_into_a_directory_with_no_collision_stays_quiet(tmp_path: Path) -> None:
    """The warning has to mean something, so it must not always appear."""
    from victor.safety.preview import preview

    (tmp_path / "dir").mkdir()
    rendered = preview("mv notes.txt dir/", tmp_path)
    assert "replacing" not in rendered
    assert rendered == "would move notes.txt to dir/"


def test_dry_run_falls_back_to_the_command_when_it_cannot_predict() -> None:
    gate = interceptor(dry_run=True)
    review = gate.review(SHELL, {"command": "make deploy"})

    assert review.decision is Decision.DENY
    assert "would run: make deploy" in review.reason


def test_dry_run_still_lets_reads_through() -> None:
    """A dry run must still be able to investigate, or it answers nothing."""
    gate = interceptor(dry_run=True)
    assert gate.review(SHELL, {"command": "git status"}).decision is Decision.ALLOW


def test_a_tripped_kill_switch_blocks_everything() -> None:
    switch = KillSwitch()
    gate = interceptor(kill_switch=switch)
    switch.trip("test")

    assert gate.review(SHELL, {"command": "ls"}).decision is Decision.DENY


def test_confirmation_is_remembered_within_a_session() -> None:
    """Re-asking for an identical retry trains the user to stop reading."""
    confirmer = AutoConfirmer(True)
    gate = interceptor(confirmer=confirmer)

    gate.review(SHELL, {"command": "rm build.log"})
    gate.review(SHELL, {"command": "rm build.log"})
    assert len(confirmer.requests) == 1


def test_disabling_confirmation_is_honoured_but_counted() -> None:
    gate = interceptor(require_confirmation=False)

    assert gate.review(SHELL, {"command": "rm x"}).decision is Decision.ALLOW
    assert gate.stats.confirmed == 0


def test_a_direct_delete_is_declared_irreversible() -> None:
    """With no trash configured, `rm` really is permanent - say so."""
    confirmer = AutoConfirmer(True)
    gate = interceptor(confirmer=confirmer)
    gate.review(SHELL, {"command": "rm notes.txt"})

    hint = confirmer.requests[0].undo_hint
    assert "cannot be undone" in hint
    assert "ran directly" in hint


def test_a_trashed_delete_is_declared_recoverable(tmp_path: Path) -> None:
    """A user who thinks a delete is permanent answers a different question."""
    from victor.safety.trash import Trash

    (tmp_path / "notes.txt").write_text("data", encoding="utf-8")
    confirmer = AutoConfirmer(True)
    gate = interceptor(
        confirmer=confirmer, trash=Trash(tmp_path / "trash", "s1"), cwd=tmp_path
    )
    gate.review(SHELL, {"command": "rm notes.txt"})

    hint = confirmer.requests[0].undo_hint
    assert "victor undo" in hint
    assert "off the disk" in hint
    assert "cannot be undone" not in hint


def test_the_request_offers_undo_when_one_exists() -> None:
    confirmer = AutoConfirmer(True)
    gate = interceptor(confirmer=confirmer)
    gate.review(ToolSpec("git", "g", {}, mutating=True), {"subcommand": "add", "args": ["x"]})

    assert "unstage" in confirmer.requests[0].undo_hint


def test_stats_track_what_happened() -> None:
    gate = interceptor(confirmer=AutoConfirmer(True))
    gate.review(SHELL, {"command": "ls"})
    gate.review(SHELL, {"command": "rm a"})
    gate.review(SHELL, {"command": "rm -rf /"})

    assert gate.stats.reviewed == 3
    assert gate.stats.allowed == 1
    assert gate.stats.confirmed == 1
    assert gate.stats.denied == 1


# --- confirmation -------------------------------------------------------


@pytest.mark.parametrize("answer", ["yes", "Yes", "yeah", "y", "go ahead", "do it", "yes, please"])
def test_affirmatives_are_understood(answer: str) -> None:
    assert interpret(answer) is True


@pytest.mark.parametrize("answer", ["no", "No", "nope", "stop", "cancel", "n", "no thanks"])
def test_negatives_are_understood(answer: str) -> None:
    assert interpret(answer) is False


@pytest.mark.parametrize("answer", ["", "   ", "maybe", "what", "banana", "go"])
def test_ambiguous_answers_are_not_a_yes(answer: str) -> None:
    """"No" misheard as "go" must not run a delete the user refused."""
    assert interpret(answer) is None


def test_typed_confirmer_refuses_without_a_terminal() -> None:
    """Piped stdin cannot answer, so the answer is no."""
    confirmer = TypedConfirmer(stream=io.StringIO("yes\n"), output=io.StringIO())
    request = ConfirmRequest("shell", "rm x", Classification(Risk.CONFIRM, "deletes"))

    assert confirmer.confirm(request) is False


def test_default_confirmer_is_deny_when_nobody_can_be_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert isinstance(build_confirmer(interactive=True), DenyingConfirmer)


class FakePipeline:
    """A voice pipeline that says things and returns scripted transcripts."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def listen(self, mode, **kwargs):
        from victor.voice.stt import Transcript

        text = self.answers.pop(0) if self.answers else ""

        class Turn:
            pass

        turn = Turn()
        turn.transcript = Transcript(text, "test", 1.0, 1.0)
        turn.text = text
        return turn


def test_spoken_confirmation_accepts_a_yes() -> None:
    pytest.importorskip("numpy")  # confirming out loud is the voice stack
    pipeline = FakePipeline("yes")
    confirmer = SpokenConfirmer(pipeline, fallback=DenyingConfirmer())
    request = ConfirmRequest("shell", "rm build.log", Classification(Risk.CONFIRM, "deletes files"))

    assert confirmer.confirm(request) is True
    assert "rm build.log" in pipeline.spoken[0]
    assert "Say yes to continue" in pipeline.spoken[0]


def test_spoken_confirmation_accepts_a_no() -> None:
    pytest.importorskip("numpy")  # confirming out loud is the voice stack
    confirmer = SpokenConfirmer(FakePipeline("no"), fallback=DenyingConfirmer())
    request = ConfirmRequest("shell", "rm x", Classification(Risk.CONFIRM, "deletes"))

    assert confirmer.confirm(request) is False


def test_spoken_confirmation_reasks_then_falls_back() -> None:
    pytest.importorskip("numpy")  # confirming out loud is the voice stack
    pipeline = FakePipeline("banana", "gibberish")
    confirmer = SpokenConfirmer(pipeline, fallback=DenyingConfirmer())
    request = ConfirmRequest("shell", "rm x", Classification(Risk.CONFIRM, "deletes"))

    assert confirmer.confirm(request) is False
    assert any("did not catch" in line for line in pipeline.spoken)


# --- the kill switch ------------------------------------------------------


def test_kill_switch_trips_once_and_records_the_source() -> None:
    switch = KillSwitch()
    first = switch.trip("Ctrl-C")
    second = switch.trip("hotkey")

    assert switch.tripped
    assert first is second  # the first trip wins
    assert switch.trip_record.source == "Ctrl-C"


def test_kill_switch_check_raises() -> None:
    from victor.safety import Aborted

    switch = KillSwitch()
    switch.check()  # no-op while armed
    switch.trip("test")
    with pytest.raises(Aborted, match="stopped by test"):
        switch.check()


def test_kill_switch_observers_fire() -> None:
    switch = KillSwitch()
    seen = []
    switch.observe(seen.append)
    switch.trip("test")

    assert len(seen) == 1


def test_a_raising_observer_does_not_block_the_abort() -> None:
    switch = KillSwitch()
    switch.observe(lambda trip: (_ for _ in ()).throw(RuntimeError("bad observer")))
    switch.trip("test")

    assert switch.tripped


def test_kill_switch_can_be_rearmed() -> None:
    switch = KillSwitch()
    switch.trip("test")
    switch.reset()

    assert not switch.tripped
    switch.check()


def test_stop_phrases_are_recognised() -> None:
    from victor.safety import is_stop_phrase

    assert is_stop_phrase("stop")
    assert is_stop_phrase("Stop.")
    assert is_stop_phrase("cancel")
    assert is_stop_phrase("stop that")
    assert not is_stop_phrase("stopwatch")
    assert not is_stop_phrase("what is the status")


def test_kill_switch_stops_a_running_command(tmp_path: Path) -> None:
    """The exit gate's 200ms claim, measured rather than asserted."""
    switch = KillSwitch()
    tool = ShellTool(cwd=tmp_path, timeout=30.0, kill_switch=switch)

    tripped_at: list[float] = []

    def trip_soon() -> None:
        time.sleep(0.3)
        tripped_at.append(time.monotonic())
        switch.trip("test")

    threading.Thread(target=trip_soon, daemon=True).start()
    result = tool.run("sleep 20")
    returned_at = time.monotonic()

    assert not result.ok
    assert result.metadata.get("aborted") is True
    latency_ms = (returned_at - tripped_at[0]) * 1000
    # The switch itself is a flag check on a 50 ms poll; what varies by platform
    # is how long the shell takes to die. PowerShell's spawn and teardown put
    # Windows around 400 ms where macOS is around 26 ms, and that is a property
    # of the shell rather than a regression in the kill switch - so the budget
    # is per-platform rather than one number that would have to be loose enough
    # for Windows and therefore meaningless on Unix.
    budget_ms = 800 if platform.system() == "Windows" else 200
    assert latency_ms < budget_ms, f"abort took {latency_ms:.0f}ms (budget {budget_ms}ms)"


# --- the journal ----------------------------------------------------------


def test_journal_records_an_action(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "journal.jsonl", session="s1")
    entry = journal.record(
        "shell", {"command": "rm x"}, risk=Risk.CONFIRM, decision="allow", ok=True
    )

    assert entry.id
    assert list(journal)[0].tool == "shell"


def test_journal_survives_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    ActionJournal(path).record("shell", {"command": "mkdir x"}, risk=Risk.CONFIRM, decision="allow")

    assert len(list(ActionJournal(path))) == 1


def test_journal_skips_a_corrupt_line(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = ActionJournal(path)
    journal.record("shell", {"command": "mkdir a"}, risk=Risk.CONFIRM, decision="allow")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ truncated\n")
    journal.record("shell", {"command": "mkdir b"}, risk=Risk.CONFIRM, decision="allow")

    assert len(list(journal)) == 2


def test_a_direct_delete_gets_no_undo_recipe(tmp_path: Path) -> None:
    """A delete that bypassed the trash cannot be walked back, and says so."""
    journal = ActionJournal(tmp_path / "j.jsonl")
    entry = journal.record(
        "shell", {"command": "rm notes.txt"}, risk=Risk.CONFIRM, decision="allow"
    )

    assert entry.undo is None
    assert "ran directly" in entry.no_undo_reason
    assert not entry.reversible


def test_a_trashed_delete_gets_a_restore_recipe(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "j.jsonl")
    entry = journal.record(
        "shell",
        {"command": "rm notes.txt"},
        risk=Risk.CONFIRM,
        decision="allow",
        metadata={
            "trashed": [
                {
                    "original": str(tmp_path / "notes.txt"),
                    "stored": str(tmp_path / "trash" / "ab_notes.txt"),
                    "size": 4,
                    "is_dir": False,
                }
            ]
        },
    )

    assert entry.reversible
    assert entry.undo is not None
    assert entry.undo.tool == "trash"
    assert "restore 1 item" in entry.undo.description


def test_failed_actions_get_no_undo(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "j.jsonl")
    entry = journal.record(
        "git", {"subcommand": "add", "args": ["x"]}, risk=Risk.CONFIRM, decision="allow", ok=False
    )

    assert entry.undo is None
    assert "did not succeed" in entry.no_undo_reason


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ({"subcommand": "add", "args": ["file.py"]}, "reset"),
        ({"subcommand": "commit", "args": ["-m", "x"]}, "reset"),
        ({"subcommand": "stash", "args": []}, "stash"),
        ({"subcommand": "tag", "args": ["v1"]}, "tag"),
    ],
)
def test_git_undo_recipes(arguments: dict, expected: str) -> None:
    undo, _ = plan_undo("git", arguments)

    assert undo is not None
    assert undo.arguments["subcommand"] == expected


def test_git_operations_without_an_inverse_say_so() -> None:
    undo, reason = plan_undo("git", {"subcommand": "push", "args": ["origin", "main"]})

    assert undo is None
    assert "no exact inverse" in reason


def test_mkdir_can_be_undone() -> None:
    undo, _ = plan_undo("shell", {"command": "mkdir newdir"})

    assert undo is not None
    assert "rmdir" in undo.arguments["command"]


def test_touch_is_deliberately_not_undone() -> None:
    """Deleting a touched file could destroy content written since."""
    undo, reason = plan_undo("shell", {"command": "touch notes.txt"})

    assert undo is None
    assert "could destroy content" in reason


# --- undo execution -------------------------------------------------------


def test_undo_reverses_a_git_add(tmp_path: Path, settings: Settings) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")

    journal = ActionJournal(tmp_path / "j.jsonl")
    registry = build_registry(
        settings, cwd=tmp_path, interceptor=interceptor(confirmer=AutoConfirmer(True))
    )
    registry.run("git", {"subcommand": "add", "args": ["a.txt"]})
    entry = journal.record(
        "git", {"subcommand": "add", "args": ["a.txt"]}, risk=Risk.CONFIRM, decision="allow"
    )

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "a.txt" in staged.stdout

    result = undo_entry(journal, registry, entry)
    assert result.ran and result.error is None

    staged_after = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "a.txt" not in staged_after.stdout
    assert journal.get(entry.id).undone_at is not None


def test_undo_refuses_an_irreversible_entry(tmp_path: Path, settings: Settings) -> None:
    journal = ActionJournal(tmp_path / "j.jsonl")
    entry = journal.record("shell", {"command": "rm x"}, risk=Risk.CONFIRM, decision="allow")
    registry = build_registry(settings, cwd=tmp_path, interceptor=interceptor())

    result = undo_entry(journal, registry, entry)
    assert not result.ran
    assert "cannot be restored" in (result.error or "")


def test_undo_last_finds_the_most_recent_reversible_action(
    tmp_path: Path, settings: Settings
) -> None:
    journal = ActionJournal(tmp_path / "j.jsonl")
    journal.record("shell", {"command": "mkdir keep"}, risk=Risk.CONFIRM, decision="allow")
    journal.record("shell", {"command": "rm gone.txt"}, risk=Risk.CONFIRM, decision="allow")

    assert journal.last_reversible().arguments["command"] == "mkdir keep"


def test_undo_last_returns_none_when_nothing_is_reversible(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "j.jsonl")
    journal.record("shell", {"command": "rm x"}, risk=Risk.CONFIRM, decision="allow")

    assert journal.last_reversible() is None


def test_undo_actually_removes_a_created_directory(
    tmp_path: Path, settings: Settings
) -> None:
    journal = ActionJournal(tmp_path / "j.jsonl")
    registry = build_registry(
        settings, cwd=tmp_path, interceptor=interceptor(confirmer=AutoConfirmer(True))
    )
    registry.run("shell", {"command": "mkdir newdir"})
    journal.record("shell", {"command": "mkdir newdir"}, risk=Risk.CONFIRM, decision="allow")
    assert (tmp_path / "newdir").is_dir()

    results = undo_last(journal, registry)
    assert results and results[0].ran
    assert not (tmp_path / "newdir").exists()


def test_the_kill_switch_reaps_the_whole_process_tree(tmp_path: Path) -> None:
    """The exit gate says "no orphaned processes".

    A shell that spawns a background child is the ordinary case - `make test`
    spawns pytest which spawns workers. Killing only the direct child would
    leave the grandchild running.
    """
    marker = tmp_path / "grandchild-alive"
    # Written in Python rather than shell so it runs on both platforms. The
    # original used a POSIX subshell and `&`, which PowerShell cannot parse -
    # and skipping the test on Windows would leave the one path that differs
    # most between the two (taskkill /T versus killpg) completely untested,
    # which is the class of gap that produced the P5 Windows defects.
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        "import pathlib, sys, time\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "while True:\n"
        "    marker.touch()\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    spawner = tmp_path / "spawner.py"
    spawner.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    script = _shell_invocation(sys.executable, str(spawner), str(grandchild), str(marker))
    switch = KillSwitch()
    tool = ShellTool(cwd=tmp_path, timeout=30.0, kill_switch=switch)

    threading.Thread(
        target=lambda: (time.sleep(0.4), switch.trip("test")), daemon=True
    ).start()
    result = tool.run(script)
    assert result.metadata.get("aborted") is True

    # If the grandchild survived it keeps refreshing the marker's mtime.
    marker.unlink(missing_ok=True)
    time.sleep(0.4)
    assert not marker.exists(), "a grandchild process outlived the kill switch"


def test_the_spoken_prompt_is_punctuated_into_sentences(tmp_path: Path) -> None:
    """Piper emits one chunk per sentence, so full stops are what let a long
    prompt start playing before it has finished generating."""
    from victor.safety.trash import Trash

    for name in ("a.log", "b.log"):
        (tmp_path / name).write_text("x" * 100, encoding="utf-8")

    confirmer = AutoConfirmer(True)
    gate = interceptor(
        confirmer=confirmer, trash=Trash(tmp_path / "trash", "s1"), cwd=tmp_path
    )
    gate.review(SHELL, {"command": "rm *.log"})

    spoken = confirmer.requests[0].spoken()
    assert ".." not in spoken
    assert "more This" not in spoken  # the missing full stop this guards
    assert spoken.count(".") >= 3
    assert spoken.endswith("Say yes to continue, or no to stop.")
