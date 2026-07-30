from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

from victor.config import Settings
from victor.safety import ActionJournal, AutoConfirmer, SafetyInterceptor, undo_last
from victor.safety.trash import Trash, describe, parse_delete
from victor.tools import build_registry


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- recognising a delete -------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["rm notes.txt", "rm -f notes.txt", "rm -rf notes.txt", "rm -- notes.txt", "del notes.txt"],
)
def test_plain_deletes_are_recognised(command: str, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    plan = parse_delete(command, tmp_path)

    assert plan is not None
    assert plan.paths == [(tmp_path / "notes.txt").resolve()]


@pytest.mark.parametrize(
    "command",
    [
        "rm notes.txt && echo done",
        "rm $(cat list.txt)",
        "find . -name '*.tmp' | xargs rm",
        "rm notes.txt > log.txt",
        "ls -la",
        "git rm cached.txt",
    ],
)
def test_complex_commands_are_left_alone(command: str, tmp_path: Path) -> None:
    """A partial understanding of a destructive command is worse than none."""
    assert parse_delete(command, tmp_path) is None


def test_globs_are_expanded(tmp_path: Path) -> None:
    for name in ("a.log", "b.log", "keep.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    plan = parse_delete("rm *.log", tmp_path)
    assert plan is not None
    assert {p.name for p in plan.paths} == {"a.log", "b.log"}


def test_missing_paths_are_dropped(tmp_path: Path) -> None:
    assert parse_delete("rm ghost.txt", tmp_path) is None


def test_recursive_flag_is_detected(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    plan = parse_delete("rm -rf dir", tmp_path)
    assert plan is not None and plan.recursive


def test_describe_summarises_the_damage(tmp_path: Path) -> None:
    for i in range(8):
        (tmp_path / f"f{i}.log").write_text("x" * 100, encoding="utf-8")

    plan = parse_delete("rm *.log", tmp_path)
    assert plan is not None
    summary = describe(plan)
    assert "8 items" in summary
    assert "and 3 more" in summary


# --- storing and restoring ------------------------------------------------


def test_a_file_survives_a_round_trip_byte_identical(tmp_path: Path) -> None:
    """The P3 exit gate: delete, undo, restored byte-identical."""
    target = tmp_path / "important.bin"
    target.write_bytes(os.urandom(4096))
    before = digest(target)

    trash = Trash(tmp_path / ".trash", "s1")
    item = trash.store(target)

    assert not target.exists()
    assert Path(item.stored).exists()

    Trash.restore(item)
    assert target.exists()
    assert digest(target) == before


def test_a_directory_survives_a_round_trip(tmp_path: Path) -> None:
    tree = tmp_path / "project"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    before = digest(tree / "src" / "main.py")

    trash = Trash(tmp_path / ".trash", "s1")
    item = trash.store(tree)
    assert not tree.exists()

    Trash.restore(item)
    assert digest(tree / "src" / "main.py") == before


def test_same_named_files_do_not_collide(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "config.json").write_text("A", encoding="utf-8")
    (tmp_path / "b" / "config.json").write_text("B", encoding="utf-8")

    trash = Trash(tmp_path / ".trash", "s1")
    first = trash.store(tmp_path / "a" / "config.json")
    second = trash.store(tmp_path / "b" / "config.json")

    assert first.stored != second.stored
    Trash.restore(first)
    Trash.restore(second)
    assert (tmp_path / "a" / "config.json").read_text() == "A"
    assert (tmp_path / "b" / "config.json").read_text() == "B"


def test_restore_refuses_to_clobber(tmp_path: Path) -> None:
    """Something newer now occupies the path; it wins over the deleted thing."""
    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")

    trash = Trash(tmp_path / ".trash", "s1")
    item = trash.store(target)
    target.write_text("new", encoding="utf-8")

    with pytest.raises(FileExistsError):
        Trash.restore(item)
    assert target.read_text() == "new"


def test_restore_recreates_a_missing_parent(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "path" / "file.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("x", encoding="utf-8")

    trash = Trash(tmp_path / ".trash", "s1")
    item = trash.store(nested)
    import shutil

    shutil.rmtree(tmp_path / "deep")

    Trash.restore(item)
    assert nested.read_text() == "x"


def test_manifest_records_every_move(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    trash = Trash(tmp_path / ".trash", "s1")
    trash.store_all([tmp_path / "a.txt", tmp_path / "b.txt"])

    assert len(list(trash.entries())) == 2


# --- housekeeping ---------------------------------------------------------


def test_old_sessions_are_pruned(tmp_path: Path) -> None:
    root = tmp_path / ".trash"
    old = root / "20200101T000000"
    old.mkdir(parents=True)
    (old / "junk").write_text("x", encoding="utf-8")
    os.utime(old, (time.time() - 30 * 86400,) * 2)

    removed = Trash(root, "current", retention_days=7).prune()
    assert removed == 1
    assert not old.exists()


def test_pruning_never_touches_the_running_session(tmp_path: Path) -> None:
    root = tmp_path / ".trash"
    trash = Trash(root, "current", retention_days=0, max_bytes=0)
    (tmp_path / "f.txt").write_text("x" * 1000, encoding="utf-8")
    trash.store(tmp_path / "f.txt")

    trash.prune()
    assert trash.session_dir.exists()


def test_size_cap_drops_oldest_first(tmp_path: Path) -> None:
    root = tmp_path / ".trash"
    for i, name in enumerate(("old", "newer")):
        d = root / name
        d.mkdir(parents=True)
        (d / "blob").write_bytes(b"x" * 2000)
        os.utime(d, (time.time() - (10 - i) * 86400,) * 2)

    Trash(root, "current", retention_days=365, max_bytes=1000).prune()
    assert not (root / "old").exists()


# --- through the tool path ------------------------------------------------


def test_shell_delete_is_rerouted_and_undoable(
    tmp_path: Path, settings: Settings
) -> None:
    """End to end: the agent deletes, `undo` puts it back byte-identical."""
    target = tmp_path / "report.txt"
    target.write_bytes(os.urandom(2048))
    before = digest(target)

    trash = Trash(tmp_path / ".trash", "s1")
    journal = ActionJournal(tmp_path / "journal.jsonl", session="s1")
    gate = SafetyInterceptor(
        confirmer=AutoConfirmer(True), journal=journal, trash=trash, cwd=tmp_path
    )
    registry = build_registry(settings, cwd=tmp_path, interceptor=gate, trash=trash)

    result = registry.run("shell", {"command": "rm report.txt"})
    assert result.ok
    assert not target.exists()
    assert "trash" in result.output

    entry = journal.last_reversible()
    assert entry is not None and entry.undo is not None
    assert entry.undo.tool == "trash"

    results = undo_last(journal, registry)
    assert results and results[0].ran and results[0].error is None
    assert target.exists()
    assert digest(target) == before


def test_undoing_twice_is_refused(tmp_path: Path, settings: Settings) -> None:
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    trash = Trash(tmp_path / ".trash", "s1")
    journal = ActionJournal(tmp_path / "journal.jsonl", session="s1")
    gate = SafetyInterceptor(
        confirmer=AutoConfirmer(True), journal=journal, trash=trash, cwd=tmp_path
    )
    registry = build_registry(settings, cwd=tmp_path, interceptor=gate, trash=trash)

    registry.run("shell", {"command": "rm f.txt"})
    assert undo_last(journal, registry)[0].ran
    assert undo_last(journal, registry) == []  # nothing reversible left


def test_a_complex_delete_still_runs_for_real(
    tmp_path: Path, settings: Settings
) -> None:
    """Commands the parser declines are executed exactly as written."""
    (tmp_path / "a.log").write_text("x", encoding="utf-8")
    trash = Trash(tmp_path / ".trash", "s1")
    gate = SafetyInterceptor(confirmer=AutoConfirmer(True), trash=trash, cwd=tmp_path)
    registry = build_registry(settings, cwd=tmp_path, interceptor=gate, trash=trash)

    # `;` rather than `&&`: both are in the parser's too-complex set, but `&&`
    # is a parse error in PowerShell 5.1, which is the shell this tool uses on
    # Windows. The test is about the parser declining, not about the operator.
    result = registry.run("shell", {"command": "rm a.log ; echo removed"})
    assert result.ok
    assert not (tmp_path / "a.log").exists()
    assert "trash" not in result.output


def test_deletes_still_require_confirmation(tmp_path: Path, settings: Settings) -> None:
    """Recoverable is not the same as unremarkable - it is still a delete."""
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    trash = Trash(tmp_path / ".trash", "s1")
    confirmer = AutoConfirmer(False)
    gate = SafetyInterceptor(confirmer=confirmer, trash=trash, cwd=tmp_path)
    registry = build_registry(settings, cwd=tmp_path, interceptor=gate, trash=trash)

    result = registry.run("shell", {"command": "rm f.txt"})
    assert not result.ok
    assert (tmp_path / "f.txt").exists()
    assert len(confirmer.requests) == 1


def test_the_reported_size_is_measured_before_the_move(
    tmp_path: Path, settings: Settings
) -> None:
    """describe() after store_all() would report 0 B - nothing is there any more."""
    (tmp_path / "blob.bin").write_bytes(b"x" * 4096)
    trash = Trash(tmp_path / ".trash", "s1")
    gate = SafetyInterceptor(confirmer=AutoConfirmer(True), trash=trash, cwd=tmp_path)
    registry = build_registry(settings, cwd=tmp_path, interceptor=gate, trash=trash)

    result = registry.run("shell", {"command": "rm blob.bin"})
    assert result.ok
    assert "0 B" not in result.output
    assert "4.0 KB" in result.output


# --- the name the user typed, not the file at the far end -------------------


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs privilege on Windows")
def test_deleting_a_symlink_leaves_its_target_alone(
    tmp_path: Path, settings: Settings
) -> None:
    """`rm shortcut.txt` removes the link. It does not touch what it points at.

    `_expand` and `Trash.store` both called `resolve()`, which follows the final
    component - so rerouting the delete through the trash changed what the
    command meant. The old code reported "Moved 1 item (13 B): important.txt",
    left the link in place, and took the only copy of a file the user had not
    named. This is the one bug in this module that destroys data the user was
    not shown.
    """
    target = tmp_path / "important.txt"
    target.write_text("the only copy", encoding="utf-8")
    link = tmp_path / "shortcut.txt"
    link.symlink_to(target)

    trash = Trash(tmp_path / ".trash", "s1")
    gate = SafetyInterceptor(confirmer=AutoConfirmer(True), trash=trash, cwd=tmp_path)
    registry = build_registry(settings, cwd=tmp_path, interceptor=gate, trash=trash)

    result = registry.run("shell", {"command": "rm shortcut.txt"})

    assert result.ok, result.error
    assert "shortcut.txt" in result.output
    assert "important.txt" not in result.output
    assert not link.is_symlink(), "the link itself was not removed"
    assert target.read_text(encoding="utf-8") == "the only copy", "the target was destroyed"


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs privilege on Windows")
def test_a_restored_symlink_is_still_a_symlink(tmp_path: Path) -> None:
    """Undo has to put back what was taken, which was a link and not a copy."""
    target = tmp_path / "important.txt"
    target.write_text("the only copy", encoding="utf-8")
    link = tmp_path / "shortcut.txt"
    link.symlink_to(target)

    trash = Trash(tmp_path / ".trash", "s1")
    item = trash.store(link)
    assert not link.is_symlink()

    Trash.restore(item)
    assert link.is_symlink()
    assert link.resolve() == target.resolve()


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs privilege on Windows")
def test_a_dangling_symlink_is_still_something_rm_removes(tmp_path: Path) -> None:
    """`exists()` follows the link, so a broken one looked like nothing at all -
    and the delete went out to the shell with no trash behind it."""
    link = tmp_path / "broken.txt"
    link.symlink_to(tmp_path / "never-existed.txt")

    plan = parse_delete("rm broken.txt", tmp_path)

    assert plan is not None, "a broken link was not recognised as a delete"
    assert [p.name for p in plan.paths] == ["broken.txt"]


def test_a_filename_containing_glob_characters_still_gets_a_trash(
    tmp_path: Path,
) -> None:
    """`file[1].txt` is an ordinary filename; `[1]` is a character class that
    matches "1" instead of itself. The glob found nothing, so the delete was
    never rerouted - it went to the shell unbacked, and the preview gave up and
    echoed the command back."""
    awkward = tmp_path / "file[1].txt"
    awkward.write_text("x" * 50, encoding="utf-8")

    plan = parse_delete('rm "file[1].txt"', tmp_path)

    assert plan is not None, "the delete was not recognised, so nothing backed it up"
    assert [p.name for p in plan.paths] == ["file[1].txt"]
    assert "file[1].txt" in describe(plan)


def test_everything_after_a_double_dash_is_a_filename(tmp_path: Path) -> None:
    """POSIX says so, and `--` is used for exactly this: a name starting with a
    dash. It was dropped as a flag, so the file the user needed `--` for was the
    one left behind - and the delete was still reported as done."""
    (tmp_path / "-report.txt").write_text("x" * 50, encoding="utf-8")
    (tmp_path / "data.txt").write_text("x" * 50, encoding="utf-8")

    plan = parse_delete('rm -- "-report.txt" data.txt', tmp_path)

    assert plan is not None
    assert sorted(p.name for p in plan.paths) == ["-report.txt", "data.txt"]
