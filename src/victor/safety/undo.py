"""Executing an undo recipe.

Command-shaped undos run back through the ordinary tool path, so they are
classified and journalled like anything else. Reversing a mistake is still an
action on the user's machine, and exempting it from the safety layer would make
"undo" the one unguarded way to run a command.

Restoring from the trash is the exception, and deliberately so. It is not a
command the model composed - it is a manifest this process wrote when it moved
the files, and every step is a `move` back to a path recorded at the time. There
is nothing for a classifier to adjudicate, and routing it through the shell
would mean building a delete-shaped command string to undo a delete.
"""

from __future__ import annotations

from ..tools import ToolRegistry
from .journal import ActionJournal, Entry, UndoResult
from .trash import Trash, TrashedItem


def undo_entry(
    journal: ActionJournal, registry: ToolRegistry, entry: Entry
) -> UndoResult:
    """Reverse one journalled action."""
    if entry.undo is None:
        return UndoResult(
            entry=entry,
            ran=False,
            error=entry.no_undo_reason or "this action has no undo recipe",
        )
    if entry.undone_at is not None:
        return UndoResult(entry=entry, ran=False, error=f"already undone at {entry.undone_at}")

    recipe = entry.undo
    if recipe.tool == "trash":
        return _restore_from_trash(journal, entry)

    result = registry.run(recipe.tool, recipe.arguments)
    if not result.ok:
        return UndoResult(
            entry=entry,
            ran=True,
            output=result.output,
            error=result.error or "the undo command failed",
            steps=[recipe.description],
        )

    journal.mark_undone(entry.id)
    return UndoResult(
        entry=entry,
        ran=True,
        output=result.output,
        steps=[recipe.description],
    )


def _restore_from_trash(journal: ActionJournal, entry: Entry) -> UndoResult:
    """Move trashed paths back where they came from.

    Restores in reverse order so a directory and something inside it come back
    in the right sequence, and reports partial success honestly: if three files
    return and one is blocked by something now occupying its path, that is four
    separate facts the user needs.
    """
    assert entry.undo is not None
    raw = entry.undo.arguments.get("items") or []
    items = [TrashedItem.from_json(i) for i in raw]

    restored: list[str] = []
    failures: list[str] = []

    for item in reversed(items):
        try:
            Trash.restore(item)
        except (FileNotFoundError, FileExistsError, OSError) as exc:
            failures.append(f"{item.original}: {exc}")
        else:
            restored.append(item.original)

    steps = [f"restored {p}" for p in restored]
    if failures:
        return UndoResult(
            entry=entry,
            ran=True,
            output="\n".join(steps),
            error="could not restore: " + "; ".join(failures),
            steps=steps,
        )

    journal.mark_undone(entry.id)
    return UndoResult(
        entry=entry,
        ran=True,
        output="\n".join(steps) or "nothing to restore",
        steps=steps,
    )


def undo_last(
    journal: ActionJournal, registry: ToolRegistry, count: int = 1
) -> list[UndoResult]:
    """Reverse the most recent ``count`` reversible actions, newest first."""
    results: list[UndoResult] = []
    for _ in range(max(1, count)):
        entry = journal.last_reversible()
        if entry is None:
            break
        result = undo_entry(journal, registry, entry)
        results.append(result)
        if result.error:
            break  # stop at the first failure rather than compounding it
    return results
