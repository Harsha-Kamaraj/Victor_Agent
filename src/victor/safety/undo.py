"""Executing an undo recipe.

Undo runs the recorded inverse through the ordinary tool path, which means it
is classified and journalled like anything else. Reversing a mistake is still
an action on the user's machine, and exempting it from the safety layer would
make "undo" the one unguarded way to run a command.
"""

from __future__ import annotations

from ..tools import ToolRegistry
from .journal import ActionJournal, Entry, UndoResult


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


def undo_last(journal: ActionJournal, registry: ToolRegistry) -> UndoResult | None:
    """Reverse the most recent reversible action, if there is one."""
    entry = journal.last_reversible()
    if entry is None:
        return None
    return undo_entry(journal, registry, entry)
