"""The action journal.

Every action the agent executes is recorded here, whether it succeeded or not,
along with an undo recipe **where a real inverse exists**.

That qualifier is the whole design. It is tempting to present "undo" as a
general capability, but most side effects have no inverse: `rm` does not, and
neither does anything that reached the network. Recording a fake undo for those
would be worse than recording none, because it would encourage confirming a
delete on the belief that it can be walked back.

So entries carry either a recipe or an explicit reason why there is not one,
and `victor journal` shows the difference. Confirmation is the real protection
for irreversible actions; undo is a convenience for the reversible ones.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .classify import Risk

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Undo:
    """How to reverse one action."""

    tool: str
    arguments: dict[str, Any]
    description: str

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "description": self.description,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Undo:
        return cls(
            tool=raw["tool"],
            arguments=raw.get("arguments", {}),
            description=raw.get("description", ""),
        )


@dataclass(slots=True)
class Entry:
    """One executed action."""

    id: str
    ts: str
    session: str
    tool: str
    arguments: dict[str, Any]
    risk: str
    decision: str
    reason: str = ""
    ok: bool = True
    undo: Undo | None = None
    no_undo_reason: str = ""
    undone_at: str | None = None

    @property
    def reversible(self) -> bool:
        return self.undo is not None and self.undone_at is None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "session": self.session,
            "tool": self.tool,
            "arguments": self.arguments,
            "risk": self.risk,
            "decision": self.decision,
            "reason": self.reason,
            "ok": self.ok,
            "undo": self.undo.to_json() if self.undo else None,
            "no_undo_reason": self.no_undo_reason,
            "undone_at": self.undone_at,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Entry:
        undo = raw.get("undo")
        return cls(
            id=raw["id"],
            ts=raw["ts"],
            session=raw.get("session", ""),
            tool=raw["tool"],
            arguments=raw.get("arguments", {}),
            risk=raw.get("risk", ""),
            decision=raw.get("decision", ""),
            reason=raw.get("reason", ""),
            ok=bool(raw.get("ok", True)),
            undo=Undo.from_json(undo) if undo else None,
            no_undo_reason=raw.get("no_undo_reason", ""),
            undone_at=raw.get("undone_at"),
        )


class ActionJournal:
    """Append-only log of executed actions, with undo where one exists."""

    def __init__(self, path: Path, *, session: str = "") -> None:
        self.path = path
        self.session = session
        self._lock = threading.Lock()

    # -- writing -----------------------------------------------------------

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        risk: Risk | str,
        decision: str,
        reason: str = "",
        ok: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Entry:
        """Append one action. Returns the entry, including its undo plan."""
        undo, why_not = plan_undo(tool, arguments, ok=ok, metadata=metadata)
        entry = Entry(
            id=uuid.uuid4().hex[:12],
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            session=self.session,
            tool=tool,
            arguments=arguments,
            risk=str(risk),
            decision=decision,
            reason=reason,
            ok=ok,
            undo=undo,
            no_undo_reason=why_not,
        )
        self._append(entry)
        return entry

    def _append(self, entry: Entry) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_json(), default=str) + "\n")

    def mark_undone(self, entry_id: str) -> bool:
        """Flag an entry as reversed. Rewrites the file atomically."""
        entries = list(self)
        found = False
        for entry in entries:
            if entry.id == entry_id:
                entry.undone_at = datetime.now(UTC).isoformat(timespec="seconds")
                found = True
        if not found:
            return False

        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for entry in entries:
                        fh.write(json.dumps(entry.to_json(), default=str) + "\n")
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return True

    # -- reading -----------------------------------------------------------

    def __iter__(self) -> Iterator[Entry]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Entry.from_json(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    continue

    def recent(self, limit: int = 20) -> list[Entry]:
        return list(self)[-limit:][::-1]

    def get(self, entry_id: str) -> Entry | None:
        return next((e for e in self if e.id == entry_id), None)

    def last_reversible(self) -> Entry | None:
        """The most recent action that can still be undone."""
        return next((e for e in reversed(list(self)) if e.reversible), None)


# --- undo planning --------------------------------------------------------

#: git subcommands with an exact inverse, given the arguments used.
_GIT_UNDO = {
    # `git reset -- <paths>` rather than `git reset HEAD <paths>`: the latter
    # fails with "ambiguous argument 'HEAD'" in a repository with no commits
    # yet, which is exactly the `git init && git add .` case. An undo recipe
    # that only works sometimes is worse than none, because it is offered in
    # the confirmation prompt as a reason to say yes.
    "add": lambda args: Undo(
        "git",
        {"subcommand": "reset", "args": ["--", *(args or ["."])]},
        "unstage those paths" if args else "unstage everything",
    ),
    "commit": lambda args: Undo(
        "git",
        {"subcommand": "reset", "args": ["--soft", "HEAD~1"]},
        "undo the commit, keeping the changes staged",
    ),
    "stash": lambda args: (
        Undo("git", {"subcommand": "stash", "args": ["pop"]}, "restore the stashed changes")
        if not args or args[0] in {"push", "save"}
        else None
    ),
    "tag": lambda args: (
        Undo("git", {"subcommand": "tag", "args": ["-d", args[0]]}, f"delete tag {args[0]}")
        if args and not args[0].startswith("-")
        else None
    ),
}

#: Why a given command cannot be reversed, phrased for the user.
#:
#: The delete entries apply only when the command was *not* rerouted through
#: the trash - a complex `rm` inside a pipeline, or a session with no trash
#: configured. A trashed delete never reaches here; its manifest produces a
#: restore recipe in :func:`plan_undo` first.
_IRREVERSIBLE = {
    "rm": "this delete ran directly, so the files cannot be restored",
    "rmdir": "this delete ran directly, so the directories cannot be restored",
    "unlink": "this delete ran directly, so the file cannot be restored",
    "dd": "overwritten data cannot be recovered",
    "truncate": "discarded file contents cannot be recovered",
    "kill": "a terminated process cannot be resumed",
    "pkill": "terminated processes cannot be resumed",
    "killall": "terminated processes cannot be resumed",
    "curl": "a network request cannot be recalled",
    "wget": "a network request cannot be recalled",
    "ssh": "commands run on another machine are outside this journal",
}


def plan_undo(
    tool: str,
    arguments: dict[str, Any],
    *,
    ok: bool = True,
    metadata: dict[str, Any] | None = None,
) -> tuple[Undo | None, str]:
    """Work out how to reverse an action, or why it cannot be.

    Returns ``(recipe, reason_if_none)``. Only exact inverses are offered; a
    plausible-looking approximation would be more dangerous than nothing.

    ``metadata`` is the executed tool's result metadata. It is what turns a
    delete from irreversible into reversible: if the command was rerouted
    through the trash, the manifest of moved files is right there.
    """
    if not ok:
        return None, "the action did not succeed, so there is nothing to undo"

    trashed = (metadata or {}).get("trashed")
    if trashed:
        count = len(trashed)
        return (
            Undo(
                "trash",
                {"items": trashed},
                f"restore {count} item{'s' if count != 1 else ''} from the trash",
            ),
            "",
        )

    if tool == "read_file":
        return None, "reading changes nothing"

    if tool == "git":
        subcommand = str(arguments.get("subcommand", ""))
        args = [str(a) for a in arguments.get("args") or []]
        builder = _GIT_UNDO.get(subcommand)
        if builder is None:
            return None, f"git {subcommand} has no exact inverse"
        recipe = builder([a for a in args if not a.startswith("-")])
        if recipe is None:
            return None, f"git {subcommand} with these arguments has no exact inverse"
        return recipe, ""

    if tool == "shell":
        return _plan_shell_undo(str(arguments.get("command", "")))

    return None, f"{tool} has no undo recipe"


def _plan_shell_undo(command: str) -> tuple[Undo | None, str]:
    """Undo for the handful of shell commands with a real inverse."""
    from .classify import _head, _tokenize

    collapsed = " ".join(command.split())
    name, args, _ = _head(_tokenize(collapsed))
    if name is None:
        return None, "could not identify the command"

    positional = [a for a in args if not a.startswith("-")]

    if name == "mkdir" and positional:
        # rmdir only removes empty directories, so this cannot destroy work
        # that arrived after the mkdir.
        return (
            Undo(
                "shell",
                {"command": f"rmdir {' '.join(shlex_quote(p) for p in positional)}"},
                "remove the directories, if they are still empty",
            ),
            "",
        )

    if name == "touch" and positional:
        return (
            None,
            "touch may have created files or only changed timestamps; "
            "deleting them could destroy content written since",
        )

    if name in _IRREVERSIBLE:
        return None, _IRREVERSIBLE[name]

    if name == "git":
        subcommand = next((a for a in args if not a.startswith("-")), "")
        return plan_undo("git", {"subcommand": subcommand, "args": args[1:]})

    return None, f"{name} has no known inverse"


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


@dataclass
class UndoResult:
    """What happened when an undo was attempted."""

    entry: Entry
    ran: bool
    output: str = ""
    error: str | None = None
    steps: list[str] = field(default_factory=list)
