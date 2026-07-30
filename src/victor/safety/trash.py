"""Reversible deletes.

The plan's requirement: *"deletes become moves to `~/.victor/trash/<session>/`;
each entry records its inverse, so `victor undo` replays backward."*

This is the difference between an agent you can recover from and one you cannot.
An LLM that deletes the wrong directory is not a hypothetical, and confirmation
only helps if the user correctly predicts what a glob expands to. Moving instead
of unlinking means a wrong answer to a prompt is embarrassing rather than
permanent.

The rewrite is deliberate and visible: the confirmation prompt says the files
are being moved to the trash and names the command that restores them. Silently
changing what a command does would be worse than not doing it at all.

Two things this does *not* try to be. It is not a general `rm` replacement -
anything with pipes, redirects or command substitution is left alone and runs as
the user wrote it. And it is not unbounded: the trash prunes by age and size, so
it cannot quietly consume the disk it was meant to protect.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_RETENTION_DAYS = 7
DEFAULT_MAX_BYTES = 2 * 1024**3  # 2 GiB
MANIFEST = "manifest.jsonl"

#: Commands whose whole purpose is deletion, per platform.
_DELETE_COMMANDS = {"rm", "del", "erase", "unlink", "rmdir", "Remove-Item"}

#: Anything here means the command does more than delete a list of paths, so it
#: is left exactly as written rather than half-understood.
_TOO_COMPLEX = re.compile(r"[|&;<>`]|\$\(")


@dataclass(frozen=True, slots=True)
class TrashedItem:
    """One path moved into the trash, and where it came from."""

    original: str
    stored: str
    size: int
    is_dir: bool

    def to_json(self) -> dict[str, object]:
        return {
            "original": self.original,
            "stored": self.stored,
            "size": self.size,
            "is_dir": self.is_dir,
        }

    @classmethod
    def from_json(cls, raw: dict) -> TrashedItem:
        return cls(
            original=raw["original"],
            stored=raw["stored"],
            size=int(raw.get("size", 0)),
            is_dir=bool(raw.get("is_dir", False)),
        )


class Trash:
    """A session-scoped holding area for deleted files."""

    def __init__(
        self,
        root: Path,
        session: str = "",
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.root = root
        self.session = session or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self.retention_days = retention_days
        self.max_bytes = max_bytes

    @property
    def session_dir(self) -> Path:
        return self.root / self.session

    # -- storing -----------------------------------------------------------

    def store(self, path: Path) -> TrashedItem:
        """Move ``path`` into the trash and record how to put it back."""
        source = locate(path)
        if not source.exists() and not source.is_symlink():
            raise FileNotFoundError(source)

        self.session_dir.mkdir(parents=True, exist_ok=True)
        # A flat store with a unique prefix: two files called config.json from
        # different directories must not collide.
        stored = self.session_dir / f"{uuid.uuid4().hex[:8]}_{source.name}"
        # A symlink is a thing in its own right: not a directory even when it
        # points at one, and its own size rather than its target's.
        is_dir = source.is_dir() and not source.is_symlink()
        size = 0 if source.is_symlink() else _size_of(source)

        shutil.move(str(source), str(stored))
        item = TrashedItem(
            original=str(source), stored=str(stored), size=size, is_dir=is_dir
        )
        self._append_manifest(item)
        return item

    def store_all(self, paths: list[Path]) -> list[TrashedItem]:
        return [self.store(p) for p in paths]

    def _append_manifest(self, item: TrashedItem) -> None:
        line = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **item.to_json()}
        with (self.session_dir / MANIFEST).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    # -- restoring ---------------------------------------------------------

    @staticmethod
    def restore(item: TrashedItem, *, overwrite: bool = False) -> Path:
        """Put a trashed path back where it came from.

        Refuses to clobber by default: if something now occupies the original
        path, that something is more current than what was deleted.
        """
        stored = Path(item.stored)
        original = Path(item.original)
        if not stored.exists():
            raise FileNotFoundError(f"trash entry is gone: {stored}")
        if original.exists() and not overwrite:
            raise FileExistsError(f"{original} exists again; refusing to overwrite")

        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stored), str(original))
        return original

    # -- housekeeping ------------------------------------------------------

    def entries(self) -> Iterator[TrashedItem]:
        if not self.root.exists():
            return
        for manifest in sorted(self.root.glob(f"*/{MANIFEST}")):
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    yield TrashedItem.from_json(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    continue

    def total_bytes(self) -> int:
        return sum(_size_of(p) for p in self.root.glob("*") if p.is_dir())

    def prune(self) -> int:
        """Drop sessions older than the retention window, then oldest-first
        until the trash fits its size cap. Returns sessions removed."""
        if not self.root.exists():
            return 0

        removed = 0
        cutoff = time.time() - self.retention_days * 86400
        sessions = sorted(
            (p for p in self.root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )

        for session in list(sessions):
            if session.name == self.session:
                continue  # never prune the run in progress
            if session.stat().st_mtime < cutoff:
                shutil.rmtree(session, ignore_errors=True)
                sessions.remove(session)
                removed += 1

        while sessions and self.total_bytes() > self.max_bytes:
            oldest = sessions.pop(0)
            if oldest.name == self.session:
                break
            shutil.rmtree(oldest, ignore_errors=True)
            removed += 1
        return removed


def _size_of(path: Path) -> int:
    # A symlink first, because is_file() and is_dir() both follow it - so the
    # preview for `rm shortcut.txt` quoted the size of the file at the far end,
    # which is the one thing the delete does not touch.
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


# --- recognising a delete -------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeletePlan:
    """A shell command understood well enough to reroute through the trash."""

    command: str
    paths: list[Path]
    recursive: bool

    @property
    def count(self) -> int:
        return len(self.paths)


def parse_delete(command: str, cwd: Path) -> DeletePlan | None:
    """Recognise a plain delete and resolve its targets.

    Returns ``None`` when the command is anything more than "remove these
    paths" - pipes, redirects, substitution, multiple statements. Those run
    unmodified, because a partial understanding of a destructive command is
    more dangerous than none.
    """
    text = command.strip()
    if not text or _TOO_COMPLEX.search(text):
        return None

    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if not tokens:
        return None

    name = tokens[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if name not in _DELETE_COMMANDS:
        return None

    recursive = False
    operands: list[str] = []
    # POSIX: everything after `--` is an operand, however it starts. Without
    # this, `rm -- -report.txt data.txt` dropped -report.txt as though it were a
    # flag and then reported the delete as done - the one file the user needed
    # `--` for was the one silently left behind.
    end_of_flags = False
    for token in tokens[1:]:
        if not end_of_flags and token == "--":
            end_of_flags = True
            continue
        if not end_of_flags and token.startswith("-") and len(token) > 1:
            if re.search(r"[rR]", token.lstrip("-")):
                recursive = True
            continue
        operands.append(token)

    if not operands:
        return None

    resolved: list[Path] = []
    for operand in operands:
        resolved.extend(_expand(operand, cwd))

    if not resolved:
        return None
    return DeletePlan(command=text, paths=resolved, recursive=recursive)


def locate(path: Path) -> Path:
    """Absolute, normalised, and **not** through a final symlink.

    ``resolve()`` follows the last component too, so ``rm shortcut.txt`` planned
    a delete of whatever ``shortcut.txt`` pointed at. That is not what ``rm``
    does - it removes the link and leaves the target alone - so rerouting
    through the trash was quietly destroying a file the user had not named, and
    the preview confirmed the wrong one. Parents are still resolved, which is
    what keeps ``/tmp`` and ``/private/tmp`` from looking like two places.
    """
    if path.is_symlink():
        return path.parent.resolve() / path.name
    return path.resolve()


def _expand(operand: str, cwd: Path) -> list[Path]:
    """Glob-expand one operand against ``cwd``, skipping what does not exist."""
    if any(ch in operand for ch in "*?["):
        base = Path(operand)
        try:
            if base.is_absolute():
                matches = sorted(Path(base.anchor).glob(str(base.relative_to(base.anchor))))
            else:
                matches = sorted(cwd.glob(operand))
        except (ValueError, OSError):
            matches = []
        if matches:
            return [locate(m) for m in matches]
        # A glob that matched nothing may be a literal name that happens to
        # contain glob characters. `file[1].txt` is an ordinary filename, and
        # `[1]` is a character class that matches "1" rather than itself - so it
        # matched nothing, the delete was never rerouted through the trash, and
        # the preview fell back to echoing the command. Fall through and try the
        # name as written.

    path = Path(operand)
    if not path.is_absolute():
        path = cwd / path
    if not path.exists() and not path.is_symlink():
        # is_symlink as well as exists: a link with nothing at the far end is
        # still a thing `rm` removes, and exists() follows the link to answer.
        return []
    return [locate(path)]


def describe(plan: DeletePlan, limit: int = 5) -> str:
    """A human-readable preview of what a delete would remove."""
    names = [p.name + ("/" if p.is_dir() else "") for p in plan.paths[:limit]]
    listed = ", ".join(names)
    if plan.count > limit:
        listed += f", and {plan.count - limit} more"
    total = sum(_size_of(p) for p in plan.paths)
    return f"{plan.count} item{'s' if plan.count != 1 else ''} ({_human(total)}): {listed}"


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def trash_for(data_dir: Path, session: str = "") -> Trash:
    return Trash(data_dir / "trash", session)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_RETENTION_DAYS",
    "DeletePlan",
    "Trash",
    "TrashedItem",
    "describe",
    "parse_delete",
    "trash_for",
]
