"""Getting things into memory: files on demand, fixes automatically.

Two sources, and the second is the one that matters.

``victor index <path>`` is the ordinary one - chunk some files, embed them,
store them. Useful, unremarkable.

:class:`ErrorFixWatcher` is the plan's "auto-capture hook", and it is what makes
this a memory rather than a search box. Nobody curates a knowledge base of their
own mistakes; if remembering a fix requires deciding to remember it, the store
stays empty. So the agent watches its own shell traffic and works out what
counts as a fix.

**What counts as a fix, and why not the obvious thing.** The plan says: when a
command exits non-zero and a later one succeeds, store the pair. Taken
literally that is too weak - ``pytest`` fails, you run ``ls`` to look around,
and ``ls`` succeeds. Storing "the fix for this traceback is ls" is worse than
storing nothing, because it will be recalled with confidence the next time.

The signal used instead is stronger and needs no judgement: **the same command
failed, and then later succeeded.** Whatever ran in between is the fix. That is
the shape of every real debugging session, it is verifiable rather than
inferred, and when it never happens - the command is still broken - nothing is
stored, which is correct.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHUNK_CHARS = 1_200
"""Roughly 300 tokens. Big enough to hold a function or a traceback, small
enough that recall returns the relevant part rather than a whole file."""

CHUNK_OVERLAP = 150
"""So a fact sitting on a chunk boundary is not cut in half."""

MAX_FILE_BYTES = 512_000
"""Anything larger is a build artefact, a lockfile or a dataset."""

MAX_ERROR_CHARS = 2_000
"""Tracebacks can be enormous; the useful part is the head and the exception."""

#: Extensions worth reading. An allowlist rather than a denylist: the failure
#: mode of guessing wrong is embedding a megabyte of minified JavaScript.
TEXT_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".kt",
        ".sh", ".bash", ".zsh", ".ps1", ".sql", ".r",
        ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini",
        ".cfg", ".env", ".conf", ".html", ".css", ".scss",
    }
)

#: Directories never worth indexing, whatever is in them.
SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
        "env", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", "target", ".next", ".cache", "site-packages",
        ".idea", ".vscode", "coverage", "htmlcov", ".victor",
    }
)



def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on line boundaries, with overlap.

    Line boundaries rather than a fixed character count because the things
    being indexed are code and tracebacks, where a line is the unit of meaning
    and cutting one in half produces a chunk that matches nothing.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for line in lines:
        if length + len(line) > size and current:
            chunks.append("".join(current))
            # Carry the tail of this chunk into the next one.
            carried: list[str] = []
            carried_length = 0
            for previous in reversed(current):
                if carried_length + len(previous) > overlap:
                    break
                carried.insert(0, previous)
                carried_length += len(previous)
            current, length = carried, carried_length
        current.append(line)
        length += len(line)

    if current:
        chunks.append("".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def iter_files(root: Path, *, max_bytes: int = MAX_FILE_BYTES) -> Iterator[Path]:
    """Text files under ``root`` worth indexing."""
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() in TEXT_SUFFIXES:
            yield root
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield path


def read_text(path: Path) -> str:
    """File contents, or empty if it turns out not to be text after all."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def index_path(memory: Any, root: Path, *, on_file: Any = None) -> tuple[int, int]:
    """Chunk and store every indexable file under ``root``.

    Returns ``(files read, chunks stored)`` - the two differ because chunks
    already known are skipped, so re-indexing an unchanged project stores
    nothing and says so.
    """
    files = 0
    batch: list[tuple[str, str, str, dict[str, Any]]] = []
    for path in iter_files(Path(root)):
        text = read_text(path)
        if not text.strip():
            continue
        files += 1
        if on_file is not None:
            on_file(path)
        for number, chunk in enumerate(chunk_text(text)):
            batch.append((chunk, "file", str(path), {"chunk": number}))
    return files, memory.add_batch(batch)


# --- the auto-capture hook -------------------------------------------------


def command_head(command: str) -> str:
    """The part of a command line that identifies what was being run.

    ``pytest -x tests/`` and ``pytest tests/`` are the same attempt at the same
    thing, so a fix found for one applies to the other. Flags and paths vary
    between attempts in a way the name does not.
    """
    tokens = command.strip().split()
    if not tokens:
        return ""
    name = tokens[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    # Keep the subcommand: `git push` and `git status` are not the same thing.
    subcommanded = {"git", "npm", "yarn", "pnpm", "cargo", "go", "docker", "pip", "poetry"}
    if name in subcommanded and len(tokens) > 1 and not tokens[1].startswith("-"):
        return f"{name} {tokens[1]}"
    return name


def is_diagnostic(command: str) -> bool:
    """Was this command looking, rather than changing?

    Delegated to the P3 classifier rather than answered again here. That is the
    same question the safety layer asks before deciding whether to confirm
    something, and two definitions of "this only reads" drift apart: the first
    version of this function kept its own list of command names, which called
    ``printf 'x' > helper.py`` diagnostic because it starts with ``printf``.
    It writes a file. The classifier already knew that - it has a rule for
    redirects - and reusing it means a fix that writes a file is recognised as
    a fix, and any future improvement to one benefits the other.
    """
    if not command.strip():
        return True
    from ..safety.classify import Risk, classify_shell

    try:
        return classify_shell(command).risk is Risk.SAFE
    except Exception:
        # A command the classifier cannot parse is not obviously a read, and
        # treating it as one would silently drop it from the interventions.
        return False


_NOISE = re.compile(r"^\s*(File \"|\s+at |\s{4,})")


def summarise_error(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    """The part of a failure worth embedding.

    A traceback's frames are mostly paths that differ between machines and runs;
    the exception line at the end is the part that identifies the problem, and
    the first few lines usually say what was being attempted. Keeping both ends
    and dropping the middle is the same trade the tool-output truncator makes.
    """
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if len("\n".join(lines)) <= limit:
        return "\n".join(lines)

    signal = [line for line in lines if not _NOISE.match(line)]
    kept = (signal or lines)[:6] + ["..."] + (signal or lines)[-6:]
    joined = "\n".join(kept)
    return joined if len(joined) <= limit else joined[:limit]


@dataclass
class Attempt:
    """A command that failed, and what has been tried since."""

    command: str
    error: str
    interventions: list[str] = field(default_factory=list)


class ErrorFixWatcher:
    """Watches shell traffic and stores ``(error -> fix)`` when one is proven.

    Feed it every shell result. It stays silent until the same command that
    failed later succeeds, and then it knows both halves of the pair.
    """

    def __init__(self, memory: Any, *, max_interventions: int = 8) -> None:
        self.memory = memory
        self.max_interventions = max_interventions
        self._failed: dict[str, Attempt] = {}
        self.captured = 0

    def observe(self, command: str, *, ok: bool, output: str = "") -> str | None:
        """Record one shell result. Returns a note when a fix was captured."""
        head = command_head(command)
        if not head:
            return None

        if not ok:
            error = summarise_error(output)
            if error:
                # A repeat failure keeps the interventions tried so far: the
                # user is still working on this one.
                existing = self._failed.get(head)
                if existing is None:
                    self._failed[head] = Attempt(command=command, error=error)
                else:
                    existing.error = error
            return None

        attempt = self._failed.pop(head, None)
        if attempt is None:
            # A success for something that never failed. It might still be the
            # fix for something else that is outstanding, so record it there.
            self._note_intervention(command)
            return None

        if not attempt.interventions:
            # It failed, then succeeded, and nothing happened in between -
            # flaky, or a transient the user waited out. There is no fix to
            # remember, so remember nothing.
            return None

        return self._store(attempt)

    def _note_intervention(self, command: str) -> None:
        if is_diagnostic(command):
            return  # looking around is not fixing
        for attempt in self._failed.values():
            if len(attempt.interventions) < self.max_interventions:
                attempt.interventions.append(command.strip())

    def _store(self, attempt: Attempt) -> str | None:
        fix = "\n".join(attempt.interventions)
        text = f"$ {attempt.command}\n{attempt.error}"
        record = self.memory.remember_fix(
            error=text,
            fix=fix,
            command=attempt.command,
        )
        if record is None:
            return None
        self.captured += 1
        return f"remembered how {command_head(attempt.command)} was fixed"

    @property
    def outstanding(self) -> list[str]:
        """Commands that have failed and not yet been seen to succeed."""
        return sorted(self._failed)
