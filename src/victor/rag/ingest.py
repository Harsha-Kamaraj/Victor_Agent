"""Getting things into memory: files on demand, fixes automatically.

Two sources, and the second is the one that matters.

``victor index <path>`` is the ordinary one - chunk some files, embed them,
store them. Useful, unremarkable.

:class:`ErrorFixWatcher` is the plan's "auto-capture hook", and it is what makes
this a memory rather than a search box. Nobody curates a knowledge base of their
own mistakes; if remembering a fix requires deciding to remember it, the store
stays empty. So the agent watches its own traffic and works out what counts as
a fix.

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

**Why this is not only about the shell.** The watcher started out reading shell
traffic alone, because a command line is a thing with an obvious identity: two
runs of ``pytest`` are two attempts at the same thing. But the failures this
agent actually hits on a desktop have the same shape - a click lands on the
wrong window, ``open_app`` brings the right one forward, the click then works -
and none of that was being remembered. :func:`describe_call` supplies the
missing piece, an identity and an is-this-an-intervention answer for any tool
call, so the same proven signal covers ``git`` and the desktop too.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
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


#: The argument that says what a call was aimed at, per tool, best first.
#:
#: Identity is the tool plus its target. Two clicks on different buttons are two
#: different attempts, and a success on one must not be taken as proof that the
#: other was fixed - which is exactly what would happen if every click shared
#: the identity "click".
TARGET_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "screen_read": (),  # no target: reading the screen is one thing
    "click": ("label", "index"),
    "type_text": ("label", "into"),
    "press_keys": ("keys",),
    "scroll": ("direction",),
    "find_on_screen": ("description",),
    "open_app": ("name",),
    "read_file": ("path",),
}

MAX_TARGET_CHARS = 60
"""Labels can be a whole paragraph of button text. Identity only needs enough
of one to tell two controls apart."""


@dataclass(frozen=True, slots=True)
class Action:
    """One tool call, in the three terms memory needs.

    ``head`` is identity: two calls sharing a head are two attempts at the same
    thing, so a success closes an earlier failure. ``line`` is how the call
    reads back to a person and to the model. ``changes`` says whether the call
    acted on the world, which is what separates a fix from looking around.
    """

    head: str
    line: str
    changes: bool
    shell: bool = False
    """Whether this really was a command line. Only those get a ``$`` when they
    are written back out; a click never was one, and dressing it up as one
    would invite the model to try running it."""

    @property
    def display(self) -> str:
        return f"$ {self.line}" if self.shell else self.line


def _short(value: Any, limit: int = MAX_TARGET_CHARS) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_call(
    tool: str, arguments: Mapping[str, Any], *, mutating: bool | None = None
) -> Action | None:
    """Turn a tool call into the terms memory works in, or ``None`` to ignore it.

    ``mutating`` is the tool's own declaration from its :class:`ToolSpec`, and
    it is the default answer to "did this change anything". Two tools override
    it, because the spec flag is a statement about the tool's *capability* while
    memory needs the truth about *this call*: ``shell`` is declared mutating
    because any command might write, and ``git`` because some subcommands do.
    Both know how to answer per call, so both are asked.
    """
    if tool == "shell":
        command = str(arguments.get("command", "")).strip()
        head = command_head(command)
        if not head:
            return None
        return Action(head, command, changes=not is_diagnostic(command), shell=True)

    if tool == "git":
        from ..tools.git import MUTATING

        sub = str(arguments.get("subcommand", "")).strip()
        if not sub:
            return None
        args = [str(a) for a in (arguments.get("args") or [])]
        line = " ".join(["git", sub, *args])
        return Action(f"git {sub}", _short(line, 120), changes=sub in MUTATING, shell=True)

    if tool in TARGET_ARGUMENTS:
        for name in TARGET_ARGUMENTS[tool]:
            value = arguments.get(name)
            if value not in (None, ""):
                target = f"{tool} {_short(value)}"
                return Action(target, target, changes=bool(mutating))
        return Action(tool, tool, changes=bool(mutating))

    # A tool nobody taught this function about. Identity falls back to the whole
    # argument list, which splits too finely rather than too coarsely: an
    # over-split identity remembers nothing, an over-merged one remembers the
    # wrong fix and recalls it with confidence.
    rendered = " ".join(f"{k}={_short(v, 30)}" for k, v in sorted(arguments.items()))
    line = f"{tool} {rendered}".strip()
    return Action(line, line, changes=bool(mutating))


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
    """An action that failed, and what has been tried since."""

    action: Action
    error: str
    interventions: list[str] = field(default_factory=list)


class ErrorFixWatcher:
    """Watches the agent's actions and stores ``(error -> fix)`` when one is proven.

    Feed it every tool result. It stays silent until the same action that failed
    later succeeds, and then it knows both halves of the pair.
    """

    def __init__(self, memory: Any, *, max_interventions: int = 8) -> None:
        self.memory = memory
        self.max_interventions = max_interventions
        self._failed: dict[str, Attempt] = {}
        self.captured = 0

    def observe(self, action: str | Action, *, ok: bool, output: str = "") -> str | None:
        """Record one result. Returns a note when a fix was captured.

        A plain string is read as a shell command, which is what the shell path
        and most of the tests pass.
        """
        act = action if isinstance(action, Action) else describe_call("shell", {"command": action})
        if act is None or not act.head:
            return None

        if not ok:
            error = summarise_error(output)
            if error:
                # A repeat failure keeps the interventions tried so far: the
                # user is still working on this one.
                existing = self._failed.get(act.head)
                if existing is None:
                    self._failed[act.head] = Attempt(action=act, error=error)
                else:
                    existing.error = error
            return None

        attempt = self._failed.pop(act.head, None)
        if attempt is None:
            # A success for something that never failed. It might still be the
            # fix for something else that is outstanding, so record it there.
            self._note_intervention(act)
            return None

        if not attempt.interventions:
            # It failed, then succeeded, and nothing happened in between -
            # flaky, or a transient the user waited out. There is no fix to
            # remember, so remember nothing.
            return None

        return self._store(attempt)

    def _note_intervention(self, action: Action) -> None:
        if not action.changes:
            return  # looking around is not fixing
        for attempt in self._failed.values():
            if len(attempt.interventions) < self.max_interventions:
                attempt.interventions.append(action.line)

    def _store(self, attempt: Attempt) -> str | None:
        fix = "\n".join(attempt.interventions)
        text = f"{attempt.action.display}\n{attempt.error}"
        record = self.memory.remember_fix(
            error=text,
            fix=fix,
            command=attempt.action.line,
        )
        if record is None:
            return None
        self.captured += 1
        return f"remembered how {attempt.action.head} was fixed"

    @property
    def outstanding(self) -> list[str]:
        """Actions that have failed and not yet been seen to succeed."""
        return sorted(self._failed)
