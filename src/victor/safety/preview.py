"""What a command would do, worked out without doing it.

The plan: *"safety/dryrun.py — renders what would happen (files matched, diff
preview) and speaks a summary before executing."*

Echoing the command back is not a preview. `rm *.log` tells the user nothing
they did not already type; **"7 files (1.2 MB): build.log, test.log, and 5
more"** tells them whether the glob matched what they meant. That difference is
the entire value of confirming: a user can only approve what they can predict,
and a glob is precisely the thing people mispredict.

Everything here is read-only. It stats and globs; it never opens a file for
writing or runs the command it is describing.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from .trash import describe as describe_delete
from .trash import parse_delete

MAX_LISTED = 6


def preview(command: str, cwd: Path | None = None) -> str:
    """One line describing the effect of ``command``, or the command itself."""
    workdir = Path(cwd or Path.cwd())
    text = " ".join(command.split())

    for describe in (_preview_delete, _preview_redirect, _preview_mkdir, _preview_move):
        rendered = describe(text, workdir)
        if rendered:
            return rendered
    return f"would run: {text}"


def _preview_delete(command: str, cwd: Path) -> str | None:
    plan = parse_delete(command, cwd)
    if plan is None:
        return None
    return f"would delete {describe_delete(plan, MAX_LISTED)}"


def _find_redirect(tokens: list[str]) -> tuple[str, str] | None:
    """``(operator, target)`` for a redirect among ``tokens``, else ``None``.

    Scans tokens rather than the raw string, which is what makes a quoted
    ``>`` harmless. ``echo "a > b" > out.txt`` used to be previewed as "would
    create b" - the regex found the first ``>`` it could see and never reached
    the real target. shlex has already decided what is a word and what is an
    operator, so a ``>`` inside quotes arrives as part of a larger token and
    cannot be mistaken for one.
    """
    for position, token in enumerate(tokens):
        if token in (">", ">>"):
            following = tokens[position + 1 :]
            return (token, following[0]) if following else None
        # `>out.txt` with no space is one token; `>&2` is not a file at all.
        if token.startswith(">") and not token.startswith(">&"):
            operator = ">>" if token.startswith(">>") else ">"
            rest = token[len(operator) :]
            if rest:
                return operator, rest
    return None


def _preview_redirect(command: str, cwd: Path) -> str | None:
    """Report whether a redirect creates a file or overwrites an existing one."""
    found = _find_redirect(_tokens(command))
    if found is None:
        return None

    operator, raw = found
    target = Path(raw)
    if not target.is_absolute():
        target = cwd / target

    if not target.exists():
        return f"would create {target.name}"
    size = target.stat().st_size
    if operator == ">>":
        return f"would append to {target.name} ({size:,} bytes now)"
    return f"would overwrite {target.name}, discarding its {size:,} bytes"


def _preview_mkdir(command: str, cwd: Path) -> str | None:
    tokens = _tokens(command)
    if not tokens or tokens[0] != "mkdir":
        return None
    targets = [t for t in tokens[1:] if not t.startswith("-")]
    existing = [t for t in targets if (cwd / t).exists()]
    if existing:
        return f"would create {', '.join(targets)} ({', '.join(existing)} already exists)"
    return f"would create {', '.join(targets)}"


def _preview_move(command: str, cwd: Path) -> str | None:
    """A move that lands on an existing path destroys it - say so.

    Two things this got wrong, both understating the change, which is the
    direction that matters: it reported only the first of several sources, so
    ``mv a b c dir/`` was previewed as moving one file; and it said nothing when
    the destination was a directory already holding a file of that name, which
    is the commonest way a move quietly destroys something.
    """
    tokens = _tokens(command)
    if not tokens or tokens[0] not in {"mv", "move", "cp"}:
        return None
    operands = [t for t in tokens[1:] if not t.startswith("-")]
    if len(operands) < 2:
        return None

    *sources, destination = operands
    verb = "copy" if tokens[0] == "cp" else "move"
    target = Path(destination) if Path(destination).is_absolute() else cwd / destination

    if target.is_file():
        size = target.stat().st_size
        moved = sources[0] if len(sources) == 1 else f"{len(sources)} files"
        return f"would {verb} {moved} over {destination}, discarding its {size:,} bytes"

    if target.is_dir():
        # Landing in a directory that already holds a file of the same name
        # overwrites it, and nothing in the command says so.
        clobbered = [s for s in sources if (target / Path(s).name).is_file()]
        if clobbered:
            listed = ", ".join(clobbered[:MAX_LISTED])
            hidden = len(clobbered) - MAX_LISTED
            more = f", and {hidden} more" if hidden > 0 else ""
            return (
                f"would {verb} {_describe_sources(sources)} into {destination}, "
                f"replacing what is already there: {listed}{more}"
            )

    return f"would {verb} {_describe_sources(sources)} to {destination}"


def _describe_sources(sources: list[str]) -> str:
    """Name every source, because naming one of three is a false preview."""
    if len(sources) == 1:
        return sources[0]
    listed = ", ".join(sources[:MAX_LISTED])
    more = f", and {len(sources) - MAX_LISTED} more" if len(sources) > MAX_LISTED else ""
    return f"{len(sources)} files ({listed}{more})"


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def spoken(summary: str) -> str:
    """The same preview, phrased for text-to-speech.

    Paths and byte counts read badly aloud, so the spoken form keeps the shape
    of the change and drops the detail the screen already shows.
    """
    collapsed = summary.replace("would ", "This would ")
    return collapsed if len(collapsed) < 140 else collapsed[:137] + "..."
