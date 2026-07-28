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

import re
import shlex
from pathlib import Path

from .trash import describe as describe_delete
from .trash import parse_delete

MAX_LISTED = 6

_REDIRECT = re.compile(r"(?<![0-9<>])(>{1,2})(?!&)\s*(\S+)")


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


def _preview_redirect(command: str, cwd: Path) -> str | None:
    """Report whether a redirect creates a file or overwrites an existing one."""
    match = _REDIRECT.search(command)
    if match is None:
        return None

    operator, raw = match.groups()
    target = Path(raw.strip("\"'"))
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
    """A move that lands on an existing path destroys it - say so."""
    tokens = _tokens(command)
    if not tokens or tokens[0] not in {"mv", "move", "cp"}:
        return None
    operands = [t for t in tokens[1:] if not t.startswith("-")]
    if len(operands) < 2:
        return None

    source, destination = operands[0], operands[-1]
    target = cwd / destination if not Path(destination).is_absolute() else Path(destination)
    verb = "copy" if tokens[0] == "cp" else "move"
    if target.is_file():
        size = target.stat().st_size
        return f"would {verb} {source} over {destination}, discarding its {size:,} bytes"
    return f"would {verb} {source} to {destination}"


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
