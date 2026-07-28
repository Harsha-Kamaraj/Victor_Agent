"""Git, as one tool with an explicit subcommand allowlist.

Exposing ``git`` through the general shell tool would work, but it would also
mean every git operation - reading a log and rewriting history alike - arrives
at the safety layer as an opaque string. Naming the subcommands here lets each
one declare whether it mutates, which is exactly what P3's interceptor needs
and what makes "read the diff" free while "reset --hard" is gated.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import ToolResult, ToolSpec

READ_ONLY = {
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "remote",
    "rev-parse",
    "describe",
    "blame",
    "shortlog",
    "ls-files",
    "config",
}

MUTATING = {
    "add",
    "commit",
    "checkout",
    "switch",
    "restore",
    "stash",
    "merge",
    "rebase",
    "reset",
    "revert",
    "cherry-pick",
    "tag",
    "fetch",
    "pull",
    "push",
    "init",
    "clone",
}

ALLOWED = READ_ONLY | MUTATING

#: Branch names whose history is shared. Rewriting these hurts other people.
DEFAULT_BRANCHES = {"main", "master", "develop", "trunk", "release"}

#: Force flags proper. `--force-with-lease` is deliberately absent: it refuses
#: to overwrite work it has not seen, which is the safe way to force-push.
FORCE_FLAGS = {"--force", "-f"}

DEFAULT_TIMEOUT = 30.0


class GitTool:
    """Run a git subcommand in a repository."""

    def __init__(
        self,
        *,
        cwd: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        allow_mutating: bool = True,
    ) -> None:
        self.cwd = Path(cwd or Path.cwd())
        self.timeout = timeout
        self.allow_mutating = allow_mutating
        self.spec = ToolSpec(
            name="git",
            description=(
                "Run a git subcommand in the current repository. "
                f"Read-only: {', '.join(sorted(READ_ONLY))}. "
                f"Mutating: {', '.join(sorted(MUTATING))}. "
                "Pass arguments as a list, without the leading 'git'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subcommand": {
                        "type": "string",
                        "description": "The git subcommand, e.g. 'status' or 'log'.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments, e.g. ['--oneline', '-5'].",
                    },
                    "cwd": {"type": "string", "description": "Repository path."},
                },
                "required": ["subcommand"],
            },
            # Declared mutating because the tool *can* mutate; the interceptor
            # gets the actual subcommand and decides per call.
            mutating=True,
        )

    def run(
        self, subcommand: str, args: list[str] | None = None, cwd: str | None = None
    ) -> ToolResult:
        if shutil.which("git") is None:
            return ToolResult(ok=False, error="git is not on PATH")

        sub = subcommand.strip()
        arguments = [str(a) for a in (args or [])]

        if sub not in ALLOWED:
            return ToolResult(
                ok=False,
                error=(
                    f"git subcommand {sub!r} is not allowed. "
                    f"Allowed: {', '.join(sorted(ALLOWED))}"
                ),
            )
        if sub in MUTATING and not self.allow_mutating:
            return ToolResult(
                ok=False,
                error=f"git {sub} would modify the repository, which is not permitted here",
            )

        refusal = _refuse_force_push(sub, arguments)
        if refusal is not None:
            return ToolResult(ok=False, error=f"refused: {refusal}. Ask the user to run it.")

        workdir = Path(cwd) if cwd else self.cwd
        if not workdir.is_dir():
            return ToolResult(ok=False, error=f"no such directory: {workdir}")

        try:
            completed = subprocess.run(
                ["git", sub, *arguments],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"git {sub} timed out after {self.timeout:.0f}s")
        except OSError as exc:
            return ToolResult(ok=False, error=f"could not run git: {exc}")

        return ToolResult(
            ok=completed.returncode == 0,
            output=completed.stdout,
            error=completed.stderr or None,
            metadata={
                "subcommand": sub,
                "args": arguments,
                "exit_code": completed.returncode,
                "mutating": sub in MUTATING,
            },
        )


def _refuse_force_push(subcommand: str, args: list[str]) -> str | None:
    """Refuse only the force pushes that destroy shared history.

    A force push to a feature branch is routine and goes through confirmation
    like any other mutation. A blanket refusal of every `--force` would make the
    tool unusable for normal rebase workflows, and users route around tools that
    refuse ordinary work.
    """
    if subcommand != "push" or not any(a in FORCE_FLAGS for a in args):
        return None

    refspecs = [a for a in args if not a.startswith("-")][1:]  # skip the remote
    if not refspecs:
        return "force push with no branch named - the current branch may be a shared one"
    if any(ref.split(":")[-1] in DEFAULT_BRANCHES for ref in refspecs):
        return "force push to a shared default branch"
    return None


def is_repository(path: Path | None = None) -> bool:
    """True if ``path`` sits inside a git working tree."""
    if shutil.which("git") is None:
        return False
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path or Path.cwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "true"
