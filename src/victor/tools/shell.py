"""Run shell commands.

This is the tool that makes Victor useful and the tool that makes it
dangerous, and the argument string is written by a language model.

P3 is what makes this genuinely safe: classification, spoken confirmation, a
dry-run preview and an undo journal. Until then this module carries its own
last-ditch denylist covering the handful of commands that are catastrophic and
irreversible. That list is **not** a security boundary - it is a guard against
an obvious model mistake, and it is deliberately short so nobody mistakes it
for the real thing.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import ToolResult, ToolSpec

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0
POLL_INTERVAL = 0.05
"""How often the wait loop checks the kill switch. Bounds abort latency."""

#: Targets whose recursive deletion cannot be recovered from and is never a
#: legitimate instruction: the filesystem root, the whole home directory, a
#: drive letter, or a bare glob of any of them.
#: The optional trailing slash matters: `rm -rf ~` and `rm -rf ~/` are the same
#: disaster, and a pattern that catches only one of them catches neither in
#: practice. The alternatives are anchored so `~/projects/old` does not match -
#: that is a specific path the user can read in a confirmation prompt.
#: A bare `*` is included: `rm -rf *` empties the working directory, which is
#: the same disaster as naming it. `rm -rf src/*` is not - that is a specific
#: path the user can read.
_DOOMED_TARGET = (
    r"(/|~/?|\$HOME/?|\$\{HOME\}/?|[A-Za-z]:\\?|/\*|~/\*|\$HOME/\*|\*|\.\.?/?\*?)"
)

#: Refused outright. Reserved for damage with no recovery path at all.
#:
#: Note what is deliberately *not* here: `rm -rf build`, `rm -rf node_modules`
#: and similar. Those are routine, and the protection the design promises for
#: them is confirmation plus the journal, not a permanent block. Widening this
#: list to every recursive delete would make the agent unable to do ordinary
#: cleanup while adding no safety - it would just push users to --yes.
CATASTROPHIC = (
    (
        re.compile(rf"\brm\s+(-[a-zA-Z-]+\s+)*{_DOOMED_TARGET}(\s|$)"),
        "recursive delete of a root, home or drive path",
    ),
    (re.compile(r"--no-preserve-root"), "explicitly disables the root guard"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem format"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw write to a block device"),
    (re.compile(r">\s*/dev/[sh]d[a-z]"), "raw write to a block device"),
    (re.compile(r"\bformat\s+[a-zA-Z]:"), "drive format"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "power state change"),
    (re.compile(r":\(\)\s*\{.*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/"), "recursive permission wipe on /"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh"), "pipe-from-network to shell"),
    (re.compile(r"\bgit\s+push\b.*--force"), "force push"),
)


def screen(command: str) -> str | None:
    """Return why a command is refused, or ``None`` if it passes."""
    collapsed = " ".join(command.split())
    for pattern, reason in CATASTROPHIC:
        if pattern.search(collapsed):
            return reason
    return None


class ShellTool:
    """Execute a command and return its output."""

    def __init__(
        self,
        *,
        cwd: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
        enabled: bool = True,
        kill_switch: Any | None = None,
        trash: Any | None = None,
    ) -> None:
        self.cwd = Path(cwd or Path.cwd())
        self.timeout = timeout
        self.env = env
        self.enabled = enabled
        self.kill_switch = kill_switch
        self.trash = trash
        self.spec = ToolSpec(
            name="shell",
            description=(
                "Run a shell command and return its stdout and stderr. "
                "Use for inspecting the system, running builds and tests, and "
                "reading files. Prefer a single specific command over a chain. "
                "Output is truncated, so avoid commands that print thousands of lines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command line to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory. Defaults to the session directory.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            f"Seconds before the command is killed (max {MAX_TIMEOUT:.0f})."
                        ),
                    },
                },
                "required": ["command"],
            },
            # Any shell command may write, so it is mutating by declaration.
            mutating=True,
        )

    @staticmethod
    def _shell_for_platform() -> tuple[str, list[str]] | None:
        """The interpreter to hand the command to, if not using ``shell=True``."""
        if platform.system() == "Windows":
            return ("powershell", ["-NoProfile", "-NonInteractive", "-Command"])
        return None

    def run(self, command: str, cwd: str | None = None, timeout: float | None = None) -> ToolResult:
        if not self.enabled:
            return ToolResult(ok=False, error="the shell tool is disabled for this session")

        refusal = screen(command)
        if refusal is not None:
            return ToolResult(
                ok=False,
                error=(
                    f"refused: {refusal}. This command is on the irreversible-damage "
                    "list and will not run. Ask the user to run it themselves."
                ),
                metadata={"refused": refusal},
            )

        workdir = Path(cwd) if cwd else self.cwd
        if not workdir.is_dir():
            return ToolResult(ok=False, error=f"no such directory: {workdir}")

        trashed = self._delete_via_trash(command, workdir)
        if trashed is not None:
            return trashed

        limit = min(float(timeout or self.timeout), MAX_TIMEOUT)
        env = {**os.environ, **(self.env or {})}
        invocation = self._shell_for_platform()

        if invocation is None:
            argv: str | list[str] = command
            use_shell = True
        else:
            executable, flags = invocation
            argv = [executable, *flags, command]
            use_shell = False

        try:
            process = subprocess.Popen(
                argv,
                shell=use_shell,
                cwd=workdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            return ToolResult(ok=False, error=f"could not run: {exc}")

        stdout, stderr, status = self._wait(process, limit, command)
        if status == "aborted":
            return ToolResult(
                ok=False,
                error="stopped by the kill switch",
                output=stdout,
                metadata={"command": command, "aborted": True},
            )
        if status == "timeout":
            return ToolResult(
                ok=False,
                error=f"timed out after {limit:.0f}s",
                output=stdout,
                metadata={"command": command, "timeout": limit},
            )

        return ToolResult(
            ok=process.returncode == 0,
            output=stdout,
            error=stderr or None,
            metadata={
                "command": command,
                "exit_code": process.returncode,
                "cwd": str(workdir),
            },
        )

    def _delete_via_trash(self, command: str, workdir: Path) -> ToolResult | None:
        """Reroute a plain delete into the trash so it can be undone.

        Returns ``None`` when the command is not a delete, or is too complex to
        rewrite safely - those run exactly as written.
        """
        if self.trash is None:
            return None

        from ..safety.trash import describe, parse_delete

        plan = parse_delete(command, workdir)
        if plan is None:
            return None

        # Measure before moving: once the paths are in the trash there is
        # nothing left at the original locations to size.
        summary = describe(plan)

        try:
            items = self.trash.store_all(plan.paths)
        except OSError as exc:
            return ToolResult(
                ok=False,
                error=f"could not move to trash: {exc}",
                metadata={"command": command},
            )

        self.trash.prune()
        return ToolResult(
            ok=True,
            output=(
                f"Moved {summary} to the trash.\n"
                "Nothing was permanently deleted - `victor undo` restores it."
            ),
            metadata={
                "command": command,
                "exit_code": 0,
                "cwd": str(workdir),
                "trashed": [item.to_json() for item in items],
            },
        )

    def _wait(
        self, process: subprocess.Popen, limit: float, command: str
    ) -> tuple[str, str, str]:
        """Wait for the process, checking the kill switch as we go.

        Without the poll loop, aborting a run would still have to wait out
        whatever the command felt like doing. Abort latency is bounded by the
        poll interval instead of by the command.
        """
        if self.kill_switch is None:
            try:
                stdout, stderr = process.communicate(timeout=limit)
                return stdout, stderr, "ok"
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return stdout, stderr, "timeout"

        deadline = time.monotonic() + limit
        while process.poll() is None:
            if self.kill_switch.tripped:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                return stdout, stderr, "aborted"
            if time.monotonic() > deadline:
                process.kill()
                stdout, stderr = process.communicate()
                return stdout, stderr, "timeout"
            time.sleep(POLL_INTERVAL)

        stdout, stderr = process.communicate()
        return stdout, stderr, "ok"


class ReadFileTool:
    """Read a text file without spending a shell call on it."""

    def __init__(self, *, cwd: Path | None = None, max_bytes: int = 200_000) -> None:
        self.cwd = Path(cwd or Path.cwd())
        self.max_bytes = max_bytes
        self.spec = ToolSpec(
            name="read_file",
            description=(
                "Read a UTF-8 text file. Returns the contents with line numbers. "
                "Use instead of `cat` so output is framed consistently."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, absolute or relative."},
                    "start_line": {"type": "integer", "description": "First line, 1-indexed."},
                    "end_line": {"type": "integer", "description": "Last line, inclusive."},
                },
                "required": ["path"],
            },
            mutating=False,
        )

    def run(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> ToolResult:
        target = Path(path)
        if not target.is_absolute():
            target = self.cwd / target
        if not target.is_file():
            return ToolResult(ok=False, error=f"no such file: {target}")
        if target.stat().st_size > self.max_bytes:
            return ToolResult(
                ok=False,
                error=f"file is {target.stat().st_size:,} bytes, over the {self.max_bytes:,} limit",
            )

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return ToolResult(ok=False, error=f"could not read: {exc}")

        first = max(1, start_line or 1)
        last = min(len(lines), end_line or len(lines))
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(first, last + 1))
        return ToolResult(
            ok=True,
            output=body,
            metadata={"path": str(target), "lines": len(lines)},
        )


def describe_environment(cwd: Path | None = None) -> dict[str, Any]:
    """Facts about the machine, for the system prompt."""
    workdir = Path(cwd or Path.cwd())
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "shell": "powershell" if platform.system() == "Windows" else os.environ.get("SHELL", "sh"),
        "cwd": str(workdir),
    }
