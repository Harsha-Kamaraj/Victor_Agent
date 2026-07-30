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

import contextlib
import os
import platform
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Any

from .base import ToolResult, ToolSpec

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0
POLL_INTERVAL = 0.05
"""How often the wait loop checks the kill switch. Bounds abort latency."""

READ_SIZE = 8192
"""Chunk size for draining a child's pipes."""

REAP_GRACE = 2.0
"""Seconds a process gets to die politely before it is killed."""

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
#: Operating-system directories whose contents are not the user's to delete.
#: Narrower than "anything system-owned" on purpose: /etc and /usr/local are
#: routinely edited by developers and stay at CONFIRM.
_SYSTEM_PATH = (
    r"(C:\\+Windows|%SystemRoot%|\$env:SystemRoot|"
    r"C:\\+Program\s+Files|/System/|/Library/Extensions)"
)

#: Verbs that would damage such a directory rather than read it.
_DESTRUCTIVE_VERB = r"\b(rm|rmdir|del|erase|rd|unlink|mv|move|Remove-Item|takeown|icacls)\b"

#: Branch names a force-push to is almost never recoverable in a team.
_DEFAULT_BRANCH = r"\b(main|master|develop|trunk|release)\b"

CATASTROPHIC = (
    (
        re.compile(rf"\brm\s+(-[a-zA-Z-]+\s+)*{_DOOMED_TARGET}(\s|$)"),
        "recursive delete of a root, home or drive path",
    ),
    (re.compile(r"--no-preserve-root"), "explicitly disables the root guard"),
    # Windows equivalents of the same disaster.
    (
        re.compile(r"\b(del|rd|rmdir)\b.*(/s\b.*)?[A-Za-z]:\\+(\s|$|\*)", re.I),
        "recursive delete of a drive root",
    ),
    (
        re.compile(r"\bRemove-Item\b.*-Recurse.*\s[A-Za-z]:\\+(\s|$|\*)", re.I),
        "recursive delete of a drive root",
    ),
    (
        re.compile(rf"{_DESTRUCTIVE_VERB}[^|;]*{_SYSTEM_PATH}", re.I),
        "writes into an OS directory",
    ),
    (re.compile(r"\bdiskpart\b", re.I), "partition table editing"),
    (re.compile(r"\bvssadmin\b.*\bdelete\b.*\bshadows\b", re.I), "deletes volume shadow copies"),
    # `\b/set` never matches: the char before `/` is a space, so there is no
    # word boundary there.
    (re.compile(r"\bbcdedit\b.*(\bdelete\b|/set\b)", re.I), "boot configuration change"),
    (re.compile(r"\breg\s+delete\b.*\bHKLM\b", re.I), "deletes a system registry hive"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem format"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw write to a block device"),
    (re.compile(r">\s*/dev/[sh]d[a-z]"), "raw write to a block device"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.I), "drive format"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "power state change"),
    (re.compile(r":\(\)\s*\{.*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/"), "recursive permission wipe on /"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh"), "pipe-from-network to shell"),
    # Force-pushing a feature branch is routine; force-pushing a shared default
    # branch destroys other people's history. Only the latter is refused here.
    # `(\s|$)` after the flag matters: without it `--force-with-lease`, which is
    # the *safe* variant, matches `--force` and gets refused.
    # The "force push with no branch named" case needs to count positional
    # arguments, which a regex cannot do - it lives in classify._classify_git_shell.
    # Two lookaheads rather than a sequence: `git push origin main --force` and
    # `git push --force origin main` are the same command, and a pattern that
    # only catches one argument order catches neither reliably.
    (
        re.compile(
            rf"\bgit\s+push\b(?=[^|;]*(--force|-f)(\s|$))(?=[^|;]*{_DEFAULT_BRANCH})"
        ),
        "force push to a shared default branch",
    ),
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
                # Put the child in its own group so the whole tree can be
                # signalled. `make test` spawns pytest which spawns workers;
                # killing only the direct child orphans the rest, and the plan's
                # exit gate says "no orphaned processes".
                **_process_group_kwargs(),
            )
        except OSError as exc:
            return ToolResult(ok=False, error=f"could not run: {exc}")

        stdout, stderr, status = self._wait(process, limit)
        if status == "aborted":
            return ToolResult(
                ok=False,
                error=_and_what_it_said("stopped by the kill switch", stderr),
                output=stdout,
                metadata={"command": command, "aborted": True},
            )
        if status == "timeout":
            return ToolResult(
                ok=False,
                error=_and_what_it_said(f"timed out after {limit:.0f}s", stderr),
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

    def _wait(self, process: subprocess.Popen, limit: float) -> tuple[str, str, str]:
        """Wait for the process, checking the kill switch as we go.

        Without the poll loop, aborting a run would still have to wait out
        whatever the command felt like doing. Abort latency is bounded by the
        poll interval instead of by the command.

        One loop, not two. An earlier version called ``communicate()`` when no
        kill switch was wired and polled when one was - so every test took the
        first branch and every real run took the second, because the CLI always
        wires a switch. The deadlock the drains below exist to prevent therefore
        lived in the only code that ever shipped, behind a green suite. Whether
        a switch is present is now the sole difference between the two.
        """
        out = _Drain(process.stdout)
        err = _Drain(process.stderr)
        switch = self.kill_switch
        deadline = time.monotonic() + limit

        while True:
            if switch is not None and switch.tripped:
                _reap(process, out, err)
                return out.collect(), err.collect(), "aborted"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _reap(process, out, err)
                return out.collect(), err.collect(), "timeout"
            try:
                # Returns the moment the child exits, so the poll interval is a
                # ceiling on how late the switch is noticed, not a floor on how
                # long a fast command takes.
                process.wait(timeout=min(POLL_INTERVAL, remaining))
            except subprocess.TimeoutExpired:
                continue
            return out.collect(), err.collect(), "ok"


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


def _process_group_kwargs() -> dict[str, Any]:
    """Popen options that make the child the root of its own killable group."""
    if platform.system() == "Windows":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags} if flags else {}
    return {"start_new_session": True}


def _and_what_it_said(reason: str, stderr: str) -> str:
    """Keep what the command said about itself before it was cut off.

    Both of these paths reported only why *we* stopped and dropped the command's
    stderr. Most build tools - cmake, npm, gradle, javac - write their
    diagnostics there and nothing to stdout, so a run that printed
    ``FATAL: missing header foo.h`` and then hung reached the model as
    ``timed out after 30s`` with no output at all: the one line that explained
    the hang was the line thrown away. No limit is needed here because
    ``ToolResult.for_model`` truncates the reason and the output together.
    """
    said = stderr.strip()
    return f"{reason}\n{said}" if said else reason


class _Drain:
    """Empties one of a child's pipes on a thread of its own.

    The wait loop cannot call ``communicate()``: that blocks until the child
    exits, which is the one thing the loop exists to avoid. But a loop that only
    polls never empties the pipe, and a child that fills the buffer - 64 KiB on
    macOS - blocks on the write until somebody reads. The command then looks
    like it hung: measured here, ``python -c "sys.stdout.write('y'*300000)"``
    sat until the 6-second timeout and came back with exactly 65,536 bytes and
    a failure, having actually finished its work in 54 ms. Any build or test run
    worth watching clears 64 KiB, so this was not an edge case.

    So the reading has to happen while the loop is still looping.
    """

    def __init__(self, stream: IO[str] | None) -> None:
        self._chunks: list[str] = []
        self._stream = stream
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        if self._stream is None:
            return
        try:
            while True:
                chunk = self._stream.read(READ_SIZE)
                if not chunk:
                    break
                self._chunks.append(chunk)
        except (OSError, ValueError):
            # The pipe was closed under us. Whatever arrived first still counts.
            pass
        finally:
            # Nobody calls communicate() any more, so closing is ours to do.
            # An inherited handle left open is what stops Windows deleting a
            # file it is holding.
            with contextlib.suppress(Exception):
                self._stream.close()

    def settle(self, timeout: float) -> bool:
        """Wait for end-of-file. False if something is still holding the pipe."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def collect(self, timeout: float = REAP_GRACE) -> str:
        self.settle(timeout)
        return "".join(self._chunks)


def _reap(process: subprocess.Popen, *pipes: _Drain, grace: float = REAP_GRACE) -> None:
    """Terminate a process and everything it spawned.

    Signals the whole group so a shell that launched children does not leave
    them running. Escalates to SIGKILL if the group ignores the polite signal,
    because a kill switch that can be declined is not a kill switch.

    An unclosed pipe is the evidence for "ignored it". The shell itself may die
    on SIGTERM while a grandchild that inherited its stdout keeps running and
    keeps the write end open; waiting on the process alone would call that a
    clean exit and walk away from the grandchild. The old code got this right by
    accident, because ``communicate()`` waits on the pipes rather than the
    process - dropping it for a plain ``wait()`` would have silently lost it.
    """
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                timeout=grace,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        process.terminate()

    if _settled(process, pipes, grace):
        return

    try:
        if platform.system() != "Windows":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except OSError:
        process.kill()

    _settled(process, pipes, grace)


def _settled(process: subprocess.Popen, pipes: tuple[_Drain, ...], grace: float) -> bool:
    """True once the process has exited and nothing is holding its pipes."""
    deadline = time.monotonic() + grace
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        return False
    return all(pipe.settle(max(0.0, deadline - time.monotonic())) for pipe in pipes)


def describe_environment(cwd: Path | None = None) -> dict[str, Any]:
    """Facts about the machine, for the system prompt."""
    workdir = Path(cwd or Path.cwd())
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "shell": "powershell" if platform.system() == "Windows" else os.environ.get("SHELL", "sh"),
        "cwd": str(workdir),
    }
