"""Desktop control, exposed to the agent.

Six tools, all sharing one :class:`~victor.desktop.actions.Desktop` so they see
the same window and the same cached tree. Each one is thin on purpose: the
interesting behaviour - index verification, kill-switch checks, modifier
release - lives in the façade, where it is tested once and cannot be skipped by
adding a seventh tool.

**The terminal hole, and why it is closed here.** P3 classifies shell commands
before they run: ``rm -rf /`` is refused, ``git push --force`` is questioned.
None of that applies to a keystroke. An agent that can type into a Terminal
window has a shell that no classifier ever sees - and it would be a shell with
the *same* privileges, reached by a path that produces no journal entry and no
confirmation prompt. So typing into a terminal emulator is refused outright,
with a pointer to the tool that does get classified. This is not a
confirmation: there is a correct way to run a command, and it is the one the
safety layer can read.
"""

from __future__ import annotations

import re
from typing import Any

from ..desktop.actions import ActionRefused, ActuationUnavailable, Desktop
from ..desktop.keys import UnknownKey
from ..desktop.uia import PerceptionUnavailable
from .base import ToolResult, ToolSpec

#: Applications whose window is a command line. Typing into one of these routes
#: around the entire safety layer, so it is refused rather than confirmed.
TERMINAL_APPS = frozenset(
    {
        "terminal",
        "iterm",
        "iterm2",
        "alacritty",
        "kitty",
        "wezterm",
        "hyper",
        "warp",
        "ghostty",
        "cmd",
        "cmd.exe",
        "command prompt",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "windows powershell",
        "windows terminal",
        "conhost",
        "conhost.exe",
        "mintty",
        "git bash",
        "console",
    }
)

#: An application name Victor will pass to the operating system. Anything with a
#: path separator, a quote or a shell metacharacter is refused: ``open_app``
#: names a program, it does not compose a command line.
_APP_NAME = re.compile(r"^[\w .+&'-]{1,64}$")


def looks_like_terminal(window_title: str, process: str) -> bool:
    """Is the agent about to type into a shell?

    Checked against both the process and the window title, because each one
    misses cases the other catches: macOS reports the process (``Terminal``)
    while the title is the running command, and Windows Terminal reports a
    title that names the shell inside it.
    """
    haystacks = (process or "", window_title or "")
    for raw in haystacks:
        cleaned = raw.strip().lower().removesuffix(".exe")
        if cleaned in TERMINAL_APPS:
            return True
        # Titles are compound - "harshak - zsh - 80x24". Any word being a known
        # terminal is enough.
        for word in re.split(r"[\s\-—|:/\\]+", cleaned):
            if word in TERMINAL_APPS:
                return True
    return False


class _DesktopTool:
    """Shared plumbing: one desktop, one way of turning errors into results."""

    def __init__(self, desktop: Desktop) -> None:
        self.desktop = desktop

    def _guard(self, call: Any, *args: Any, **kwargs: Any) -> ToolResult:
        """Run an action, turning the expected failures into readable results.

        A failed action is information the model should reason about, not an
        exception - the same rule the shell tool follows. :class:`Aborted` is
        the exception: a kill switch a tool could swallow would not be one.
        """
        try:
            result = call(*args, **kwargs)
        except (PerceptionUnavailable, ActuationUnavailable) as exc:
            return ToolResult(ok=False, error=str(exc), metadata={"available": False})
        except (UnknownKey, ActionRefused) as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=result.ok,
            output=result.for_model() if result.ok else "",
            error=None if result.ok else result.detail,
            metadata={"method": result.method, "window": result.window},
        )

    def _refuse_terminal(self, what: str) -> ToolResult | None:
        """Block keyboard input aimed at a shell. See the module docstring."""
        try:
            snapshot = self.desktop.snapshot()
        except (PerceptionUnavailable, ActuationUnavailable):
            return None  # cannot tell; the action's own guards still apply
        if looks_like_terminal(snapshot.window_title, snapshot.process):
            return ToolResult(
                ok=False,
                error=(
                    f"refused: the focused window is a terminal ({snapshot.window_title}), "
                    f"and {what} there would run a command that Victor's safety layer "
                    "never sees. Use the shell tool instead - it classifies what you "
                    "run, asks before anything destructive, and records it."
                ),
                metadata={"refused": "terminal"},
            )
        return None


class ScreenReadTool(_DesktopTool):
    """What is on screen right now. Read-only, and free."""

    def __init__(self, desktop: Desktop) -> None:
        super().__init__(desktop)
        self.spec = ToolSpec(
            name="screen_read",
            description=(
                "List the controls in the focused window, each with an index. "
                "Costs nothing and sends nothing anywhere - the operating system "
                "already knows this. Read the screen before clicking, and again "
                "after anything that might have changed it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Only show elements whose label contains this text.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many elements to return. Default 60.",
                    },
                },
            },
            mutating=False,
        )

    def run(self, filter: str = "", limit: int = 60) -> ToolResult:  # noqa: A002
        try:
            snapshot = self.desktop.snapshot(refresh=True)
        except (PerceptionUnavailable, ActuationUnavailable) as exc:
            return ToolResult(ok=False, error=str(exc), metadata={"available": False})

        if filter:
            matches = snapshot.find(filter)
            if not matches:
                return ToolResult(
                    ok=True,
                    output=(
                        f"Nothing in {snapshot.window_title} matches {filter!r}. "
                        f"The window has {len(snapshot)} elements."
                    ),
                )
            body = "\n".join(e.render() for e in matches[:limit])
            return ToolResult(ok=True, output=f"{snapshot.window_title}\n{body}")

        return ToolResult(
            ok=True,
            output=snapshot.render(limit=limit),
            metadata={"elements": len(snapshot), "cost": 0},
        )


class ClickTool(_DesktopTool):
    """Press a control by its index."""

    def __init__(self, desktop: Desktop) -> None:
        super().__init__(desktop)
        self.spec = ToolSpec(
            name="click",
            description=(
                "Click an element from the most recent screen_read, by index. "
                "You must also pass its label exactly as shown: Victor re-reads "
                "the screen and refuses the click if the index no longer points "
                "at that label, which is how it avoids clicking the wrong thing "
                "when a list has re-sorted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "The [n] from screen_read."},
                    "label": {
                        "type": "string",
                        "description": "The label shown for that index, for verification.",
                    },
                    "button": {"type": "string", "enum": ["left", "right"]},
                    "double": {"type": "boolean", "description": "Double-click instead."},
                },
                "required": ["index", "label"],
            },
            mutating=True,
        )

    def run(
        self, index: int, label: str = "", button: str = "left", double: bool = False
    ) -> ToolResult:
        return self._guard(
            self.desktop.click, int(index), label or None, button=button, double=bool(double)
        )


class TypeTextTool(_DesktopTool):
    """Type into a field."""

    def __init__(self, desktop: Desktop) -> None:
        super().__init__(desktop)
        self.spec = ToolSpec(
            name="type_text",
            description=(
                "Type text into the focused field, or into a specific element by "
                "index. Set submit to press return afterwards. Not for running "
                "commands - use the shell tool for that."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "into": {
                        "type": "integer",
                        "description": "Index of the field to type into. Optional.",
                    },
                    "label": {"type": "string", "description": "That field's label, to verify."},
                    "submit": {"type": "boolean", "description": "Press return after typing."},
                },
                "required": ["text"],
            },
            mutating=True,
        )

    def run(
        self, text: str, into: int | None = None, label: str = "", submit: bool = False
    ) -> ToolResult:
        refusal = self._refuse_terminal("typing")
        if refusal is not None:
            return refusal
        return self._guard(
            self.desktop.type_text,
            str(text),
            into=None if into is None else int(into),
            expect=label or None,
            submit=bool(submit),
        )


class PressKeysTool(_DesktopTool):
    """Press a keyboard shortcut."""

    def __init__(self, desktop: Desktop) -> None:
        super().__init__(desktop)
        self.spec = ToolSpec(
            name="press_keys",
            description=(
                "Press a keyboard shortcut. Write 'mod' for the platform's main "
                "shortcut key - mod+s saves on both Windows and macOS. Separate "
                "chords with spaces to press them in order: 'mod+a delete'. "
                "backspace always deletes leftwards, delete always rightwards."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keys": {"type": "string", "description": "e.g. 'mod+s' or 'mod+a delete'"}
                },
                "required": ["keys"],
            },
            mutating=True,
        )

    def run(self, keys: str) -> ToolResult:
        refusal = self._refuse_terminal("pressing keys")
        if refusal is not None:
            return refusal
        return self._guard(self.desktop.press_keys, str(keys))


class ScrollTool(_DesktopTool):
    """Scroll the focused window. Declared non-mutating: it reveals, it does not change."""

    def __init__(self, desktop: Desktop) -> None:
        super().__init__(desktop)
        self.spec = ToolSpec(
            name="scroll",
            description=(
                "Scroll the focused window to bring more elements into the tree. "
                "Use this when screen_read says the walk hit its limit or the "
                "thing you want is not listed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["down", "up", "left", "right"]},
                    "amount": {"type": "integer", "description": "Lines to scroll. Default 3."},
                },
            },
            mutating=False,
        )

    def run(self, direction: str = "down", amount: int = 3) -> ToolResult:
        return self._guard(self.desktop.scroll, int(amount), str(direction))


class FocusAppTool(_DesktopTool):
    """Bring an application forward, or start it."""

    def __init__(self, desktop: Desktop) -> None:
        super().__init__(desktop)
        self.spec = ToolSpec(
            name="open_app",
            description=(
                "Bring an application to the front by name, launching it if it is "
                "not running. Give the plain application name, not a path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "e.g. 'Chrome', 'File Explorer'"},
                    "launch": {
                        "type": "boolean",
                        "description": "Start it if it is not already running. Default true.",
                    },
                },
                "required": ["name"],
            },
            mutating=True,
        )

    def run(self, name: str, launch: bool = True) -> ToolResult:
        cleaned = str(name).strip()
        if not _APP_NAME.match(cleaned):
            # An "application name" containing a path or a metacharacter is
            # either a mistake or an attempt to reach the shell sideways.
            return ToolResult(
                ok=False,
                error=(
                    f"{cleaned!r} is not a plain application name. Pass a name like "
                    "'Chrome', not a path or a command line."
                ),
            )
        action = self.desktop.open_app if launch else self.desktop.focus_app
        return self._guard(action, cleaned)


def build_desktop_tools(
    *,
    desktop: Desktop | None = None,
    kill_switch: Any | None = None,
    app: str | None = None,
) -> list[Any]:
    """The desktop tool set, sharing one façade.

    Constructing :class:`Desktop` does not touch the operating system - the
    backends load lazily - so this is safe to call on a machine with no
    accessibility support. The tools report the reason when used.
    """
    shared = desktop or Desktop(kill_switch=kill_switch, app=app)
    return [
        ScreenReadTool(shared),
        ClickTool(shared),
        TypeTextTool(shared),
        PressKeysTool(shared),
        ScrollTool(shared),
        FocusAppTool(shared),
    ]
