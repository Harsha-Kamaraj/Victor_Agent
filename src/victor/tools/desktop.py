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

import contextlib
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


def _same_app(showing: str, wanted: str) -> str | bool:
    """Whether a window's owner is the application that was asked for.

    Forgiving on purpose: macOS decorates names with bidi marks, and a process
    can report "Code" for an app called "Visual Studio Code". Either name
    containing the other is enough - the point is to catch a completely
    different application, not to police spelling.
    """
    left = "".join(ch for ch in showing.lower() if ch.isalnum())
    right = "".join(ch for ch in wanted.lower() if ch.isalnum())
    if not left or not right:
        return True
    return left in right or right in left


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
            # cost 0 is stated rather than left to default: it is the claim the
            # whole project rests on, and AgentResult counts it.
            metadata={"method": result.method, "window": result.window, "cost": 0},
        )

    def _focus_target(self) -> None:
        """Bring the pinned app forward before sending it synthetic input.

        Keystrokes go to whatever the OS says is frontmost. With ``--app`` set,
        ``snapshot()`` describes the pinned window instead, so the two can name
        different windows - and then :meth:`_refuse_terminal` inspects one while
        the keys land in the other. Raising the target first collapses the
        divergence rather than trying to detect it.
        """
        app = getattr(self.desktop, "app", None)
        if not app:
            return
        with contextlib.suppress(Exception):
            self.desktop.focus_app(str(app))

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

        if snapshot.empty:
            # Two different empties, and they need different responses. A window
            # that could not be measured wants bringing into view; one with no
            # readable tree is what the vision fallback exists for.
            advice = snapshot.note or (
                f"{snapshot.window_title} reports no readable controls. "
                "This happens with canvases, games and some Electron apps. "
                "Try find_on_screen if it is available, or open_app to move "
                "to an application with a usable tree."
            )
            return ToolResult(ok=True, output=advice, metadata={"elements": 0, "cost": 0})

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

    def describe(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """What this click actually points at, for the safety layer.

        The model passes the label it was shown, and on Windows that label has
        had the file extension stripped by Explorer before Victor ever saw it.
        Read from the cached tree instead - no refresh, so this costs nothing
        and cannot change what the click resolves to a moment later.

        Never raises: enrichment failing has to leave the ordinary
        label-based classification in place, not block the call.
        """
        try:
            snapshot = self.desktop.snapshot(refresh=False)
            element = snapshot.by_index(int(arguments.get("index", -1)))
        except Exception:  # noqa: BLE001 - no tree is not a reason to refuse
            return {}
        if element is None:
            return {}
        return {
            "filename": element.filename,
            "control_type": element.control_type,
            "process": snapshot.process,
        }

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
        if into is None:
            # Typing with no target goes wherever the OS says focus is, which
            # during a run is the terminal Victor was launched from - and
            # `_refuse_terminal` below cannot catch it, because with --app
            # pinned the snapshot describes the target window while the
            # keystrokes go to the focused one. That divergence typed a message
            # into the user's shell prompt; with submit=True it would have
            # pressed return on it, running whatever was typed and bypassing
            # every check the shell tool exists to apply.
            #
            # Naming the field is also just the rule this project already has
            # for clicking: act on a control you have read, never on a guess.
            return ToolResult(
                ok=False,
                error=(
                    "type_text needs the index of the field to type into. Call "
                    "screen_read first and pass into=<index> with its label. "
                    "Without a target the text goes to whatever window happens "
                    "to be focused, which is not necessarily the app you mean."
                ),
                metadata={"refused": "no-target"},
            )
        refusal = self._refuse_terminal("typing")
        if refusal is not None:
            return refusal
        return self._guard(
            self.desktop.type_text,
            str(text),
            into=int(into),
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
        # Raise the target first: `press_keys('return')` with a pinned --app was
        # sending the keystroke to whichever window really had focus, which is
        # the terminal Victor runs in. Pressing return there submits whatever
        # sits at the prompt.
        self._focus_target()
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


class FindOnScreenTool(_DesktopTool):
    """The vision fallback, and the only tool here that costs anything.

    Exists for the surfaces with no usable tree - a canvas, a game, an Electron
    app that reports one giant unlabelled group. Everything else in this module
    is free and instant, and this one is neither, so it says so in its own
    description: the model is told the price before it chooses to pay it.

    It reports ``cost: 1`` in its metadata, which is what makes
    ``AgentResult.zero_cost_ratio`` a measurement rather than a slogan.
    """

    def __init__(self, desktop: Desktop, vision: Any, capture: Any) -> None:
        super().__init__(desktop)
        self.vision = vision
        self.capture = capture
        self.spec = ToolSpec(
            name="find_on_screen",
            description=(
                "Ask a vision model which element matches a description. Use this "
                "ONLY when screen_read did not list what you need - it spends one "
                "of a small daily allowance of image requests, while screen_read "
                "is free and unlimited. Returns an element index you can click."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What to look for, e.g. 'the blue Send button'.",
                    }
                },
                "required": ["description"],
            },
            mutating=False,
        )

    def run(self, description: str) -> ToolResult:
        from ..desktop.vision import VisionUnavailable, annotate

        try:
            snapshot = self.desktop.snapshot(refresh=True)
        except (PerceptionUnavailable, ActuationUnavailable) as exc:
            return ToolResult(ok=False, error=str(exc), metadata={"available": False})

        region = None
        if snapshot.rect is not None and not snapshot.rect.empty:
            rect = snapshot.rect
            region = (rect.left, rect.top, rect.width, rect.height)

        try:
            shot, is_new = self.capture.capture(region)
            answer = self.vision.locate(description, annotate(shot, snapshot), snapshot)
        except VisionUnavailable as exc:
            # Running out of vision has to leave a working agent. Say what
            # remains possible rather than reporting a bare failure.
            return ToolResult(
                ok=False,
                error=(
                    f"{exc}. screen_read still works and costs nothing - try that, "
                    "or scroll and read again."
                ),
                metadata={"cost": 1, "quota": "exhausted"},
            )
        except Exception as exc:  # noqa: BLE001 - a failed look is not a crash
            return ToolResult(ok=False, error=f"could not look at the screen: {exc}")

        metadata = {"cost": 1, "model": answer.model, "cached_screen": not is_new}
        if not answer.found:
            return ToolResult(
                ok=True,
                output=f"Nothing on screen matches {description!r}.",
                metadata=metadata,
            )
        return ToolResult(
            ok=True,
            output=(
                f"{answer} - click index {answer.index} with label "
                f"{answer.element.label!r}."
                if answer.element
                else str(answer)
            ),
            metadata=metadata | {"index": answer.index},
        )


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
        result = self._guard(action, cleaned)
        if not result.ok:
            return result

        # Say so when the window in front is not the one that was asked for.
        # `open_app('Messages')` reported "brought it forward" while the tree
        # being read belonged to VS Code - the app had been activated but owns
        # no window, so the reader fell through to the topmost windowed app.
        # Reporting that as plain success sent the model looking for a WhatsApp
        # chat in an editor, and it spent every remaining step there.
        showing = self._window_owner()
        if showing and not _same_app(showing, cleaned):
            return ToolResult(
                ok=False,
                error=(
                    f"asked for {cleaned!r} but the window in front is {showing!r}. "
                    f"{cleaned!r} may have no open window. Open one, or work with "
                    f"{showing!r}, or name a different application."
                ),
                output=result.output,
                metadata={**result.metadata, "showing": showing},
            )
        return result

    def _window_owner(self) -> str:
        """Which application the readable window actually belongs to."""
        try:
            snapshot = self.desktop.snapshot(refresh=True)
        except (PerceptionUnavailable, ActuationUnavailable):
            return ""
        return str(snapshot.process or "")


def build_desktop_tools(
    *,
    desktop: Desktop | None = None,
    kill_switch: Any | None = None,
    app: str | None = None,
    vision: Any | None = None,
) -> list[Any]:
    """The desktop tool set, sharing one façade.

    Constructing :class:`Desktop` does not touch the operating system - the
    backends load lazily - so this is safe to call on a machine with no
    accessibility support. The tools report the reason when used.

    ``find_on_screen`` appears only when a vision client is supplied. Offering a
    tool that cannot be served would spend a step to learn that, and the step is
    the expensive part.
    """
    shared = desktop or Desktop(kill_switch=kill_switch, app=app)
    tools: list[Any] = [
        ScreenReadTool(shared),
        ClickTool(shared),
        TypeTextTool(shared),
        PressKeysTool(shared),
        ScrollTool(shared),
        FocusAppTool(shared),
    ]
    if vision is not None:
        from ..desktop.capture import ScreenCapture

        tools.append(FindOnScreenTool(shared, vision, ScreenCapture()))
    return tools
