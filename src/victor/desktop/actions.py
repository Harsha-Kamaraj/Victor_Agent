"""Acting on what P4 can see.

This is where the project's central bet gets cashed. P4 turned a window into a
numbered list; this module turns an integer from that list back into a handle
the operating system already gave us, and asks the OS to press it. No pixel is
ever chosen by a model. The only coordinates that exist here come from a
rectangle the accessibility layer reported, and they are used only when a
control offers no programmatic action at all.

Three properties are worth stating up front, because they are the difference
between a demo and something you would leave running.

**The index is re-verified before it is used.** A snapshot is a photograph, and
a list that re-sorts between the photograph and the click will hand element 7 to
a different button. So every action re-reads the tree and checks that element 7
still carries the label the model thought it was clicking. A mismatch is
refused, not guessed at: the model is told what it found instead and asked to
look again. This costs about 20 ms and removes an entire class of "it clicked
the wrong thing" failures.

**Modifiers are always released.** A crash between key-down and key-up leaves
Ctrl held on the user's machine, which is a far worse outcome than a failed
action - every subsequent keystroke becomes a shortcut. Releases happen in a
``finally``, and the abort path runs them too. On macOS "held" is subtler than
a key being down: the window server's modifier *flags* are inherited by every
newly created event, so a chord leaks onto whatever is posted next until the
state is explicitly cleared. See :meth:`MacActuator.release_modifiers`.

**Every action checks the kill switch first.** Stopping mid-task must actually
stop, including between the two halves of a double click.

The :class:`Actuator` protocol exists for the same reason P4's ``Backend`` does:
:class:`FakeActuator` records what would have happened, so resolution,
verification, kill-switch handling and error reporting are all tested on a
machine with no desktop at all.
"""

from __future__ import annotations

import contextlib
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..errors import VictorError
from . import keys as keymap
from .elements import Element, Snapshot
from .uia import PerceptionUnavailable, TreeReader

SETTLE_SECONDS = 0.35
"""How long to let a window redraw before reading it again.

Chosen from what the tree looks like rather than what feels responsive: read too
soon after a click and the snapshot shows the *previous* screen, which the model
then reasons about. A wrong answer delivered quickly is the failure mode this
number exists to prevent."""

KEYSTROKE_INTERVAL = 0.004
"""Pause between synthesised keystrokes.

Not politeness - applications drop events posted faster than they drain their
queue, and the symptom is a silently truncated string rather than an error.
Four milliseconds types about 250 characters a second, which is faster than
anyone and slow enough for every application tried."""

TYPE_CHUNK = 40
"""Characters between kill-switch checks while typing.

Typing is the one action that takes meaningful time, so an abort should land
mid-string rather than after the last character. It is *not* the number of
characters per event: one event carries exactly one character, because that is
what applications expect - see :meth:`MacActuator.type_text`."""


class ActuationUnavailable(VictorError):
    """Nothing on this machine can drive the mouse and keyboard."""

    exit_code = 5


class ActionRefused(VictorError):
    """The action was well-formed but must not be performed as asked."""

    exit_code = 2


@dataclass(frozen=True, slots=True)
class ActionResult:
    """What happened, in terms the model can act on."""

    ok: bool
    detail: str
    method: str = ""
    """``accessibility`` when the OS performed the action on the control itself,
    ``synthetic`` when a real mouse or keyboard event had to be posted. The
    ratio between the two is a quality signal: synthetic clicks are the ones
    that miss."""
    window: str = ""
    element_count: int | None = None

    def for_model(self) -> str:
        parts = [self.detail]
        if self.window:
            count = "" if self.element_count is None else f", {self.element_count} elements"
            parts.append(f"Now showing: {self.window}{count}.")
        return " ".join(parts)


@runtime_checkable
class Actuator(Protocol):
    """Posts input to the operating system. One implementation per platform."""

    name: str

    def available(self) -> tuple[bool, str]: ...

    def press(self, element: Element) -> ActionResult:
        """Perform the control's own action, if it has one."""
        ...

    def click_point(self, x: int, y: int, *, button: str = "left", count: int = 1) -> ActionResult:
        ...

    def type_text(self, text: str) -> ActionResult: ...

    def set_value(self, element: Element, text: str) -> ActionResult:
        """Write straight into a field, bypassing the keyboard entirely."""
        ...

    def key(self, chord: keymap.Chord) -> ActionResult: ...

    def scroll(self, dx: int, dy: int) -> ActionResult: ...

    def focus_app(self, name: str) -> ActionResult: ...

    def launch_app(self, name: str) -> ActionResult: ...

    def release_modifiers(self) -> None: ...


# --- macOS -----------------------------------------------------------------


class MacActuator:
    """Quartz events and AX actions.

    Prefers ``AXPress`` on the element over a synthetic click for the usual
    reason: pressing the control is what a button *is*, while clicking its
    centre is a guess that happens to be right most of the time. It is also
    invisible - no cursor jumps across the screen while the agent works.
    """

    name = "mac"

    def __init__(self) -> None:
        self._quartz: Any = None
        self._services: Any = None
        self._workspace: Any = None
        self._source: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._quartz is not None:
            return self._quartz, self._services
        if platform.system() != "Darwin":
            raise ActuationUnavailable(
                f"the Quartz actuator is macOS-only; this is {platform.system()}"
            )
        try:
            import ApplicationServices
            import Quartz
            from AppKit import NSWorkspace
        except ImportError as exc:
            raise ActuationUnavailable(
                f"PyObjC is not installed ({exc}). pip install -e '.[desktop]'"
            ) from exc
        self._quartz, self._services, self._workspace = Quartz, ApplicationServices, NSWorkspace
        # One HID event source, reused for every synthesised event, so what is
        # posted is indistinguishable from a keyboard as far as the receiving
        # application is concerned.
        #
        # A NULL source also works on everything tried here - this is not a
        # workaround for a bug, and it was briefly mistaken for one while
        # chasing a TextEdit failure that turned out to be a modal alert. It is
        # kept because it is the conventional form, it costs one object per
        # process, and "the event has no source" is one fewer difference to
        # think about when an application does ignore an event.
        self._source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        return Quartz, ApplicationServices

    def available(self) -> tuple[bool, str]:
        try:
            _, services = self._load()
        except ActuationUnavailable as exc:
            return False, str(exc)
        if not services.AXIsProcessTrusted():
            return False, (
                "macOS has not granted Accessibility permission to this program. "
                "Grant it in System Settings > Privacy & Security > Accessibility. "
                "Without it Victor can neither read nor drive other applications."
            )
        return True, "Quartz events ready"

    # -- acting on a control ----------------------------------------------

    def _actions(self, handle: Any) -> list[str]:
        _, services = self._load()
        try:
            err, names = services.AXUIElementCopyActionNames(handle, None)
        except Exception:
            return []
        return list(names) if err == 0 and names else []

    def press(self, element: Element) -> ActionResult:
        _, services = self._load()
        handle = element.handle
        if handle is None:
            return ActionResult(False, "this element has no accessibility handle")

        available = self._actions(handle)
        # In preference order: press it, confirm it, pick it. AXPress covers
        # buttons, links and menu items; the others cover controls that model
        # themselves as a choice rather than a push.
        for action in ("AXPress", "AXConfirm", "AXPick", "AXOpen"):
            if action not in available:
                continue
            try:
                err = services.AXUIElementPerformAction(handle, action)
            except Exception as exc:  # noqa: BLE001 - report, then fall back
                return ActionResult(False, f"{action} raised {exc}")
            if err == 0:
                return ActionResult(
                    True, f"pressed {element.label!r}", method="accessibility"
                )
        return ActionResult(
            False,
            f"{element.label!r} offers no press action"
            + (f" (only {', '.join(available)})" if available else ""),
        )

    def set_value(self, element: Element, text: str) -> ActionResult:
        _, services = self._load()
        if element.handle is None:
            return ActionResult(False, "this element has no accessibility handle")
        try:
            err = services.AXUIElementSetAttributeValue(
                element.handle, services.kAXValueAttribute, text
            )
        except Exception as exc:  # noqa: BLE001
            return ActionResult(False, f"setting the value raised {exc}")
        if err == 0:
            return ActionResult(
                True, f"set {element.label!r} to {text!r}", method="accessibility"
            )
        return ActionResult(False, f"{element.label!r} refused a direct value ({err})")

    # -- synthetic input ---------------------------------------------------

    def click_point(self, x: int, y: int, *, button: str = "left", count: int = 1) -> ActionResult:
        quartz, _ = self._load()
        if button == "right":
            down, up, kind = (
                quartz.kCGEventRightMouseDown,
                quartz.kCGEventRightMouseUp,
                quartz.kCGMouseButtonRight,
            )
        else:
            down, up, kind = (
                quartz.kCGEventLeftMouseDown,
                quartz.kCGEventLeftMouseUp,
                quartz.kCGMouseButtonLeft,
            )

        point = quartz.CGPointMake(float(x), float(y))
        # Move first. Menus and hover-activated controls open on the move, not
        # on the press, and clicking without moving misses them entirely.
        move = quartz.CGEventCreateMouseEvent(
            self._source, quartz.kCGEventMouseMoved, point, kind
        )
        quartz.CGEventPost(quartz.kCGHIDEventTap, move)

        for click in range(1, count + 1):
            for event_type in (down, up):
                event = quartz.CGEventCreateMouseEvent(self._source, event_type, point, kind)
                # A double click is one click with a click-count of two, not two
                # clicks - applications read this field to tell them apart.
                quartz.CGEventSetIntegerValueField(event, quartz.kCGMouseEventClickState, click)
                quartz.CGEventPost(quartz.kCGHIDEventTap, event)
            time.sleep(0.02)

        what = "double-clicked" if count > 1 else f"{button}-clicked"
        return ActionResult(True, f"{what} at ({x}, {y})", method="synthetic")

    def type_text(self, text: str) -> ActionResult:
        quartz, _ = self._load()
        # The character rides on the event as a Unicode string rather than as a
        # keycode, so this works on any keyboard layout - mapping characters to
        # keycodes types gibberish on Dvorak, AZERTY or anything non-US.
        #
        # One character per event, deliberately. Setting a whole string on a
        # single event looks like it should work and does in text fields, but
        # anything handling `keyDown:` itself sees one keystroke and drops the
        # rest: Calculator takes "8*8" and shows 8. A silently truncated string
        # is a worse failure than a slow one.
        for character in text:
            for pressed in (True, False):
                event = quartz.CGEventCreateKeyboardEvent(self._source, 0, pressed)
                # Clear the flags explicitly. A new event *inherits the current
                # global modifier state*, so after a chord like cmd+a every
                # subsequent keystroke silently arrives as cmd+<key> - typing
                # "Victor" into a document becomes six menu shortcuts, and the
                # symptom is a tool that reports success while nothing happens.
                quartz.CGEventSetFlags(event, 0)
                quartz.CGEventKeyboardSetUnicodeString(event, 1, character)
                quartz.CGEventPost(quartz.kCGHIDEventTap, event)
            time.sleep(KEYSTROKE_INTERVAL)
        return ActionResult(True, f"typed {len(text)} characters", method="synthetic")

    def key(self, chord: keymap.Chord) -> ActionResult:
        quartz, _ = self._load()
        code, flags = keymap.mac_keycode(chord)
        # Modifiers ride on the event as flags rather than as separate key-down
        # events, so no physical modifier key is ever left held. The key-up
        # carries no flags, which is what a real keyboard does - release the
        # key, then release the modifiers - and what stops the flags leaking
        # onto whatever is posted next.
        for pressed, carried in ((True, flags), (False, 0)):
            event = quartz.CGEventCreateKeyboardEvent(self._source, code, pressed)
            quartz.CGEventSetFlags(event, carried)
            quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        time.sleep(0.02)
        self.release_modifiers()
        return ActionResult(True, f"pressed {chord}", method="synthetic")

    def scroll(self, dx: int, dy: int) -> ActionResult:
        quartz, _ = self._load()
        event = quartz.CGEventCreateScrollWheelEvent(
            self._source, quartz.kCGScrollEventUnitLine, 2, int(dy), int(dx)
        )
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        return ActionResult(True, f"scrolled ({dx}, {dy}) lines", method="synthetic")

    # -- applications ------------------------------------------------------

    def focus_app(self, name: str) -> ActionResult:
        self._load()
        workspace = self._workspace.sharedWorkspace()
        wanted = name.strip().lower()
        for app in workspace.runningApplications():
            label = app.localizedName() or ""
            if label.lower() == wanted or wanted in label.lower():
                # 1 << 1 is NSApplicationActivateIgnoringOtherApps: bring it
                # forward even though Victor is not the active application.
                app.activateWithOptions_(1 << 1)
                time.sleep(SETTLE_SECONDS)
                return ActionResult(True, f"focused {label}", method="accessibility")
        return ActionResult(False, f"no running application matches {name!r}")

    def launch_app(self, name: str) -> ActionResult:
        focused = self.focus_app(name)
        if focused.ok:
            return ActionResult(True, f"{name} was already running; brought it forward")
        opener = shutil.which("open") or "/usr/bin/open"
        result = subprocess.run(  # noqa: S603 - fixed binary, name is validated upstream
            [opener, "-a", name],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            return ActionResult(False, (result.stderr or "could not open it").strip())
        time.sleep(SETTLE_SECONDS * 3)  # applications are slower than windows
        return ActionResult(True, f"launched {name}")

    #: Virtual keycodes of the modifier keys, for clearing the flag state.
    _MODIFIER_KEYCODES = (55, 56, 58, 59, 63)  # cmd, shift, option, control, fn

    def release_modifiers(self) -> None:
        """Clear the window server's idea of which modifiers are down.

        This method was originally a no-op, on the reasoning that flags ride on
        each event so nothing can be left held. That was wrong in a way that
        took a while to see: the *global* modifier state persists, and every
        subsequently created event inherits it. After one cmd+a, a keystroke
        that should type "V" arrives as cmd+V, and the tool reports success
        while nothing appears.

        Posting a key-up for each modifier with no flags set puts the state
        back. It is harmless when nothing was held, which is why it runs
        unconditionally rather than only after a chord.
        """
        if self._quartz is None:
            return
        quartz = self._quartz
        for code in self._MODIFIER_KEYCODES:
            event = quartz.CGEventCreateKeyboardEvent(self._source, code, False)
            quartz.CGEventSetFlags(event, 0)
            quartz.CGEventPost(quartz.kCGHIDEventTap, event)


# --- Windows ---------------------------------------------------------------


class WindowsActuator:
    """UI Automation patterns, with real input as the fallback.

    Mirrors :class:`MacActuator` deliberately. Where macOS has ``AXPress``,
    UI Automation has control patterns - ``Invoke`` for buttons, ``Toggle`` for
    checkboxes, ``SelectionItem`` for list rows - and the same preference
    applies: use the pattern, and post a real click only when there is none.
    """

    name = "windows"

    def __init__(self) -> None:
        self._auto: Any = None
        self._held: list[int] = []

    def _load(self) -> Any:
        if self._auto is not None:
            return self._auto
        if platform.system() != "Windows":
            raise ActuationUnavailable(
                f"the UI Automation actuator is Windows-only; this is {platform.system()}"
            )
        try:
            import uiautomation
        except ImportError as exc:
            raise ActuationUnavailable(
                f"uiautomation is not installed ({exc}). pip install -e '.[desktop]'"
            ) from exc
        self._auto = uiautomation
        return uiautomation

    def available(self) -> tuple[bool, str]:
        try:
            self._load()
        except ActuationUnavailable as exc:
            return False, str(exc)
        return True, "UI Automation input ready"

    def press(self, element: Element) -> ActionResult:
        self._load()
        handle = element.handle
        if handle is None:
            return ActionResult(False, "this element has no accessibility handle")

        # In preference order, matching what the control claims to be.
        attempts = (
            ("GetInvokePattern", "Invoke", "pressed"),
            ("GetTogglePattern", "Toggle", "toggled"),
            ("GetSelectionItemPattern", "Select", "selected"),
            ("GetExpandCollapsePattern", "Expand", "expanded"),
        )
        for getter, method, verb in attempts:
            try:
                pattern = getattr(handle, getter)()
            except Exception:
                continue
            if pattern is None:
                continue
            try:
                getattr(pattern, method)()
            except Exception:
                continue
            return ActionResult(
                True, f"{verb} {element.label!r}", method="accessibility"
            )
        return ActionResult(False, f"{element.label!r} supports no invocable pattern")

    def set_value(self, element: Element, text: str) -> ActionResult:
        self._load()
        try:
            pattern = element.handle.GetValuePattern()
            pattern.SetValue(text)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(False, f"{element.label!r} refused a direct value ({exc})")
        return ActionResult(True, f"set {element.label!r} to {text!r}", method="accessibility")

    def click_point(self, x: int, y: int, *, button: str = "left", count: int = 1) -> ActionResult:
        auto = self._load()
        click = auto.RightClick if button == "right" else auto.Click
        for _ in range(count):
            click(x, y, waitTime=0.05)
        what = "double-clicked" if count > 1 else f"{button}-clicked"
        return ActionResult(True, f"{what} at ({x}, {y})", method="synthetic")

    def type_text(self, text: str) -> ActionResult:
        auto = self._load()
        # SendKeys reads braces as special-key syntax, so a literal brace has to
        # be escaped or `{"a": 1}` becomes a lookup for a key named `"a"`.
        #
        # Escaping happens *after* slicing, not before: escaping first and then
        # slicing can cut `{{}` in half and send the two halves as separate
        # sequences, which is the sort of bug that only shows up on the one
        # input that straddles the boundary.
        escaped = text.replace("{", "{{}").replace("}", "{}}")
        auto.SendKeys(escaped, interval=KEYSTROKE_INTERVAL, waitTime=0)
        return ActionResult(True, f"typed {len(text)} characters", method="synthetic")

    def key(self, chord: keymap.Chord) -> ActionResult:
        auto = self._load()
        modifiers, code = keymap.windows_keycodes(chord)
        # Record the intent *before* pressing anything. The obvious order -
        # press, then append - leaves a window one statement wide where a key
        # is physically down and untracked, so the `finally` below releases
        # nothing. A COM error there, or a KeyboardInterrupt (which is how the
        # kill switch is triggered), leaves Ctrl held down for the whole
        # machine, not just for Victor.
        self._held.extend(modifiers)
        try:
            for modifier in modifiers:
                auto.PressKey(modifier)
            auto.PressKey(code)
            auto.ReleaseKey(code)
        finally:
            self.release_modifiers()
        return ActionResult(True, f"pressed {chord}", method="synthetic")

    def scroll(self, dx: int, dy: int) -> ActionResult:
        auto = self._load()
        if dy:
            auto.WheelDown(-dy) if dy < 0 else auto.WheelUp(dy)
        return ActionResult(True, f"scrolled ({dx}, {dy}) lines", method="synthetic")

    def focus_app(self, name: str) -> ActionResult:
        """Bring a window forward, and check that it actually came forward.

        ``SetActive`` cannot beat the Windows foreground lock: when the calling
        process is not itself in the foreground, the request is downgraded to
        flashing the taskbar button, and it reports no error. Without the check
        below this returned ``ok=True`` while the foreground window was still
        whatever the user was using - the agent then read one window and typed
        into another.

        The verification is the fix. A workaround for the lock could be added on
        top, but a workaround that is not verified is how this got here.
        """
        auto = self._load()
        try:
            window = auto.WindowControl(searchDepth=1, SubName=name)
            if not window.Exists(maxSearchSeconds=2):
                return ActionResult(False, f"no window matches {name!r}")
            window.SetActive()
            window.SetTopmost(False)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(False, f"could not focus {name!r}: {exc}")

        time.sleep(SETTLE_SECONDS)
        foreground = self._foreground_title(auto)
        wanted = name.strip().casefold()
        if wanted and wanted not in (foreground or "").casefold():
            return ActionResult(
                False,
                f"asked Windows to focus {name!r} but the foreground window is "
                f"{foreground or 'unknown'!r}. Windows refuses foreground changes "
                "from a background process; click the window once, or start "
                "Victor from a terminal that is already in front.",
            )
        return ActionResult(True, f"focused {foreground or name}", method="accessibility")

    @staticmethod
    def _foreground_title(auto: Any) -> str:
        """The title of whatever is actually in front right now."""
        try:
            control = auto.GetForegroundControl()
        except Exception:
            return ""
        return str(getattr(control, "Name", "") or "") if control is not None else ""

    def launch_app(self, name: str) -> ActionResult:
        # Ask whether it is running *before* trying to focus it. Now that
        # focus_app verifies its own result, it can fail for two unrelated
        # reasons - not running, and running but blocked by the foreground lock
        # - and launching a second copy is only right for the first.
        auto = self._load()
        running = False
        with contextlib.suppress(Exception):
            window = auto.WindowControl(searchDepth=1, SubName=name)
            running = bool(window.Exists(maxSearchSeconds=1))
        if running:
            return self.focus_app(name)

        try:
            subprocess.Popen(  # noqa: S603 - name is validated upstream
                ["cmd", "/c", "start", "", name],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            return ActionResult(False, f"could not launch {name!r}: {exc}")
        time.sleep(SETTLE_SECONDS * 3)
        return ActionResult(True, f"launched {name}")

    def release_modifiers(self) -> None:
        """Release every modifier, tracked or not.

        Unconditional, to match :meth:`MacActuator.release_modifiers`. Gating
        this on ``_held`` made it depend on the bookkeeping being correct at
        exactly the moment the bookkeeping was most likely to be wrong - and a
        redundant ReleaseKey on a key that is already up does nothing, while a
        missed one hands the user a machine where every keystroke is a shortcut.
        """
        if self._auto is None:
            return
        auto = self._auto
        # Tracked ones first and in reverse, so a chord unwinds in the order it
        # was built; then the full set, in case anything was pressed off-book.
        ordered = list(reversed(self._held))
        ordered += [c for c in keymap.WINDOWS_MODIFIER_CODES.values() if c not in ordered]
        self._held.clear()
        for modifier in ordered:
            with contextlib.suppress(Exception):
                auto.ReleaseKey(modifier)


# --- testing ---------------------------------------------------------------


@dataclass
class FakeActuator:
    """Records what would have happened. The reason P5 is testable anywhere."""

    name: str = "fake"
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    pressable: bool = True
    """When False, elements report no press action, forcing the synthetic path."""
    released: int = 0

    def available(self) -> tuple[bool, str]:
        return True, "fake actuator"

    def press(self, element: Element) -> ActionResult:
        self.calls.append(("press", (element.index, element.label)))
        if not self.pressable:
            return ActionResult(False, f"{element.label!r} offers no press action")
        return ActionResult(True, f"pressed {element.label!r}", method="accessibility")

    def click_point(self, x: int, y: int, *, button: str = "left", count: int = 1) -> ActionResult:
        self.calls.append(("click_point", (x, y, button, count)))
        return ActionResult(True, f"clicked at ({x}, {y})", method="synthetic")

    def type_text(self, text: str) -> ActionResult:
        self.calls.append(("type_text", (text,)))
        return ActionResult(True, f"typed {len(text)} characters", method="synthetic")

    def set_value(self, element: Element, text: str) -> ActionResult:
        self.calls.append(("set_value", (element.index, text)))
        if not self.pressable:
            return ActionResult(False, "no value pattern")
        return ActionResult(True, f"set {element.label!r}", method="accessibility")

    def key(self, chord: keymap.Chord) -> ActionResult:
        self.calls.append(("key", (str(chord),)))
        return ActionResult(True, f"pressed {chord}", method="synthetic")

    def scroll(self, dx: int, dy: int) -> ActionResult:
        self.calls.append(("scroll", (dx, dy)))
        return ActionResult(True, f"scrolled ({dx}, {dy})", method="synthetic")

    def focus_app(self, name: str) -> ActionResult:
        self.calls.append(("focus_app", (name,)))
        return ActionResult(True, f"focused {name}", method="accessibility")

    def launch_app(self, name: str) -> ActionResult:
        self.calls.append(("launch_app", (name,)))
        return ActionResult(True, f"launched {name}")

    def release_modifiers(self) -> None:
        self.released += 1


def select_actuator() -> Actuator:
    """The right input driver for this operating system."""
    system = platform.system()
    if system == "Windows":
        return WindowsActuator()
    if system == "Darwin":
        return MacActuator()
    return UnsupportedActuator(system)


class UnsupportedActuator:
    """Placeholder for platforms with no actuation backend."""

    name = "unsupported"

    def __init__(self, system: str) -> None:
        self.system = system

    def available(self) -> tuple[bool, str]:
        return False, (
            f"no desktop actuation for {self.system}. Victor drives Windows via "
            "UI Automation and macOS via the Accessibility API; everything else "
            "here works, just not clicking and typing."
        )

    def _fail(self) -> ActionResult:
        raise ActuationUnavailable(self.available()[1])

    def press(self, element: Element) -> ActionResult:
        return self._fail()

    def click_point(self, x: int, y: int, *, button: str = "left", count: int = 1) -> ActionResult:
        return self._fail()

    def type_text(self, text: str) -> ActionResult:
        return self._fail()

    def set_value(self, element: Element, text: str) -> ActionResult:
        return self._fail()

    def key(self, chord: keymap.Chord) -> ActionResult:
        return self._fail()

    def scroll(self, dx: int, dy: int) -> ActionResult:
        return self._fail()

    def focus_app(self, name: str) -> ActionResult:
        return self._fail()

    def launch_app(self, name: str) -> ActionResult:
        return self._fail()

    def release_modifiers(self) -> None:
        return None


# --- the façade ------------------------------------------------------------


def normalise_label(text: str) -> str:
    """Fold a label to the form two snapshots can be compared on.

    Windows puts a ``&`` before the accelerator letter, both platforms truncate
    long labels with an ellipsis, and whitespace varies between reads of the
    same control. None of that is a difference worth refusing an action over.
    """
    folded = " ".join(text.replace("&", "").split()).casefold()
    return folded.rstrip(" .…")


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of turning an index back into a live control."""

    element: Element | None
    snapshot: Snapshot
    problem: str = ""

    @property
    def ok(self) -> bool:
        return self.element is not None


class Desktop:
    """Perception and actuation, joined - and gated on what is really there.

    Every public method here follows the same shape: check the kill switch,
    re-read the tree, verify the target still is what the model thought, act,
    then report the screen the user is now looking at. The last part matters for
    step economy: a click that also returns the new window title saves a whole
    perception round trip, and every round trip is an API call.
    """

    def __init__(
        self,
        reader: TreeReader | None = None,
        actuator: Actuator | None = None,
        *,
        kill_switch: Any | None = None,
        settle: float = SETTLE_SECONDS,
        app: str | None = None,
    ) -> None:
        self.reader = reader or TreeReader(app=app)
        self.actuator = actuator or select_actuator()
        self.kill_switch = kill_switch
        self.settle = settle

    def available(self) -> tuple[bool, str]:
        ok, detail = self.reader.available()
        if not ok:
            return False, detail
        return self.actuator.available()

    # -- resolution --------------------------------------------------------

    def snapshot(self, *, refresh: bool = False) -> Snapshot:
        return self.reader.snapshot(refresh=refresh)

    def resolve(self, index: int, expect: str | None = None) -> Resolution:
        """Find element ``index`` in a *fresh* read, and check it is still it."""
        snapshot = self.reader.snapshot(refresh=True)
        element = snapshot.by_index(index)

        if element is None:
            hint = f"indices run 0 to {len(snapshot) - 1}" if len(snapshot) else "nothing is listed"
            return Resolution(None, snapshot, f"there is no element {index}; {hint}")

        if expect:
            wanted, found = normalise_label(expect), normalise_label(element.label)
            if wanted != found and wanted not in found and found not in wanted:
                # The screen moved under us. Naming both labels lets the model
                # recover in one step instead of retrying the same wrong index.
                match = snapshot.find(expect)
                suggestion = (
                    f" {expect!r} is now element {match[0].index}." if match else ""
                )
                return Resolution(
                    None,
                    snapshot,
                    f"element {index} is {element.label!r}, not {expect!r} - "
                    f"the screen changed since you looked.{suggestion}",
                )

        if not element.enabled:
            return Resolution(None, snapshot, f"{element.label!r} is disabled")
        if element.rect.empty:
            return Resolution(None, snapshot, f"{element.label!r} has no visible area")
        return Resolution(element, snapshot)

    # -- actions -----------------------------------------------------------

    def click(
        self,
        index: int,
        expect: str | None = None,
        *,
        button: str = "left",
        double: bool = False,
    ) -> ActionResult:
        """Press element ``index``, verifying it is still ``expect`` first."""
        self._check_abort()
        resolved = self.resolve(index, expect)
        if not resolved.ok:
            return ActionResult(False, resolved.problem)
        element = resolved.element
        assert element is not None

        try:
            # A right click or a double click is asking for a mouse gesture
            # specifically; AXPress is a left single click and nothing else.
            if button == "left" and not double:
                result = self.actuator.press(element)
                if result.ok:
                    return self._after(result)
            x, y = element.rect.centre
            self._check_abort()
            result = self.actuator.click_point(
                x, y, button=button, count=2 if double else 1
            )
        finally:
            self.actuator.release_modifiers()
        return self._after(result)

    def type_text(
        self,
        text: str,
        *,
        into: int | None = None,
        expect: str | None = None,
        submit: bool = False,
    ) -> ActionResult:
        """Type into the focused control, or into element ``into``."""
        self._check_abort()
        if into is not None:
            resolved = self.resolve(into, expect)
            if not resolved.ok:
                return ActionResult(False, resolved.problem)
            element = resolved.element
            assert element is not None
            # Writing the value straight into the field beats typing it: it is
            # atomic, so a half-typed string never reaches the application, and
            # it cannot be stolen by a window that steals focus mid-word.
            direct = self.actuator.set_value(element, text)
            if direct.ok:
                return self._after(direct if not submit else self._submit(direct))
            focus = self.click(into, expect)
            if not focus.ok:
                return focus

        try:
            result = self._type_in_chunks(text)
        finally:
            self.actuator.release_modifiers()
        if result.ok and submit:
            result = self._submit(result)
        return self._after(result)

    def _type_in_chunks(self, text: str) -> ActionResult:
        """Type in pieces, so a stop lands mid-string rather than after it.

        Typing is the only action here that takes real time - a paragraph is
        several seconds - and "stop" that waits for the paragraph to finish is
        not a stop.
        """
        if len(text) <= TYPE_CHUNK:
            return self.actuator.type_text(text)

        typed = 0
        for start in range(0, len(text), TYPE_CHUNK):
            self._check_abort()
            chunk = text[start : start + TYPE_CHUNK]
            result = self.actuator.type_text(chunk)
            if not result.ok:
                return ActionResult(False, f"{result.detail} after {typed} characters")
            typed += len(chunk)
        return ActionResult(True, f"typed {typed} characters", method="synthetic")

    def _submit(self, previous: ActionResult) -> ActionResult:
        self._check_abort()
        pressed = self.actuator.key(keymap.Chord("return"))
        if not pressed.ok:
            return pressed
        return ActionResult(
            True, f"{previous.detail} and pressed return", method=previous.method
        )

    def press_keys(self, keys: str) -> ActionResult:
        """Press one or more chords: ``"mod+a delete"`` is two presses."""
        self._check_abort()
        chords = keymap.parse_sequence(keys)
        pressed: list[str] = []
        try:
            for chord in chords:
                self._check_abort()
                result = self.actuator.key(chord)
                if not result.ok:
                    return ActionResult(False, f"{result.detail} (after {' '.join(pressed)})")
                pressed.append(str(chord))
        finally:
            self.actuator.release_modifiers()
        return self._after(ActionResult(True, f"pressed {' then '.join(pressed)}", "synthetic"))

    def scroll(self, amount: int = 3, direction: str = "down") -> ActionResult:
        self._check_abort()
        vectors = {
            "down": (0, -amount),
            "up": (0, amount),
            "left": (-amount, 0),
            "right": (amount, 0),
        }
        if direction not in vectors:
            return ActionResult(False, f"direction must be one of {', '.join(vectors)}")
        dx, dy = vectors[direction]
        return self._after(self.actuator.scroll(dx, dy))

    def focus_app(self, name: str) -> ActionResult:
        self._check_abort()
        result = self.actuator.focus_app(name)
        return self._after(result)

    def open_app(self, name: str) -> ActionResult:
        self._check_abort()
        return self._after(self.actuator.launch_app(name))

    # -- reporting ---------------------------------------------------------

    def _after(self, result: ActionResult) -> ActionResult:
        """Let the screen settle, then describe what it now shows."""
        self.reader.invalidate()
        if not result.ok:
            return result
        time.sleep(self.settle)
        try:
            snapshot = self.reader.snapshot(refresh=True)
        except (PerceptionUnavailable, ActuationUnavailable):
            # The action worked; we just cannot describe the result. Saying so
            # is better than turning a successful click into a failure.
            return result
        return ActionResult(
            ok=result.ok,
            detail=result.detail,
            method=result.method,
            window=snapshot.window_title,
            element_count=len(snapshot),
        )

    def _check_abort(self) -> None:
        if self.kill_switch is None:
            return
        if self.kill_switch.tripped:
            # Release before raising. An abort that leaves Ctrl held is worse
            # than the action it prevented.
            self.actuator.release_modifiers()
        self.kill_switch.check()
