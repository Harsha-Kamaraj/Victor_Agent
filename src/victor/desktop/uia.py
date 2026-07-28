"""Reading the Windows UI Automation tree.

This is the phase the whole architecture is arranged around. Walking the tree
costs nothing, takes about 20 ms, and returns names and rectangles the OS
already knows - so the expensive, rate-limited vision model is needed only for
surfaces that have no usable tree.

Two things make the walk practical rather than merely possible:

**Bounded.** A real window's tree can be thousands of nodes deep in places
(a web page in Edge is the usual offender). The walk caps depth, node count and
wall-clock time, and reports that it truncated rather than pretending it saw
everything. An unbounded walk is how "20 ms" becomes "eight seconds" on the one
window you most wanted to read.

**Cached per window.** Re-walking on every agent step would dominate the loop's
latency for no benefit; the tree changes when the user or the agent changes it.
The cache is keyed on the window handle and invalidated on focus change.

The backend protocol exists so this stack is testable on a machine with no UI
Automation at all - :class:`FakeBackend` serves a literal tree, and every test
for filtering, indexing, caching and rendering runs everywhere.
"""

from __future__ import annotations

import contextlib
import platform
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..errors import VictorError
from .elements import Element, Rect, Snapshot, is_interesting

MAX_DEPTH = 12
MAX_ELEMENTS = 200
MAX_WALK_SECONDS = 2.0


class PerceptionUnavailable(VictorError):
    """The accessibility tree cannot be read on this machine."""

    exit_code = 5


@dataclass(frozen=True, slots=True)
class WalkLimits:
    """Guard rails for a tree walk."""

    max_depth: int = MAX_DEPTH
    max_elements: int = MAX_ELEMENTS
    max_seconds: float = MAX_WALK_SECONDS


@runtime_checkable
class Backend(Protocol):
    """Supplies raw windows and nodes. One implementation per platform."""

    name: str

    def available(self) -> tuple[bool, str]: ...

    def focused_window(self) -> Any | None: ...

    def window_info(self, window: Any) -> tuple[str, str, Rect]: ...

    def children(self, node: Any) -> list[Any]: ...

    def describe(self, node: Any) -> tuple[str, str, Rect, bool, bool, str, str]:
        """``(control_type, name, rect, enabled, focused, value, automation_id)``."""
        ...


class UIABackend:
    """Windows UI Automation via the ``uiautomation`` package."""

    name = "uia"

    def __init__(self, app_name: str | None = None) -> None:
        self.app_name = app_name
        self._auto = None

    def _load(self):
        if self._auto is not None:
            return self._auto
        if platform.system() != "Windows":
            raise PerceptionUnavailable(
                "UI Automation is a Windows API; this machine is "
                f"{platform.system()}. Screen perception is unavailable here."
            )
        try:
            import uiautomation
        except ImportError as exc:
            raise PerceptionUnavailable(
                f"uiautomation is not installed ({exc}). pip install -e '.[desktop]'"
            ) from exc
        self._auto = uiautomation
        return uiautomation

    def available(self) -> tuple[bool, str]:
        try:
            self._load()
        except PerceptionUnavailable as exc:
            return False, str(exc)
        from .session import session_locked

        locked, why = session_locked()
        if locked:
            return False, why
        return True, "UI Automation ready"

    def focused_window(self) -> Any | None:
        auto = self._load()
        if self.app_name:
            return self._named_window(auto, self.app_name)

        try:
            control = auto.GetFocusedControl()
        except Exception:
            control = None
        if control is None:
            return auto.GetForegroundControl()

        # Climb to the owning top-level window. The focused control is usually
        # a list item or a text box, and the agent wants everything in its
        # window - the toolbar especially, since that is where the verbs are.
        #
        # The stopping rule used to be "the parent is a Pane", on the theory
        # that the desktop root is a Pane. It is, but so is most of File
        # Explorer's internal scaffolding, and the climb stopped five levels
        # early:
        #
        #     ListControl   'Items View'                <- stopped here
        #     PaneControl   'Shell Folder View'
        #     PaneControl   'Folder Layout Pane'
        #     PaneControl   'Explorer Pane'
        #     PaneControl   ''
        #     PaneControl   'Downloads'
        #     WindowControl 'Downloads - File Explorer'  <- wanted this
        #     PaneControl   'Desktop 1'                  <- avoiding this
        #
        # That cost 101 of 248 elements, including Back, Forward, the address
        # bar, and every toolbar verb. Stopping at the first WindowControl gets
        # the right node and still never reaches the desktop root, because the
        # root sits above it.
        node = control
        for _ in range(MAX_DEPTH):
            if _control_type(node) == "WindowControl":
                return node
            try:
                parent = node.GetParentControl()
            except Exception:
                break
            if parent is None:
                break
            node = parent

        # No WindowControl ancestor: either an application whose top level is
        # not a window, or a climb that ran out of depth. The foreground window
        # is a better answer than a node halfway up somebody's scaffolding.
        return auto.GetForegroundControl() or control

    def _named_window(self, auto: Any, name: str) -> Any:
        """Find a top-level window by name, for ``--app``.

        Windows first, then Panes: most applications present a WindowControl at
        the top level, but a few (and the Explorer shell in some views) present
        a Pane, and a name that matches nothing is worth more than a window that
        was not asked for.
        """
        for attribute in ("WindowControl", "PaneControl"):
            control = getattr(auto, attribute, None)
            if control is None:
                continue
            try:
                window = control(searchDepth=1, SubName=name)
                if window.Exists(maxSearchSeconds=2):
                    return window
            except Exception:
                continue
        raise PerceptionUnavailable(
            f"no top-level window matches {name!r}. "
            "`victor uia --apps` lists what can be targeted."
        )

    def window_info(self, window: Any) -> tuple[str, str, Rect]:
        title = getattr(window, "Name", "") or "<untitled>"
        process = "?"
        with contextlib.suppress(Exception):
            process = _process_name(int(window.ProcessId)) or str(window.ProcessId)
        return title, process, _rect_of(window)

    def children(self, node: Any) -> list[Any]:
        try:
            return list(node.GetChildren())
        except Exception:
            return []

    def describe(self, node: Any) -> tuple[str, str, Rect, bool, bool, str, str]:
        control_type = str(getattr(node, "ControlTypeName", "") or "").removesuffix("Control")
        name = str(getattr(node, "Name", "") or "")
        enabled = bool(getattr(node, "IsEnabled", True))
        focused = bool(getattr(node, "HasKeyboardFocus", False))
        automation_id = str(getattr(node, "AutomationId", "") or "")
        value = ""
        try:
            pattern = node.GetValuePattern()
            value = str(pattern.Value or "")
        except Exception:
            value = ""
        return control_type, name, _rect_of(node), enabled, focused, value, automation_id


def _control_type(node: Any) -> str:
    """A node's UIA control type, tolerating a node that will not answer."""
    try:
        return str(getattr(node, "ControlTypeName", "") or "")
    except Exception:
        return ""


def _process_name(pid: int) -> str:
    """``explorer.exe`` rather than ``7412``.

    A bare PID tells the model and the user nothing, and macOS reports an
    application name here - so the two backends producing the same shape of
    answer is worth a short ctypes call.
    """
    if platform.system() != "Windows" or pid <= 0:
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000  # noqa: N806
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(260)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            ):
                return ""
            return buffer.value.rsplit("\\", 1)[-1]
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return ""


def list_applications() -> list[str]:
    """Names of windows that ``--app`` can target, for the running platform."""
    system = platform.system()
    if system == "Darwin":
        from .ax import mac_applications

        return mac_applications()
    if system == "Windows":
        return _windows_windows()
    return []


def _windows_windows() -> list[str]:
    """Top-level window titles, deduplicated and in on-screen order."""
    try:
        import uiautomation
    except ImportError:
        return []
    try:
        children = uiautomation.GetRootControl().GetChildren()
    except Exception:
        return []

    names: list[str] = []
    for child in children:
        if _control_type(child) not in ("WindowControl", "PaneControl"):
            continue
        name = str(getattr(child, "Name", "") or "").strip()
        # The desktop itself is a named Pane and is not something to target.
        if name and name not in names and not name.startswith("Desktop "):
            names.append(name)
    return names


def _rect_of(node: Any) -> Rect:
    try:
        r = node.BoundingRectangle
        return Rect(int(r.left), int(r.top), int(r.right), int(r.bottom))
    except Exception:
        return Rect(0, 0, 0, 0)


@dataclass
class FakeNode:
    """A node in a literal tree. The reason this stack is testable anywhere."""

    control_type: str
    name: str = ""
    rect: Rect = Rect(0, 0, 100, 30)
    enabled: bool = True
    focused: bool = False
    value: str = ""
    automation_id: str = ""
    children: list[FakeNode] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


class FakeBackend:
    """Serves a literal tree. Used by tests and by ``victor uia --demo``."""

    name = "fake"

    def __init__(self, root: FakeNode, *, title: str = "Fake Window", process: str = "fake.exe"):
        self.root = root
        self.title = title
        self.process = process

    def available(self) -> tuple[bool, str]:
        return True, "fake backend"

    def focused_window(self) -> Any | None:
        return self.root

    def window_info(self, window: Any) -> tuple[str, str, Rect]:
        return self.title, self.process, window.rect

    def children(self, node: Any) -> list[Any]:
        return list(node.children or [])

    def describe(self, node: Any) -> tuple[str, str, Rect, bool, bool, str, str]:
        return (
            node.control_type,
            node.name,
            node.rect,
            node.enabled,
            node.focused,
            node.value,
            node.automation_id,
        )


class TreeReader:
    """Walks a backend's tree into a :class:`Snapshot`."""

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        limits: WalkLimits | None = None,
        cache_ttl: float = 1.0,
        app: str | None = None,
    ) -> None:
        self.backend = backend or select_backend(app=app)
        self.limits = limits or WalkLimits()
        self.cache_ttl = cache_ttl
        self._cache: tuple[Any, float, Snapshot] | None = None

    def available(self) -> tuple[bool, str]:
        return self.backend.available()

    def snapshot(self, *, refresh: bool = False) -> Snapshot:
        """Read the focused window, using the cache when it is still warm."""
        window = self.backend.focused_window()
        if window is None:
            raise PerceptionUnavailable("no focused window")

        if not refresh and self._cache is not None:
            cached_window, taken_at, snapshot = self._cache
            if cached_window is window and (time.monotonic() - taken_at) < self.cache_ttl:
                return snapshot

        snapshot = self._walk(window)
        self._cache = (window, time.monotonic(), snapshot)
        return snapshot

    def invalidate(self) -> None:
        """Drop the cache. Call on focus change or after an action."""
        self._cache = None

    def _walk(self, window: Any) -> Snapshot:
        title, process, rect = self.backend.window_info(window)
        started = time.perf_counter()
        deadline = started + self.limits.max_seconds

        elements: list[Element] = []
        # Real trees contain the same control twice: Chrome reports its bookmark
        # bar under two parents, and UIA does the equivalent. Two identical rows
        # with different indices waste the context budget and give the model a
        # choice that has no right answer.
        seen: set[tuple[str, str, int, int, int, int]] = set()
        truncated = False
        # Breadth-first: the controls a user would reach for sit near the top of
        # the tree, so if the walk is cut short the useful ones are already in.
        frontier: list[tuple[Any, int]] = [(window, 0)]

        while frontier:
            node, depth = frontier.pop(0)

            if len(elements) >= self.limits.max_elements:
                truncated = True
                break
            if time.perf_counter() > deadline:
                truncated = True
                break

            control_type, name, node_rect, enabled, focused, value, automation_id = (
                self.backend.describe(node)
            )

            fingerprint = (
                control_type,
                name,
                node_rect.left,
                node_rect.top,
                node_rect.right,
                node_rect.bottom,
            )
            if (
                depth
                and fingerprint not in seen
                and is_interesting(control_type, name, node_rect)
            ):
                seen.add(fingerprint)
                elements.append(
                    Element(
                        index=len(elements),
                        control_type=control_type,
                        name=name,
                        rect=node_rect,
                        enabled=enabled,
                        focused=focused,
                        value=value,
                        automation_id=automation_id,
                        depth=depth,
                        handle=node,
                    )
                )

            if depth < self.limits.max_depth:
                frontier.extend((child, depth + 1) for child in self.backend.children(node))

        # Every element is filtered on having a visible rectangle, so a window
        # the OS will not measure produces an empty list that looks exactly like
        # a window with nothing in it. They are not the same problem: one is
        # "there is nothing to click", the other is "the window is minimised, on
        # another desktop, or off-screen". Distinguishing them here is cheap and
        # saves everyone downstream from guessing.
        note = ""
        if not elements and rect.empty:
            note = (
                f"{title} reports no position or size, so none of its controls "
                "could be measured. It is probably minimised, on another desktop, "
                "or positioned off-screen. Bring it into view and read again."
            )

        return Snapshot(
            window_title=title,
            process=process,
            elements=tuple(elements),
            rect=rect,
            truncated=truncated,
            duration_ms=(time.perf_counter() - started) * 1000,
            backend=self.backend.name,
            note=note,
        )


def select_backend(*, app: str | None = None) -> Backend:
    """The right tree reader for this operating system.

    The plan targeted Windows only, and UI Automation is a Windows API - but the
    information Victor needs is exposed by every desktop accessibility layer,
    and the Backend protocol was put here so a second one would be an addition
    rather than a rewrite. macOS is that second one.

    An unsupported platform gets a backend that fails with a clear reason when
    used, rather than an import error at start-up: the rest of Victor - voice,
    shell, git, safety, memory - works fine without screen perception.
    """
    system = platform.system()
    if system == "Windows":
        return UIABackend(app_name=app)
    if system == "Darwin":
        from .ax import AXBackend

        return AXBackend(app_name=app)
    return UnsupportedBackend(system)


class UnsupportedBackend:
    """Placeholder for platforms with no perception backend."""

    name = "unsupported"

    def __init__(self, system: str) -> None:
        self.system = system

    def available(self) -> tuple[bool, str]:
        return False, (
            f"no accessibility backend for {self.system}. "
            "Windows uses UI Automation and macOS uses the Accessibility API; "
            "everything else in Victor works here, just not screen perception."
        )

    def _fail(self) -> Any:
        raise PerceptionUnavailable(self.available()[1])

    def focused_window(self) -> Any | None:
        return self._fail()

    def window_info(self, window: Any) -> tuple[str, str, Rect]:
        return self._fail()

    def children(self, node: Any) -> list[Any]:
        return self._fail()

    def describe(self, node: Any) -> tuple[str, str, Rect, bool, bool, str, str]:
        return self._fail()


def demo_tree() -> FakeNode:
    """A small, realistic tree - the README's example, made runnable."""
    return FakeNode(
        "Window",
        "Inbox - Mail",
        Rect(0, 0, 1440, 900),
        children=[
            FakeNode(
                "ToolBar",
                "Main",
                Rect(0, 40, 1440, 120),
                children=[
                    FakeNode("Button", "Compose", Rect(24, 180, 140, 220)),
                    FakeNode("Edit", "Search mail", Rect(300, 60, 900, 100)),
                    FakeNode("Button", "Settings", Rect(1400, 60, 1440, 100)),
                    FakeNode("Button", "Archive", Rect(150, 180, 260, 220), enabled=False),
                ],
            ),
            FakeNode(
                "List",
                "Messages",
                Rect(0, 130, 1440, 900),
                children=[
                    FakeNode("ListItem", "Invoice for March", Rect(0, 130, 1440, 190)),
                    FakeNode("ListItem", "Re: deployment", Rect(0, 190, 1440, 250)),
                ],
            ),
            FakeNode("Pane", "", Rect(0, 0, 0, 0)),  # filtered: no name, no size
        ],
    )
