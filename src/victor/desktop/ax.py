"""Reading the macOS accessibility tree.

The sibling of :mod:`victor.desktop.uia`. Same idea, different OS API: macOS
exposes the same information through ``AXUIElement`` that Windows exposes
through UI Automation, so the whole design above the backend - indexing,
filtering, bounding, caching, Set-of-Mark prompting - is unchanged.

This is why P4 put a :class:`~victor.desktop.uia.Backend` protocol between the
tree walk and the OS. Adding macOS support is a new backend, not a port.

Two things differ from Windows and are worth knowing:

**Permission is explicit.** macOS refuses to expose other applications' trees
until the user grants Accessibility permission to the terminal or app running
Victor. There is no way to ask for it programmatically that does not lie about
what it needs, so :meth:`AXBackend.available` reports the exact System Settings
pane instead.

**"Frontmost" is not always what you want.** The literal frontmost process is
sometimes a helper with no windows (``WindowManager`` when nothing is focused),
so ``--app`` targets an application by name. On Windows that is rarely needed;
here it is the difference between a useful dump and an empty one.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

from .elements import Rect
from .uia import PerceptionUnavailable

#: AX roles mapped onto the control-type vocabulary the rest of Victor uses.
#: Keeping one vocabulary means the agent's prompt, the filter and the tests do
#: not need to know which OS produced a snapshot.
ROLE_MAP: dict[str, str] = {
    "AXButton": "Button",
    "AXMenuButton": "SplitButton",
    "AXPopUpButton": "ComboBox",
    "AXComboBox": "ComboBox",
    "AXCheckBox": "CheckBox",
    "AXRadioButton": "RadioButton",
    "AXTextField": "Edit",
    "AXTextArea": "Edit",
    "AXSearchField": "Edit",
    "AXSecureTextField": "Edit",
    "AXStaticText": "Text",
    "AXLink": "Hyperlink",
    "AXMenuItem": "MenuItem",
    "AXMenuBarItem": "MenuItem",
    "AXRow": "ListItem",
    "AXCell": "ListItem",
    "AXList": "List",
    "AXTable": "Table",
    "AXOutline": "Tree",
    "AXOutlineRow": "TreeItem",
    "AXToolbar": "ToolBar",
    "AXTabGroup": "TabItem",
    "AXSlider": "Slider",
    "AXIncrementor": "Slider",
    "AXImage": "Image",
    "AXWindow": "Window",
    "AXSheet": "Window",
    "AXGroup": "Group",
    "AXScrollArea": "Group",
    "AXSplitGroup": "Group",
    "AXWebArea": "Document",
    "AXTextGroup": "Group",
}


#: Subroles that name an otherwise-untitled control. The window buttons are the
#: common case: they have no AXTitle at all, but every user calls them these.
SUBROLE_NAMES: dict[str, str] = {
    "AXCloseButton": "Close",
    "AXMinimizeButton": "Minimise",
    "AXZoomButton": "Zoom",
    "AXFullScreenButton": "Full screen",
    "AXToolbarButton": "Toolbar",
    "AXSearchField": "Search",
    "AXSecureTextField": "Password",
    "AXSortButton": "Sort",
}


def _map_role(role: str) -> str:
    """Translate an AX role, falling back to a readable form of the raw name."""
    if role in ROLE_MAP:
        return ROLE_MAP[role]
    return role.removeprefix("AX") or "Unknown"


@dataclass(frozen=True, slots=True)
class _Frameworks:
    """The PyObjC symbols this backend needs, resolved once."""

    workspace: Any
    services: Any


class AXBackend:
    """macOS accessibility tree via PyObjC."""

    name = "ax"

    def __init__(self, app_name: str | None = None) -> None:
        self.app_name = app_name
        self._fw: _Frameworks | None = None

    # -- loading -----------------------------------------------------------

    def _load(self) -> _Frameworks:
        if self._fw is not None:
            return self._fw
        if platform.system() != "Darwin":
            raise PerceptionUnavailable(
                f"the accessibility backend is macOS-only; this is {platform.system()}"
            )
        try:
            import ApplicationServices
            from AppKit import NSWorkspace
        except ImportError as exc:
            raise PerceptionUnavailable(
                f"PyObjC is not installed ({exc}). pip install -e '.[desktop]'"
            ) from exc
        self._fw = _Frameworks(workspace=NSWorkspace, services=ApplicationServices)
        return self._fw

    def available(self) -> tuple[bool, str]:
        try:
            fw = self._load()
        except PerceptionUnavailable as exc:
            return False, str(exc)

        if not fw.services.AXIsProcessTrusted():
            return False, (
                "macOS has not granted Accessibility permission to this program. "
                "Grant it in System Settings > Privacy & Security > Accessibility, "
                "then run this again."
            )
        return True, "macOS Accessibility ready"

    # -- attributes --------------------------------------------------------

    def _attr(self, element: Any, attribute: str) -> Any:
        """Read one AX attribute, or ``None`` if it is unsupported."""
        fw = self._load()
        try:
            err, value = fw.services.AXUIElementCopyAttributeValue(element, attribute, None)
        except Exception:
            return None
        return value if err == 0 else None

    def _rect(self, element: Any) -> Rect:
        """Position and size, unwrapped from their AXValue boxes."""
        fw = self._load()
        position = self._attr(element, fw.services.kAXPositionAttribute)
        size = self._attr(element, fw.services.kAXSizeAttribute)
        if position is None or size is None:
            return Rect(0, 0, 0, 0)

        try:
            ok_p, point = fw.services.AXValueGetValue(
                position, fw.services.kAXValueCGPointType, None
            )
            ok_s, extent = fw.services.AXValueGetValue(
                size, fw.services.kAXValueCGSizeType, None
            )
        except Exception:
            return Rect(0, 0, 0, 0)
        if not (ok_p and ok_s):
            return Rect(0, 0, 0, 0)

        left, top = int(point.x), int(point.y)
        return Rect(left, top, left + int(extent.width), top + int(extent.height))

    # -- the Backend protocol ---------------------------------------------

    def focused_window(self) -> Any | None:
        fw = self._load()
        ok, detail = self.available()
        if not ok:
            raise PerceptionUnavailable(detail)

        target = self._target_application(fw)
        if target is None:
            raise PerceptionUnavailable(
                f"no running application named {self.app_name!r}"
                if self.app_name
                else "no frontmost application"
            )

        ax_app = fw.services.AXUIElementCreateApplication(target.processIdentifier())
        window = self._attr(ax_app, fw.services.kAXFocusedWindowAttribute)
        if window is not None:
            return window

        # A helper process can be frontmost with no focused window; fall back to
        # its first real one before giving up.
        windows = self._attr(ax_app, fw.services.kAXWindowsAttribute)
        if windows and len(windows):
            return windows[0]

        raise PerceptionUnavailable(
            f"{target.localizedName()} has no accessible windows. "
            "Focus an application window, or pass --app to name one."
        )

    def _target_application(self, fw: _Frameworks) -> Any | None:
        workspace = fw.workspace.sharedWorkspace()
        if self.app_name is None:
            return workspace.frontmostApplication()

        wanted = self.app_name.strip().lower()
        for app in workspace.runningApplications():
            name = app.localizedName() or ""
            if name.lower() == wanted or wanted in name.lower():
                return app
        return None

    def window_info(self, window: Any) -> tuple[str, str, Rect]:
        fw = self._load()
        title = self._attr(window, fw.services.kAXTitleAttribute) or "<untitled>"
        process = self.app_name or ""
        if not process:
            app = fw.workspace.sharedWorkspace().frontmostApplication()
            process = app.localizedName() if app else "?"
        return str(title), str(process), self._rect(window)

    def children(self, node: Any) -> list[Any]:
        fw = self._load()
        kids = self._attr(node, fw.services.kAXChildrenAttribute)
        return list(kids) if kids else []

    def describe(self, node: Any) -> tuple[str, str, Rect, bool, bool, str, str]:
        fw = self._load()
        s = fw.services
        role = str(self._attr(node, s.kAXRoleAttribute) or "")
        control_type = _map_role(role)

        raw_value = self._attr(node, s.kAXValueAttribute)
        value = "" if raw_value is None else str(raw_value)

        # AXTitle is often empty on macOS where Windows would have a Name, so
        # fall through the other label-bearing attributes rather than showing
        # the agent an unnamed button it cannot refer to.
        #
        # Subrole comes before description deliberately. A window button has no
        # title, a canonical subrole ("Close"), and sometimes a whole sentence
        # of description - Finder's zoom button describes itself as "this button
        # also has an action to zoom the window". The short canonical name is
        # what a person would say and what the model should match on.
        name = ""
        subrole = str(self._attr(node, getattr(s, "kAXSubroleAttribute", "")) or "")
        for candidate in (
            self._attr(node, s.kAXTitleAttribute),
            SUBROLE_NAMES.get(subrole, ""),
            self._attr(node, s.kAXDescriptionAttribute),
            self._attr(node, getattr(s, "kAXHelpAttribute", s.kAXDescriptionAttribute)),
        ):
            if candidate:
                name = str(candidate)
                break
        if not name and control_type == "Text":
            name = value  # static text carries its label in the value

        enabled = self._attr(node, s.kAXEnabledAttribute)
        focused = self._attr(node, s.kAXFocusedAttribute)
        identifier = self._attr(node, getattr(s, "kAXIdentifierAttribute", s.kAXRoleAttribute))

        return (
            control_type,
            name,
            self._rect(node),
            True if enabled is None else bool(enabled),
            bool(focused),
            value if value != name else "",
            str(identifier or "") if identifier != role else "",
        )


def list_applications() -> list[str]:
    """Names of running applications that have a user interface."""
    if platform.system() != "Darwin":
        return []
    try:
        from AppKit import NSWorkspace
    except ImportError:
        return []
    return [
        app.localizedName()
        for app in NSWorkspace.sharedWorkspace().runningApplications()
        # activationPolicy 0 is NSApplicationActivationPolicyRegular: apps with
        # a Dock icon and windows, rather than background agents.
        if app.activationPolicy() == 0 and app.localizedName()
    ]
