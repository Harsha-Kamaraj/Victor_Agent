"""Is anyone actually looking at this screen?

A locked screen is not a broken accessibility layer, but it looks exactly like
one. macOS keeps answering questions about the window tree while the screen is
locked and simply refuses to report geometry, so every rectangle comes back
empty, every element is filtered out for having no visible area, and Victor
reports a window with nothing in it. Windows does something similar: input goes
to the secure desktop, so keystrokes land nowhere.

Both symptoms send you looking at the tree walk, which is fine. This module
exists so the answer arrives in one line instead.

Cheap enough to call before every session - one system query, no allocation -
and deliberately fail-open: if the lock state cannot be determined, say the
screen is usable rather than blocking work on a guess.
"""

from __future__ import annotations

import platform


def session_locked() -> tuple[bool, str]:
    """``(locked, human explanation)``. Unknown states report unlocked."""
    system = platform.system()
    if system == "Darwin":
        return _mac_locked()
    if system == "Windows":
        return _windows_locked()
    return False, ""


def _mac_locked() -> tuple[bool, str]:
    try:
        import Quartz
    except ImportError:
        return False, ""
    try:
        session = Quartz.CGSessionCopyCurrentDictionary()
    except Exception:
        return False, ""
    if not session:
        # No window server session at all: a launch daemon, or ssh without a
        # console. Either way there is no screen to drive.
        return True, (
            "this process has no window server session, so there is no screen to "
            "read or drive. Run Victor from a logged-in desktop session."
        )
    if session.get("CGSSessionScreenIsLocked"):
        return True, (
            "the screen is locked. macOS stops reporting window positions while "
            "it is, so every control looks like it has no visible area. Unlock "
            "the screen and try again."
        )
    if not session.get("kCGSSessionOnConsoleKey", True):
        return True, (
            "this login session is not the one on screen - another user is at "
            "the console. Victor can only drive the session it is displayed in."
        )
    return False, ""


def _windows_locked() -> tuple[bool, str]:
    """Ask whether the input desktop can be opened.

    ``OpenInputDesktop`` returns NULL when the workstation is locked or a secure
    desktop (the UAC prompt, the login screen) is in front, because those run on
    a different desktop that an ordinary process may not touch. That is the
    documented way to detect the state, and it is also exactly the condition
    that matters: if this fails, synthesised input would go nowhere.
    """
    try:
        import ctypes
    except ImportError:  # pragma: no cover - ctypes is always present
        return False, ""
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        DESKTOP_SWITCHDESKTOP = 0x0100
        handle = user32.OpenInputDesktop(0, False, DESKTOP_SWITCHDESKTOP)
        if not handle:
            return True, (
                "the workstation is locked, or a secure desktop such as a UAC "
                "prompt is in front. Input cannot reach the normal desktop from "
                "here. Unlock the screen or dismiss the prompt, then try again."
            )
        user32.CloseDesktop(handle)
    except Exception:
        # Never block work because the probe itself failed.
        return False, ""
    return False, ""
