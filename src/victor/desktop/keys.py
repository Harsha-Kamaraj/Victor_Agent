"""Naming keys once, for both platforms.

An agent that wants to save a file should be able to say ``mod+s`` and have it
mean ⌘S on macOS and Ctrl+S on Windows. Without that, the model has to know
which machine it is on before it can press a key, and every prompt grows a
platform conditional.

So this module defines one vocabulary and two code tables. ``mod`` is the
platform's *primary shortcut modifier*; ``cmd`` is its *command/super* key.
They are the same thing on macOS and different on Windows, which is exactly the
distinction that makes ``mod+s`` portable and ``cmd+r`` (Win+R, the Run dialog)
still expressible.

Two naming decisions are deliberate and are documented for the model in the
tool description:

* ``backspace`` always deletes to the left, and ``delete`` always deletes to the
  right - on *both* platforms. macOS labels its ⌫ key "delete", so the obvious
  alternative would have been to follow the keycap; that would make the same
  chord do different things on the two machines, which is worse than disagreeing
  with a keycap.
* Everything is lower case and joined by ``+``. There is one spelling of a
  chord, so a test for one is a test for all of them.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from ..errors import VictorError


class UnknownKey(VictorError):
    """A chord named a key this vocabulary does not have."""

    exit_code = 2


#: Modifier spellings a model might reasonably produce, folded onto four names.
#: ``mod`` is left out here because it is not a modifier - it is a request to
#: pick one, resolved per platform by :func:`resolve`.
MODIFIER_ALIASES: dict[str, str] = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "ctl": "ctrl",
    "alt": "alt",
    "option": "alt",
    "opt": "alt",
    "shift": "shift",
    "cmd": "cmd",
    "command": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "win": "cmd",
    "windows": "cmd",
}

#: Key-name spellings folded onto canonical names.
KEY_ALIASES: dict[str, str] = {
    "enter": "return",
    "ret": "return",
    "esc": "escape",
    "del": "delete",
    "bksp": "backspace",
    "bs": "backspace",
    "spacebar": "space",
    "pgup": "pageup",
    "page_up": "pageup",
    "pgdn": "pagedown",
    "pagedn": "pagedown",
    "page_down": "pagedown",
    "ins": "insert",
    "caps": "capslock",
}

#: macOS virtual key codes (ANSI layout), for ``CGEventCreateKeyboardEvent``.
#: Text is typed as Unicode rather than through this table - see
#: :meth:`victor.desktop.actions.MacActuator.type_text` - so this only needs the
#: keys that appear in shortcuts.
MAC_KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25,
    "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33,
    "i": 34, "p": 35, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42,
    ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "`": 50,
    "return": 36, "tab": 48, "space": 49, "backspace": 51, "escape": 53,
    "delete": 117, "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126, "help": 114,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}

#: macOS modifier flag bits (``CGEventFlags``).
MAC_MODIFIER_FLAGS: dict[str, int] = {
    "shift": 1 << 17,
    "ctrl": 1 << 18,
    "alt": 1 << 19,
    "cmd": 1 << 20,
}

#: Windows virtual key codes.
WINDOWS_KEYCODES: dict[str, int] = {
    "backspace": 0x08, "tab": 0x09, "return": 0x0D, "escape": 0x1B,
    "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "insert": 0x2D, "delete": 0x2E, "capslock": 0x14,
    ";": 0xBA, "=": 0xBB, ",": 0xBC, "-": 0xBD, ".": 0xBE, "/": 0xBF,
    "`": 0xC0, "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}
WINDOWS_KEYCODES.update({chr(c): c for c in range(0x30, 0x3A)})  # 0-9
WINDOWS_KEYCODES.update({chr(c + 32): c for c in range(0x41, 0x5B)})  # a-z
WINDOWS_KEYCODES.update({f"f{n}": 0x6F + n for n in range(1, 13)})  # F1-F12

#: Windows modifier virtual key codes.
WINDOWS_MODIFIER_CODES: dict[str, int] = {
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "cmd": 0x5B,  # Left Windows key
}


@dataclass(frozen=True, slots=True)
class Chord:
    """One key press, with modifiers held down around it."""

    key: str
    modifiers: tuple[str, ...] = ()

    def __str__(self) -> str:
        return "+".join([*self.modifiers, self.key])

    @property
    def destructive_hint(self) -> bool:
        """Whether this chord is one of the few that discard work.

        Used by the safety classifier. Deliberately narrow: the point is to
        catch "empty the trash" and "delete without confirmation", not to make
        every shortcut a question.
        """
        mods = set(self.modifiers)
        return self.key in {"delete", "backspace"} and bool(mods & {"cmd", "ctrl", "shift"})


def _canonical_modifier(token: str, system: str) -> str:
    if token == "mod":
        # The key people mean when they say "the shortcut key".
        return "cmd" if system == "Darwin" else "ctrl"
    try:
        return MODIFIER_ALIASES[token]
    except KeyError:
        raise UnknownKey(f"{token!r} is not a modifier") from None


def parse(text: str, *, system: str | None = None) -> Chord:
    """Turn ``"mod+shift+p"`` into a :class:`Chord` for this platform.

    Raises :class:`UnknownKey` rather than pressing something approximate. A
    chord that silently loses a modifier is how "select all and delete" becomes
    "delete".
    """
    system = system or platform.system()
    raw = text.strip().lower()
    if not raw:
        raise UnknownKey("no keys given")

    # A bare "+" is the plus key, and "ctrl++" is ctrl plus that key. Splitting
    # naively would produce an empty final token, so peel the trailing case off
    # before splitting on the separator.
    if raw.endswith("++"):
        head, final = raw[:-2], "+"
    elif raw == "+":
        head, final = "", "+"
    else:
        parts = raw.split("+")
        head, final = "+".join(parts[:-1]), parts[-1]

    tokens = [t for t in head.split("+") if t] if head else []
    modifiers: list[str] = []
    for token in tokens:
        canonical = _canonical_modifier(token, system)
        if canonical not in modifiers:
            modifiers.append(canonical)

    key = KEY_ALIASES.get(final, final)
    if not key:
        raise UnknownKey(f"{text!r} names modifiers but no key")
    if key in MODIFIER_ALIASES or key == "mod":
        raise UnknownKey(f"{text!r} ends in a modifier; name the key to press")

    # Order modifiers canonically so the same chord has one representation, and
    # so the journal and the tests do not depend on how the model spelled it.
    order = {"ctrl": 0, "alt": 1, "shift": 2, "cmd": 3}
    return Chord(key=key, modifiers=tuple(sorted(modifiers, key=lambda m: order[m])))


def parse_sequence(text: str, *, system: str | None = None) -> tuple[Chord, ...]:
    """Parse space-separated chords: ``"mod+a delete"`` is two presses."""
    chords = tuple(parse(part, system=system) for part in text.split() if part)
    if not chords:
        raise UnknownKey("no keys given")
    return chords


def mac_keycode(chord: Chord) -> tuple[int, int]:
    """``(virtual keycode, modifier flags)`` for a macOS event."""
    try:
        code = MAC_KEYCODES[chord.key]
    except KeyError:
        raise UnknownKey(
            f"macOS has no keycode for {chord.key!r} in Victor's table. "
            "Use type_text for ordinary characters."
        ) from None
    flags = 0
    for modifier in chord.modifiers:
        flags |= MAC_MODIFIER_FLAGS[modifier]
    return code, flags


def windows_keycodes(chord: Chord) -> tuple[list[int], int]:
    """``(modifier key codes, key code)`` for a Windows event."""
    try:
        code = WINDOWS_KEYCODES[chord.key]
    except KeyError:
        raise UnknownKey(
            f"Windows has no virtual key for {chord.key!r} in Victor's table. "
            "Use type_text for ordinary characters."
        ) from None
    return [WINDOWS_MODIFIER_CODES[m] for m in chord.modifiers], code


def known_keys() -> list[str]:
    """Every key name that works on both platforms. Shown by ``victor keys``."""
    shared = set(MAC_KEYCODES) & set(WINDOWS_KEYCODES)
    return sorted(shared)
