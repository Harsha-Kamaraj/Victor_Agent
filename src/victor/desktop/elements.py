"""What the agent sees on screen.

The project's central bet lives in this type. Most computer-use agents ask a
vision model "where should I click?" and get back coordinates that are wrong by
enough to hit the neighbouring control. Victor asks the operating system what is
on screen and gets exact names and rectangles, locally, for free - then hands
the model a numbered list and lets it pick an integer.

An :class:`Element` is therefore addressed by **index**, not position. The model
never sees a coordinate and cannot invent one; P5 resolves the index back to a
handle the OS already gave us.

Nothing here is Windows-specific. The UIA backend fills these in on Windows;
tests fill them in from a literal tree, which is what makes the whole perception
and actuation stack testable on a machine that has no UI Automation at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Control types worth showing an agent. A window contains hundreds of panes,
#: groups and static text; listing them all buries the six things you can act
#: on and wastes the context budget that the model needs for reasoning.
INTERACTABLE = frozenset(
    {
        "Button",
        "CheckBox",
        "ComboBox",
        "Edit",
        "Document",
        "Hyperlink",
        "ListItem",
        "MenuItem",
        "RadioButton",
        "Slider",
        "SplitButton",
        "TabItem",
        "Tree",
        "TreeItem",
    }
)

#: Types kept only when they carry a name, because a labelled one is often the
#: only handle on an area of the screen.
CONTEXTUAL = frozenset({"Text", "Image", "Group", "List", "Table", "ToolBar", "Window"})


@dataclass(frozen=True, slots=True)
class Rect:
    """A bounding box in screen coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centre(self) -> tuple[int, int]:
        """Where P5 would click. Derived from the OS rectangle, never guessed."""
        return (self.left + self.width // 2, self.top + self.height // 2)

    @property
    def empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def __str__(self) -> str:
        return f"({self.left},{self.top})-({self.right},{self.bottom})"


@dataclass(frozen=True, slots=True)
class Element:
    """One addressable thing on screen."""

    index: int
    control_type: str
    name: str
    rect: Rect
    enabled: bool = True
    focused: bool = False
    value: str = ""
    automation_id: str = ""
    depth: int = 0
    handle: Any = field(default=None, compare=False, repr=False)
    """The backend's native object. P5 acts through this, never through pixels."""

    @property
    def actionable(self) -> bool:
        return self.enabled and not self.rect.empty

    @property
    def label(self) -> str:
        """The best available human name for this element."""
        return self.name or self.value or self.automation_id or f"<{self.control_type}>"

    def render(self) -> str:
        """The line the model sees.

        Deliberately compact: this list is sent on every perception step, and
        at ~8,000 tokens a minute the difference between 60 and 200 characters
        per element decides whether a busy window fits at all.
        """
        parts = [f"[{self.index}]", f"{self.control_type:<12}", f'"{self.truncated_label}"']
        line = " ".join(parts)
        flags = []
        if not self.enabled:
            flags.append("disabled")
        if self.focused:
            flags.append("focused")
        if self.value and self.value != self.name:
            flags.append(f"value={self.value[:30]!r}")
        suffix = f"  {' '.join(flags)}" if flags else ""
        return f"{line:<48} {self.rect}{suffix}"

    @property
    def truncated_label(self) -> str:
        label = " ".join(self.label.split())
        return label if len(label) <= 60 else label[:57] + "..."


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The interactable elements of one window at one moment."""

    window_title: str
    process: str
    elements: tuple[Element, ...]
    rect: Rect | None = None
    truncated: bool = False
    duration_ms: float = 0.0
    backend: str = ""
    note: str = ""
    """Why this snapshot is emptier than expected, when the walk can tell.

    An empty list has two very different causes - a window genuinely without
    controls, and a window Victor could not measure - and they need different
    responses from both the user and the model. Reporting "0 elements" for both
    sends everyone looking in the wrong place."""

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def by_index(self, index: int) -> Element | None:
        return next((e for e in self.elements if e.index == index), None)

    def find(self, text: str, *, control_type: str | None = None) -> list[Element]:
        """Elements whose label contains ``text``, case-insensitively."""
        needle = text.strip().lower()
        return [
            e
            for e in self.elements
            if needle in e.label.lower()
            and (control_type is None or e.control_type == control_type)
        ]

    def render(self, limit: int | None = None) -> str:
        """The whole list as the model sees it."""
        shown = self.elements if limit is None else self.elements[:limit]
        header = f"{self.window_title} ({self.process}) - {len(self.elements)} elements"
        lines = [header, "-" * len(header)]
        lines.extend(e.render() for e in shown)
        if limit is not None and len(self.elements) > limit:
            lines.append(f"... {len(self.elements) - limit} more not shown")
        if self.truncated:
            lines.append("... tree walk hit its limit; some elements are missing")
        if self.note:
            lines.append(f"... {self.note}")
        return "\n".join(lines)

    @property
    def empty(self) -> bool:
        return not self.elements


def is_interesting(control_type: str, name: str, rect: Rect) -> bool:
    """Should this element be shown to the agent?

    Filtering happens here rather than in the backend so the rule is one
    testable function instead of a condition buried in a tree walk.
    """
    if rect.empty:
        return False
    if control_type in INTERACTABLE:
        return True
    # A labelled container is often the only handle on a region; an unlabelled
    # one is scaffolding.
    return control_type in CONTEXTUAL and bool(name.strip())
