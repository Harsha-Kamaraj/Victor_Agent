"""Screen perception and desktop control.

Perception (P4) is read-only by construction and lives in :mod:`elements`,
:mod:`uia`, :mod:`ax`, :mod:`capture` and :mod:`vision`. Actuation (P5) lives in
:mod:`actions` and :mod:`keys`, and every path into it is gated behind the P3
safety layer. Keeping the two apart is why perception could be built and tested
out of order, and why reading a screen still costs nothing and asks nothing.
"""

from .actions import (
    ActionRefused,
    ActionResult,
    ActuationUnavailable,
    Actuator,
    Desktop,
    FakeActuator,
    MacActuator,
    Resolution,
    UnsupportedActuator,
    WindowsActuator,
    normalise_label,
    select_actuator,
)
from .ax import AXBackend, list_applications
from .capture import (
    CaptureUnavailable,
    ScreenCapture,
    Screenshot,
    hamming,
    perceptual_hash,
)
from .elements import Element, Rect, Snapshot, is_interesting
from .uia import (
    Backend,
    FakeBackend,
    FakeNode,
    PerceptionUnavailable,
    TreeReader,
    UIABackend,
    UnsupportedBackend,
    WalkLimits,
    demo_tree,
    select_backend,
)
from .vision import VisionAnswer, VisionClient, VisionUnavailable, annotate

__all__ = [
    "AXBackend",
    "ActionRefused",
    "ActionResult",
    "ActuationUnavailable",
    "Actuator",
    "Backend",
    "CaptureUnavailable",
    "Desktop",
    "Element",
    "FakeActuator",
    "MacActuator",
    "Resolution",
    "UnsupportedActuator",
    "WindowsActuator",
    "normalise_label",
    "select_actuator",
    "FakeBackend",
    "FakeNode",
    "PerceptionUnavailable",
    "Rect",
    "ScreenCapture",
    "Screenshot",
    "Snapshot",
    "TreeReader",
    "UIABackend",
    "UnsupportedBackend",
    "VisionAnswer",
    "VisionClient",
    "VisionUnavailable",
    "WalkLimits",
    "annotate",
    "demo_tree",
    "hamming",
    "is_interesting",
    "list_applications",
    "perceptual_hash",
    "select_backend",
]
