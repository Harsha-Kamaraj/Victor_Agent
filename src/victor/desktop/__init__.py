"""Screen perception: read the accessibility tree, capture only when needed.

Read-only by construction. Nothing in this package moves a mouse or presses a
key - that is P5, and it is gated behind the P3 safety layer. Keeping the two
apart is why perception could be built and tested out of order.
"""

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
    "Backend",
    "CaptureUnavailable",
    "Element",
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
