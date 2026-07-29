"""Screen capture, sized and cached for a scarce vision budget.

Three decisions, all driven by the same fact: vision is ~250 requests a day.

**Downscale before sending.** A 4K screenshot costs several times the tokens of
a 768px one and tells the model nothing extra about which button to press. The
long edge is capped and the aspect ratio preserved.

**Hash before sending.** The plan calls for a perceptual-hash cache so "an
unchanged screen never re-bills a VLM call". An agent that looks twice while
deciding is the normal case, not the exception, and the second look is free.

**Never capture speculatively.** Capture happens when something asks for it, so
the quota ledger is the only thing standing between a loop and the day's
budget - and it can only do that job if nothing captures behind its back.
"""

from __future__ import annotations

import base64
import io
import sys
import time
from dataclasses import dataclass
from typing import Any

from ..errors import VictorError

MAX_EDGE = 768
JPEG_QUALITY = 70
HASH_SIZE = 8
"""Perceptual hash grid. 8x8 gives a 64-bit fingerprint - enough to notice a
dialog opening, coarse enough to ignore a blinking cursor."""

DEFAULT_DISTANCE = 3
"""Hamming distance under which two screens count as the same. Zero would make
the cache useless (a clock ticks); large numbers would miss a real change."""


class CaptureUnavailable(VictorError):
    """Screen capture is not possible here."""

    exit_code = 5


@dataclass(frozen=True, slots=True)
class Screenshot:
    """A captured, downscaled image ready to send."""

    data: bytes
    width: int
    height: int
    fingerprint: str
    media_type: str = "image/jpeg"
    duration_ms: float = 0.0

    @property
    def kilobytes(self) -> float:
        return len(self.data) / 1024

    def as_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def as_data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.as_base64()}"


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise CaptureUnavailable(
            f"Pillow is not installed ({exc}). pip install -e '.[desktop]'"
        ) from exc
    return Image


# --- backends --------------------------------------------------------------
#
# Two, and macOS wins where both exist. `mss` is the portable choice and the one
# the plan named, but on macOS 26 it raises `ValueError: could not convert
# string to float: ''` before taking a single frame: it parses the OS version
# with `float(platform.mac_ver()[0])`, and `mac_ver()` returns an empty string
# there. Quartz is already a dependency on macOS for the accessibility tree and
# for synthesising events, so preferring it costs nothing and removes a library
# that breaks on a new OS release from the critical path.


def _quartz() -> Any | None:
    """The Quartz module, or ``None`` off macOS / without pyobjc."""
    if sys.platform != "darwin":
        return None
    try:
        import Quartz
    except ImportError:
        return None
    return Quartz


SCREEN_RECORDING_HINT = (
    "screen recording permission is not granted, so macOS returns a blank "
    "image instead of an error. Grant it to your terminal in System Settings "
    "-> Privacy & Security -> Screen Recording, then restart the terminal - "
    "the permission is only picked up on launch."
)


def _is_blank(image: Any) -> bool:
    """Is this a uniformly coloured rectangle rather than a screen?

    Worth its own check because of how macOS refuses. Without screen recording
    permission it does not raise and does not return nil - it hands back a
    perfectly valid image of nothing, which would be downscaled, hashed, and
    sent to a vision model, spending one of ~250 daily requests to ask which
    button is on a black rectangle.
    """
    extrema = image.convert("L").getextrema()
    return extrema[0] == extrema[1]


def _grab_quartz(region: tuple[int, int, int, int] | None):
    quartz = _quartz()
    Image = _require_pillow()

    if region is None:
        image = quartz.CGDisplayCreateImage(quartz.CGMainDisplayID())
    else:
        left, top, width, height = region
        image = quartz.CGWindowListCreateImage(
            quartz.CGRectMake(left, top, max(1, width), max(1, height)),
            quartz.kCGWindowListOptionOnScreenOnly,
            quartz.kCGNullWindowID,
            quartz.kCGWindowImageDefault,
        )
    if image is None:
        raise CaptureUnavailable(f"macOS returned no image. Most likely {SCREEN_RECORDING_HINT}")

    width = quartz.CGImageGetWidth(image)
    height = quartz.CGImageGetHeight(image)
    stride = quartz.CGImageGetBytesPerRow(image)
    data = quartz.CGDataProviderCopyData(quartz.CGImageGetDataProvider(image))
    # Retina displays give back a buffer at the backing scale, so the returned
    # image is larger than the requested rectangle. That is fine and even
    # desirable - it is downscaled next, and from more detail.
    picture = Image.frombuffer("RGBA", (width, height), bytes(data), "raw", "BGRA", stride, 1)
    picture = picture.convert("RGB")
    if _is_blank(picture):
        raise CaptureUnavailable(f"the screen came back blank: {SCREEN_RECORDING_HINT}")
    return picture


def _grab_mss(region: tuple[int, int, int, int] | None):
    Image = _require_pillow()
    try:
        import mss
    except ImportError as exc:
        raise CaptureUnavailable(
            f"mss is not installed ({exc}). pip install -e '.[desktop]'"
        ) from exc

    try:
        with mss.mss() as sct:
            if region is None:
                monitor = sct.monitors[0]
            else:
                left, top, width, height = region
                monitor = {
                    "left": left,
                    "top": top,
                    "width": max(1, width),
                    "height": max(1, height),
                }
            raw = sct.grab(monitor)
    except CaptureUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - mss raises bare exceptions
        raise CaptureUnavailable(f"mss could not capture the screen: {exc}") from exc
    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def _backend_name() -> str:
    return "quartz" if _quartz() is not None else "mss"


def perceptual_hash(image: Any, size: int = HASH_SIZE) -> str:
    """A difference hash: each bit is "is this pixel brighter than the next?".

    Robust to the things that change constantly and do not matter - a caret,
    a clock, compression noise - and sensitive to layout changes, which are
    exactly what would make a fresh vision call worth paying for.
    """
    small = image.convert("L").resize((size + 1, size), _require_pillow().LANCZOS)
    # getdata() is deprecated in Pillow 11 and goes away in 14; get_flattened_data
    # is the replacement but does not exist on older versions the plan supports.
    flatten = getattr(small, "get_flattened_data", None)
    pixels = list(flatten() if callable(flatten) else small.getdata())
    bits = []
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            bits.append("1" if pixels[offset + col] > pixels[offset + col + 1] else "0")
    return f"{int(''.join(bits), 2):0{size * size // 4}x}"


def hamming(a: str, b: str) -> int:
    """How many bits differ between two fingerprints."""
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class ScreenCapture:
    """Captures the screen, downscaled, with a same-screen cache."""

    def __init__(
        self,
        *,
        max_edge: int = MAX_EDGE,
        quality: int = JPEG_QUALITY,
        distance: int = DEFAULT_DISTANCE,
        grabber: Any | None = None,
    ) -> None:
        self.max_edge = max_edge
        self.quality = quality
        self.distance = distance
        self._grabber = grabber
        self._last: Screenshot | None = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def available() -> tuple[bool, str]:
        """Whether a capture would actually produce a picture of the screen.

        This used to answer "is mss importable", which is a different question
        and answered yes on a machine where capture was completely broken -
        `victor doctor` printed "screen capture ready" beside a vision tool that
        could not take a screenshot. So it now tries one.
        """
        try:
            _require_pillow()
        except CaptureUnavailable as exc:
            return False, str(exc)
        try:
            ScreenCapture()._grab(None)
        except CaptureUnavailable as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001 - a broken backend is a SKIP
            return False, f"screen capture failed: {exc}"
        return True, f"screen capture ready ({_backend_name()})"

    def _grab(self, region: tuple[int, int, int, int] | None):
        """Return a PIL image of the screen, or of ``region``.

        ``region`` is ``(left, top, width, height)`` - the same shape the
        accessibility tree reports a window rectangle in, so a caller can pass
        one straight through. It was previously unpacked here as
        ``(left, top, right, bottom)`` while the only caller passed width and
        height, which nothing caught because no test captured a region and the
        backend was too broken to run.
        """
        if self._grabber is not None:
            return self._grabber(region)
        if _quartz() is not None:
            return _grab_quartz(region)
        return _grab_mss(region)

    def capture(
        self, region: tuple[int, int, int, int] | None = None, *, force: bool = False
    ) -> tuple[Screenshot, bool]:
        """Capture the screen.

        Returns ``(screenshot, is_new)``. ``is_new`` is False when the screen is
        perceptually unchanged since the last capture, which is the caller's cue
        that a fresh vision call would buy nothing.
        """
        started = time.perf_counter()
        image = self._grab(region)
        image = self._downscale(image)
        fingerprint = perceptual_hash(image)

        if (
            not force
            and self._last is not None
            and hamming(fingerprint, self._last.fingerprint) <= self.distance
        ):
            self.hits += 1
            return self._last, False

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality, optimize=True)
        shot = Screenshot(
            data=buffer.getvalue(),
            width=image.width,
            height=image.height,
            fingerprint=fingerprint,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        self._last = shot
        self.misses += 1
        return shot, True

    def _downscale(self, image: Any):
        Image = _require_pillow()
        longest = max(image.width, image.height)
        if longest <= self.max_edge:
            return image
        scale = self.max_edge / longest
        return image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )

    def reset(self) -> None:
        self._last = None

    @property
    def savings(self) -> str:
        total = self.hits + self.misses
        if not total:
            return "no captures yet"
        return f"{self.hits}/{total} captures served from cache"
