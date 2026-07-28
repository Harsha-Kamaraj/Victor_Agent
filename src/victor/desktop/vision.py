"""The vision fallback - used only when the tree is not enough.

Two provider formats, one client. Gemini takes ``inline_data``; Groq speaks the
OpenAI ``image_url`` shape. Both are here because the routing chain crosses
providers: when Gemini's ~250/day is spent, vision continues on Groq's separate
allowance rather than stopping.

**Set-of-Mark, not coordinates.** The screenshot is annotated with the numbered
boxes from the UIA snapshot, and the model is asked which *number* to act on.
It never returns a coordinate, so it cannot return a wrong one - the rectangle
comes from the operating system either way. That keeps the failure mode
"chose the wrong button", which is recoverable and visible, rather than
"clicked 30 pixels off a button", which is neither.

**Checks the budget first.** The plan requires degrading gracefully with a
spoken "I'm out of vision quota for today". Running out of vision must leave a
working agent, not a crashed one.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import Settings
from ..errors import NoProviderAvailable, ProviderError
from ..providers import Router, Selection, Workload
from ..tracing import Trace
from .capture import Screenshot
from .elements import Element, Snapshot

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

MAX_TOKENS = 256

SYSTEM = """\
You are looking at a screenshot of an application window. Numbered boxes have \
been drawn over the controls the operating system reported.

Answer with the NUMBER of the element the user's request refers to, and nothing \
else. If no numbered element matches, answer NONE.

Never guess pixel coordinates. Only the numbers are addressable.\
"""

_NUMBER = re.compile(r"\b(\d{1,3})\b")


@dataclass(frozen=True, slots=True)
class VisionAnswer:
    """What the model picked, and what it cost."""

    index: int | None
    raw: str
    model: str
    latency_ms: float
    element: Element | None = None

    @property
    def found(self) -> bool:
        return self.index is not None

    def __str__(self) -> str:
        if self.element is not None:
            return f"[{self.index}] {self.element.label}"
        return "no matching element" if self.index is None else f"[{self.index}]"


class VisionUnavailable(ProviderError):
    """No vision model can be called right now - usually the day's quota."""


class VisionClient:
    """Asks a vision model which numbered element to act on."""

    def __init__(
        self,
        settings: Settings,
        router: Router,
        *,
        client: httpx.Client | None = None,
        trace: Trace | None = None,
        timeout: float = 45.0,
    ) -> None:
        self._settings = settings
        self._router = router
        self._client = client or httpx.Client(timeout=timeout)
        self._trace = trace or Trace.disabled()

    def available(self) -> tuple[bool, str]:
        """Whether a vision call could be made at all, without making one."""
        try:
            selection = self._router.select(Workload.VISION)
        except NoProviderAvailable as exc:
            return False, str(exc)
        return True, selection.key

    def locate(
        self, request: str, shot: Screenshot, snapshot: Snapshot | None = None
    ) -> VisionAnswer:
        """Ask which numbered element matches ``request``.

        Raises :class:`VisionUnavailable` when the budget is spent, so callers
        can say so out loud and fall back to the tree rather than crash.
        """
        try:
            selection = self._router.select(Workload.VISION)
        except NoProviderAvailable as exc:
            raise VisionUnavailable(
                "no vision quota left today - falling back to the accessibility tree"
            ) from exc

        self._trace.selection(selection)
        prompt = _prompt(request, snapshot)
        self._router.record(selection)

        started = time.perf_counter()
        with self._trace.span("vision.locate", model=selection.key, request=request) as span:
            raw = self._call(selection, prompt, shot)
            span["answer"] = raw[:80]
        latency_ms = (time.perf_counter() - started) * 1000

        index = _parse_index(raw)
        element = snapshot.by_index(index) if snapshot and index is not None else None
        if index is not None and snapshot is not None and element is None:
            # The model named an element that does not exist. Treat as no answer
            # rather than passing an unusable index to the actuation layer.
            self._trace.event("vision.invalid_index", index=index, count=len(snapshot))
            index = None

        return VisionAnswer(
            index=index, raw=raw, model=selection.key, latency_ms=latency_ms, element=element
        )

    # -- transport ---------------------------------------------------------

    def _call(self, selection: Selection, prompt: str, shot: Screenshot) -> str:
        provider = selection.spec.provider
        key = self._settings.secret(selection.spec.credential or "") or ""

        if provider == "gemini":
            url = GEMINI_ENDPOINT.format(model=selection.spec.model)
            payload = {
                "system_instruction": {"parts": [{"text": SYSTEM}]},
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": shot.media_type,
                                    "data": shot.as_base64(),
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {"maxOutputTokens": MAX_TOKENS, "temperature": 0.0},
            }
            headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        elif provider == "groq":
            url = GROQ_ENDPOINT
            payload = {
                "model": selection.spec.model,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": shot.as_data_url()}},
                        ],
                    },
                ],
            }
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        else:
            raise ProviderError(f"no vision endpoint known for provider {provider!r}")

        try:
            response = self._client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{selection.key}: {type(exc).__name__}: {exc}") from exc

        if response.status_code == 429:
            raise VisionUnavailable(f"{selection.key}: rate limited")
        if response.status_code in (401, 403):
            env = (selection.spec.credential or "api key").upper()
            raise ProviderError(f"{selection.key}: {env} rejected (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise ProviderError(
                f"{selection.key}: HTTP {response.status_code}: {response.text[:200]}"
            )

        return _extract_text(response.json(), provider, selection.key)

    def close(self) -> None:
        self._client.close()


def _prompt(request: str, snapshot: Snapshot | None) -> str:
    lines = [f"Request: {request}"]
    if snapshot is not None and not snapshot.empty:
        lines.append("")
        lines.append("The numbered elements the operating system reported:")
        lines.append(snapshot.render(limit=60))
    lines.append("")
    lines.append("Which number should be acted on? Answer with the number alone.")
    return "\n".join(lines)


def _extract_text(body: dict[str, Any], provider: str, key: str) -> str:
    try:
        if provider == "gemini":
            parts = body["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        return str(body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            f"{key}: unexpected response shape: {json.dumps(body)[:200]}"
        ) from exc


def _parse_index(answer: str) -> int | None:
    cleaned = answer.strip()
    if not cleaned or cleaned.upper().startswith("NONE"):
        return None
    match = _NUMBER.search(cleaned)
    return int(match.group(1)) if match else None


def annotate(shot: Screenshot, snapshot: Snapshot, *, limit: int = 60) -> Screenshot:
    """Draw the numbered boxes the model is asked to choose between.

    Set-of-Mark prompting: the numbers come from the OS tree, so the model's
    job is recognition rather than localisation - the thing vision models are
    reliable at, instead of the thing they are not.
    """
    from PIL import Image, ImageDraw

    image = Image.open(__import__("io").BytesIO(shot.data)).convert("RGB")
    # The snapshot is in screen coordinates; the screenshot has been downscaled.
    window = snapshot.rect
    if window is None or window.empty:
        return shot
    scale_x = image.width / window.width
    scale_y = image.height / window.height

    draw = ImageDraw.Draw(image)
    for element in list(snapshot)[:limit]:
        if not element.actionable:
            continue
        left = int((element.rect.left - window.left) * scale_x)
        top = int((element.rect.top - window.top) * scale_y)
        right = int((element.rect.right - window.left) * scale_x)
        bottom = int((element.rect.bottom - window.top) * scale_y)
        if right <= left or bottom <= top:
            continue
        draw.rectangle([left, top, right, bottom], outline=(255, 40, 40), width=2)
        label = str(element.index)
        draw.rectangle([left, top, left + 8 * len(label) + 6, top + 16], fill=(255, 40, 40))
        draw.text((left + 3, top + 2), label, fill=(255, 255, 255))

    buffer = __import__("io").BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return Screenshot(
        data=buffer.getvalue(),
        width=image.width,
        height=image.height,
        fingerprint=shot.fingerprint,
        duration_ms=shot.duration_ms,
    )
