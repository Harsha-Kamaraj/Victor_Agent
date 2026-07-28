"""Speech to text.

Charges the audio it sends against the ledger's audio-seconds budget *before*
the request goes out, so a transcription that times out still costs what it
actually consumed. Whisper bills by audio duration, not by response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from ..config import Settings
from ..errors import ProviderError, QuotaExhausted
from ..providers import Router, Selection, Workload
from ..tracing import Trace
from .audio import STT_SAMPLE_RATE, Segment

#: Transcription endpoint per provider. Groq speaks the OpenAI audio API.
ENDPOINTS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1/audio/transcriptions",
}


@dataclass(frozen=True, slots=True)
class Transcript:
    """What was said, and what it cost to find out."""

    text: str
    model: str
    audio_seconds: float
    latency_ms: float

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def __str__(self) -> str:
        return self.text


class Transcriber:
    """Sends a :class:`Segment` to whichever STT model the router allows."""

    def __init__(
        self,
        settings: Settings,
        router: Router,
        *,
        client: httpx.Client | None = None,
        trace: Trace | None = None,
        timeout: float = 30.0,
        attempts: int = 3,
        backoff: float = 0.5,
    ) -> None:
        self._settings = settings
        self._router = router
        self._client = client or httpx.Client(timeout=timeout)
        self._trace = trace or Trace.disabled()
        self._attempts = max(1, attempts)
        self._backoff = backoff

    def transcribe(
        self,
        segment: Segment,
        *,
        language: str | None = "en",
        prompt: str | None = None,
    ) -> Transcript:
        """Transcribe one utterance.

        ``prompt`` biases the decoder - later phases pass domain vocabulary
        ("git", "PowerShell", app names) so command words survive a bad mic.
        """
        audio = segment.resampled(STT_SAMPLE_RATE)
        duration = audio.duration
        selection = self._router.select(Workload.STT, audio_seconds=duration)
        self._trace.selection(selection)

        endpoint = ENDPOINTS.get(selection.spec.provider)
        if endpoint is None:
            raise ProviderError(f"no STT endpoint known for provider {selection.spec.provider!r}")

        # Charge before the call: the audio is spent whether or not we get a
        # response back.
        self._router.record(selection, audio_seconds=duration)

        started = time.perf_counter()
        with self._trace.span(
            "stt.transcribe", model=selection.key, audio_seconds=round(duration, 2)
        ) as span:
            text = self._post(endpoint, selection, audio.to_wav(), language, prompt)
            span["chars"] = len(text)
        latency_ms = (time.perf_counter() - started) * 1000

        return Transcript(
            text=text.strip(),
            model=selection.key,
            audio_seconds=duration,
            latency_ms=latency_ms,
        )

    # -- transport ---------------------------------------------------------

    def _post(
        self,
        endpoint: str,
        selection: Selection,
        wav: bytes,
        language: str | None,
        prompt: str | None,
    ) -> str:
        credential = selection.spec.credential
        key = self._settings.secret(credential) if credential else None
        data: dict[str, str] = {"model": selection.spec.model, "response_format": "json"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        response = self._post_with_retry(endpoint, selection, wav, data, key)

        if response.status_code == 429:
            retry = response.headers.get("retry-after")
            raise QuotaExhausted(
                selection.key,
                "provider returned 429 - the ledger's declared limits are behind reality",
                float(retry) if retry and retry.isdigit() else None,
            )
        if response.status_code in (401, 403):
            env = (selection.spec.credential or "api key").upper()
            raise ProviderError(f"{selection.key}: {env} rejected (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise ProviderError(
                f"{selection.key}: HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return str(response.json().get("text", ""))
        except ValueError as exc:
            raise ProviderError(f"{selection.key}: response was not JSON") from exc

    def _post_with_retry(
        self,
        endpoint: str,
        selection: Selection,
        wav: bytes,
        data: dict[str, str],
        key: str | None,
    ) -> httpx.Response:
        """Upload, retrying only what is worth retrying.

        Speech is the one stage the user is actively waiting through, so a
        dropped connection must not lose an utterance they already spoke - they
        would have to say it again, and the audio seconds are spent either way.

        Only transient faults are retried: connection errors and 5xx. A 429 is
        not retried here (the ledger and router handle quota, and hammering a
        rate limit makes it worse), and a 401 never becomes valid by asking
        twice.
        """
        last: Exception | None = None

        for attempt in range(self._attempts):
            if attempt:
                # 0.5s, 1s, 2s - short, because someone is waiting to be heard.
                delay = self._backoff * (2 ** (attempt - 1))
                self._trace.event(
                    "stt.retry", attempt=attempt, delay_s=round(delay, 2), model=selection.key
                )
                time.sleep(delay)

            try:
                response = self._client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                    data=data,
                )
            except httpx.HTTPError as exc:
                last = exc
                continue

            if response.status_code >= 500:
                last = ProviderError(f"HTTP {response.status_code}")
                continue
            return response

        raise ProviderError(
            f"{selection.key}: giving up after {self._attempts} attempts: "
            f"{type(last).__name__}: {last}"
        ) from last

    def close(self) -> None:
        self._client.close()
