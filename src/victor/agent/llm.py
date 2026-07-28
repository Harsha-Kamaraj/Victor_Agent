"""Chat client for the text tier.

Groq speaks the OpenAI chat API, so one client covers every text model in the
routing table. The client owns two responsibilities the rest of the agent
should not have to think about: reconciling real token usage back into the
quota ledger after each call, and retrying down the routing chain when a
provider rate-limits despite the ledger believing there was room.

That second case is not hypothetical. The ledger's limits are declared
conservatively but providers change them silently, so a 429 is treated as
authoritative: the model is marked spent for the day and the next candidate
takes over.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings
from ..errors import NoProviderAvailable, ProviderError
from ..providers import Router, Selection, Workload
from ..tracing import Trace

ENDPOINTS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}

#: Rough characters-per-token, used only to pre-check quota before a call.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A function call the model wants made."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


@dataclass(frozen=True, slots=True)
class Reply:
    """One assistant turn."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> dict[str, Any]:
        """The assistant message to append to the transcript.

        Tool calls must be echoed back verbatim or the API rejects the
        following tool-result messages as orphaned.
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments},
                }
                for call in self.tool_calls
            ]
        return message


@dataclass
class Usage:
    """Running totals for one session."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, reply: Reply) -> None:
        self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        self.models[reply.model] = self.models.get(reply.model, 0) + 1


class ChatClient:
    """Calls whichever text model the router allows, and pays for it."""

    def __init__(
        self,
        settings: Settings,
        router: Router,
        *,
        client: httpx.Client | None = None,
        trace: Trace | None = None,
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 1_024,
    ) -> None:
        self._settings = settings
        self._router = router
        self._client = client or httpx.Client(timeout=timeout)
        self._trace = trace or Trace.disabled()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = Usage()

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        """One chat completion, falling down the routing chain on 429."""
        estimate = _estimate_tokens(messages, tools)
        exhausted: list[str] = []

        while True:
            try:
                selection = self._router.select(Workload.TEXT, tokens=estimate)
            except NoProviderAvailable:
                if exhausted:
                    raise ProviderError(
                        "every text model is rate limited: " + "; ".join(exhausted)
                    ) from None
                raise

            if selection.key in exhausted:
                # The router handed back a model we already know is spent -
                # nothing left to try.
                raise ProviderError("every text model is rate limited: " + "; ".join(exhausted))

            self._trace.selection(selection)
            try:
                return self._call(selection, messages, tools, temperature, max_tokens)
            except _RateLimited as exc:
                # Trust the provider over the ledger and burn the day's budget
                # for this model, so the next select() moves on.
                self._burn(selection)
                exhausted.append(f"{selection.key} ({exc})")

    # -- transport ---------------------------------------------------------

    def _call(
        self,
        selection: Selection,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Reply:
        endpoint = ENDPOINTS.get(selection.spec.provider)
        if endpoint is None:
            raise ProviderError(f"no chat endpoint known for {selection.spec.provider!r}")

        payload: dict[str, Any] = {
            "model": selection.spec.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        key = self._settings.secret(selection.spec.credential or "")
        self._router.record(selection)  # reserve the request
        started = time.perf_counter()

        try:
            response = self._client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{selection.key}: {type(exc).__name__}: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 429:
            raise _RateLimited(response.headers.get("retry-after", "no retry-after"))
        if response.status_code in (401, 403):
            env = (selection.spec.credential or "api key").upper()
            raise ProviderError(f"{selection.key}: {env} rejected (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise ProviderError(
                f"{selection.key}: HTTP {response.status_code}: {response.text[:300]}"
            )

        reply = _parse(response.json(), selection.key, latency_ms)
        # Reconcile actual usage; the reservation only counted the request.
        self._router.reconcile(selection, tokens=reply.total_tokens)
        self.usage.add(reply)
        self._trace.event(
            "llm.reply",
            model=reply.model,
            duration_ms=round(latency_ms),
            tokens=reply.total_tokens,
            tool_calls=[str(c) for c in reply.tool_calls] or None,
            finish_reason=reply.finish_reason,
        )
        return reply

    def _burn(self, selection: Selection) -> None:
        """Mark a model spent after the provider contradicted the ledger."""
        limit = selection.spec.limits.requests_per_day
        if limit:
            self._router.record(selection, requests=limit)

    def close(self) -> None:
        self._client.close()


class _RateLimited(Exception):
    """Internal: provider said 429."""


def _parse(body: dict[str, Any], model: str, latency_ms: float) -> Reply:
    choices = body.get("choices") or []
    if not choices:
        raise ProviderError(f"{model}: response had no choices")

    message = choices[0].get("message") or {}
    usage = body.get("usage") or {}
    calls: list[ToolCall] = []

    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            # A model that emits malformed JSON should get told, not crash the
            # run. The empty dict makes the tool report bad arguments.
            parsed = {}
        calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{len(calls)}",
                name=function.get("name") or "",
                arguments=parsed if isinstance(parsed, dict) else {},
                raw_arguments=raw_args,
            )
        )

    return Reply(
        content=(message.get("content") or "").strip(),
        tool_calls=tuple(calls),
        model=model,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        latency_ms=latency_ms,
        finish_reason=choices[0].get("finish_reason", ""),
    )


def _estimate_tokens(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> int:
    """Cheap size estimate, used only to pre-check the per-minute budget."""
    size = sum(len(json.dumps(m, default=str)) for m in messages)
    if tools:
        size += sum(len(json.dumps(t)) for t in tools)
    return size // CHARS_PER_TOKEN
