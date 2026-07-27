"""Free-tier accounting.

Every provider Victor uses meters something different - requests per minute,
requests per day, tokens per minute, audio seconds per day - and the daily
windows do not even reset in the same timezone (Groq rolls at UTC midnight,
Google at midnight Pacific). The ledger normalises all of that into one
question the router can ask before spending anything:

    "is there free allowance left on this model right now?"

State lives in a single JSON file so it survives across processes and reboots.
Counting is deliberately optimistic-then-corrected: the router reserves a
request before the call and reconciles real token usage afterwards, so a crash
mid-call over-counts by one request rather than silently over-spending.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 1
_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class QuotaLimits:
    """A provider's published free allowance for one model.

    ``None`` means "not metered on this axis", not "zero".
    """

    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None
    audio_seconds_per_day: float | None = None
    reset_timezone: str = "UTC"

    @property
    def metered(self) -> bool:
        return any(
            v is not None
            for v in (
                self.requests_per_minute,
                self.requests_per_day,
                self.tokens_per_minute,
                self.tokens_per_day,
                self.audio_seconds_per_day,
            )
        )


UNMETERED = QuotaLimits()
"""For local models - Piper, fastembed, the UIA tree. Always available."""


@dataclass(frozen=True, slots=True)
class QuotaStatus:
    """The answer to "can I call this model right now?"."""

    key: str
    allowed: bool
    reason: str | None = None
    retry_after: float | None = None
    requests_remaining: int | None = None
    tokens_remaining: int | None = None
    audio_seconds_remaining: float | None = None

    def __str__(self) -> str:
        if self.allowed:
            return f"{self.key}: available"
        return f"{self.key}: {self.reason}"


@dataclass(slots=True)
class _Bucket:
    """Consumption for one model, for one daily window."""

    day: str = ""
    requests: int = 0
    tokens: int = 0
    audio_seconds: float = 0.0
    # (epoch_seconds, tokens) for everything inside the trailing minute.
    events: list[list[float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "requests": self.requests,
            "tokens": self.tokens,
            "audio_seconds": self.audio_seconds,
            "events": self.events,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> _Bucket:
        return cls(
            day=str(raw.get("day", "")),
            requests=int(raw.get("requests", 0)),
            tokens=int(raw.get("tokens", 0)),
            audio_seconds=float(raw.get("audio_seconds", 0.0)),
            events=[[float(t), float(n)] for t, n in raw.get("events", [])],
        )


class QuotaLedger:
    """Persistent, thread-safe free-tier accounting.

    Parameters
    ----------
    path:
        JSON file to persist to. Parent directory is created on first write.
    clock:
        Injectable source of epoch seconds. Tests use this to fast-forward
        through minute and day boundaries without sleeping.
    autoflush:
        Persist after every mutation. Off in hot loops; call :meth:`flush`.
    """

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
        autoflush: bool = True,
    ) -> None:
        self._path = path
        self._clock = clock
        self._autoflush = autoflush
        self._lock = threading.RLock()
        self._buckets: dict[str, _Bucket] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt ledger must never stop the agent from starting. Losing
            # today's count is a worse outcome than a crash only in theory; in
            # practice it just means the next call re-checks against the API.
            return
        if raw.get("version") != SCHEMA_VERSION:
            return
        self._buckets = {
            key: _Bucket.from_json(value) for key, value in raw.get("buckets", {}).items()
        }

    def flush(self) -> None:
        """Atomically write the ledger to disk."""
        with self._lock:
            payload = {
                "version": SCHEMA_VERSION,
                "updated_at": datetime.fromtimestamp(self._clock(), tz=ZoneInfo("UTC")).isoformat(),
                "buckets": {k: v.to_json() for k, v in self._buckets.items()},
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                os.replace(tmp, self._path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    # -- internals ---------------------------------------------------------

    def _day_key(self, limits: QuotaLimits, now: float) -> str:
        tz = ZoneInfo(limits.reset_timezone)
        return datetime.fromtimestamp(now, tz=tz).strftime("%Y-%m-%d")

    def _bucket(self, key: str, limits: QuotaLimits, now: float) -> _Bucket:
        """Fetch the bucket for ``key``, rolling it over if the day changed."""
        bucket = self._buckets.get(key)
        day = self._day_key(limits, now)
        if bucket is None:
            bucket = _Bucket(day=day)
            self._buckets[key] = bucket
        elif bucket.day != day:
            bucket.day = day
            bucket.requests = 0
            bucket.tokens = 0
            bucket.audio_seconds = 0.0
            bucket.events.clear()
        bucket.events = [e for e in bucket.events if now - e[0] < _MINUTE]
        return bucket

    @staticmethod
    def _oldest_event_age(bucket: _Bucket, now: float) -> float:
        """Seconds until the trailing-minute window frees up a slot."""
        if not bucket.events:
            return 0.0
        return max(0.0, _MINUTE - (now - bucket.events[0][0]))

    # -- public API --------------------------------------------------------

    def check(
        self,
        key: str,
        limits: QuotaLimits,
        *,
        tokens: int = 0,
        audio_seconds: float = 0.0,
    ) -> QuotaStatus:
        """Would a call of this size fit inside the free allowance?

        ``tokens`` and ``audio_seconds`` are the *estimated* cost of the call
        being considered; pass 0 to ask only about the request counters.
        """
        with self._lock:
            now = self._clock()
            bucket = self._bucket(key, limits, now)

            if not limits.metered:
                return QuotaStatus(key=key, allowed=True)

            minute_requests = len(bucket.events)
            minute_tokens = int(sum(e[1] for e in bucket.events))

            requests_remaining = (
                None
                if limits.requests_per_day is None
                else max(0, limits.requests_per_day - bucket.requests)
            )
            tokens_remaining = (
                None
                if limits.tokens_per_day is None
                else max(0, limits.tokens_per_day - bucket.tokens)
            )
            audio_remaining = (
                None
                if limits.audio_seconds_per_day is None
                else max(0.0, limits.audio_seconds_per_day - bucket.audio_seconds)
            )

            def deny(reason: str, retry_after: float | None) -> QuotaStatus:
                return QuotaStatus(
                    key=key,
                    allowed=False,
                    reason=reason,
                    retry_after=retry_after,
                    requests_remaining=requests_remaining,
                    tokens_remaining=tokens_remaining,
                    audio_seconds_remaining=audio_remaining,
                )

            # Daily limits first: they are the ones that end the day's budget,
            # and their retry_after is measured in hours, not seconds.
            seconds_to_reset = self._seconds_until_day_reset(limits, now)
            if limits.requests_per_day is not None and bucket.requests >= limits.requests_per_day:
                return deny(
                    f"daily request limit reached ({bucket.requests}/{limits.requests_per_day})",
                    seconds_to_reset,
                )
            if limits.tokens_per_day is not None and bucket.tokens + tokens > limits.tokens_per_day:
                return deny(
                    f"daily token limit reached ({bucket.tokens}/{limits.tokens_per_day})",
                    seconds_to_reset,
                )
            if (
                limits.audio_seconds_per_day is not None
                and bucket.audio_seconds + audio_seconds > limits.audio_seconds_per_day
            ):
                return deny(
                    "daily audio limit reached "
                    f"({bucket.audio_seconds:.0f}/{limits.audio_seconds_per_day:.0f}s)",
                    seconds_to_reset,
                )

            # Per-minute limits are transient - worth reporting separately so
            # the router can choose to wait rather than fall back.
            if (
                limits.requests_per_minute is not None
                and minute_requests >= limits.requests_per_minute
            ):
                return deny(
                    f"rate limited ({minute_requests}/{limits.requests_per_minute} req/min)",
                    self._oldest_event_age(bucket, now),
                )
            if (
                limits.tokens_per_minute is not None
                and minute_tokens + tokens > limits.tokens_per_minute
            ):
                return deny(
                    f"rate limited ({minute_tokens}/{limits.tokens_per_minute} tok/min)",
                    self._oldest_event_age(bucket, now),
                )

            return QuotaStatus(
                key=key,
                allowed=True,
                requests_remaining=requests_remaining,
                tokens_remaining=tokens_remaining,
                audio_seconds_remaining=audio_remaining,
            )

    def _seconds_until_day_reset(self, limits: QuotaLimits, now: float) -> float:
        tz = ZoneInfo(limits.reset_timezone)
        local = datetime.fromtimestamp(now, tz=tz)
        tomorrow = local.replace(hour=0, minute=0, second=0, microsecond=0)
        return (tomorrow.timestamp() + 86400) - now

    def record(
        self,
        key: str,
        limits: QuotaLimits,
        *,
        requests: int = 1,
        tokens: int = 0,
        audio_seconds: float = 0.0,
    ) -> None:
        """Charge a completed (or in-flight) call against the allowance."""
        with self._lock:
            now = self._clock()
            bucket = self._bucket(key, limits, now)
            bucket.requests += requests
            bucket.tokens += tokens
            bucket.audio_seconds += audio_seconds
            for _ in range(max(requests, 0)):
                bucket.events.append([now, float(tokens)])
            if requests <= 0 and tokens and bucket.events:
                # Post-hoc token reconciliation for a request already counted.
                bucket.events[-1][1] += float(tokens)
            if self._autoflush:
                self.flush()

    def snapshot(self, keys: dict[str, QuotaLimits]) -> list[QuotaStatus]:
        """Current status for every model in ``keys``, for display."""
        return [self.check(key, limits) for key, limits in keys.items()]

    def reset(self, key: str | None = None) -> None:
        """Wipe accounting for one model, or all of them."""
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)
            if self._autoflush:
                self.flush()

    def usage(self, key: str) -> tuple[int, int, float]:
        """Raw ``(requests, tokens, audio_seconds)`` for the current window."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return (0, 0, 0.0)
            return (bucket.requests, bucket.tokens, bucket.audio_seconds)
