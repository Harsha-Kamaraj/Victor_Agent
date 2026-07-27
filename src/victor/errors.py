"""Exception hierarchy.

Everything Victor raises deliberately descends from :class:`VictorError` so the
CLI can distinguish "we knew this could happen" from a genuine crash and print
accordingly.
"""

from __future__ import annotations


class VictorError(Exception):
    """Base class for every error Victor raises on purpose."""

    exit_code = 1


class ConfigError(VictorError):
    """Configuration is missing or self-contradictory."""

    exit_code = 2


class ProviderError(VictorError):
    """A provider call failed."""


class QuotaExhausted(ProviderError):
    """A provider's free allowance is spent for the current window."""

    exit_code = 3

    def __init__(self, key: str, reason: str, retry_after: float | None = None) -> None:
        self.key = key
        self.reason = reason
        self.retry_after = retry_after
        detail = f"{key}: {reason}"
        if retry_after is not None:
            detail += f" (retry in {retry_after:.0f}s)"
        super().__init__(detail)


class NoProviderAvailable(ProviderError):
    """No candidate in a workload's routing chain can serve the request."""

    exit_code = 3

    def __init__(self, workload: str, attempts: list[str]) -> None:
        self.workload = workload
        self.attempts = attempts
        detail = "\n".join(f"  - {a}" for a in attempts)
        super().__init__(f"no provider available for {workload}:\n{detail}")
