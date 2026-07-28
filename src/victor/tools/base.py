"""Tool contract, registry, and the seam the safety layer plugs into.

Two things here matter beyond the obvious plumbing.

**Output is truncated before it reaches the model.** The text tier allows 8,000
tokens per minute. One ``git log`` or a noisy build can exceed that on its own,
and the failure mode is not a nice error - it is the agent losing the earlier
half of its own conversation. Truncation happens at the tool boundary, keeping
the head and tail, which is where the useful part of command output lives.

**Every call passes through an interceptor.** P2 ships a permissive one that
allows everything except the obviously catastrophic. P3 replaces it with the
real classifier, confirmation flow and journal. Building the seam now means
that swap is a constructor argument rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ..errors import VictorError

MAX_OUTPUT_CHARS = 4_000
"""Ceiling on what one tool result may contribute to the context."""

TRUNCATION_NOTE = "\n... [{dropped:,} characters omitted] ...\n"


class ToolError(VictorError):
    """A tool could not run at all - not to be confused with a failed command."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What a tool is called, what it does, and what it takes."""

    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool = False
    """Declares intent up front, so the P3 interceptor does not have to infer
    it by parsing arguments after the fact."""

    def to_openai(self) -> dict[str, Any]:
        """The JSON-schema form the chat API expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of one tool call.

    ``ok=False`` means the command ran and failed - a non-zero exit code is
    information the model should see and reason about, not an exception.
    """

    ok: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def for_model(self, limit: int = MAX_OUTPUT_CHARS) -> str:
        """Render for the conversation, truncated to fit the token budget."""
        body = self.output.strip()
        if self.error:
            body = f"{body}\n[stderr] {self.error.strip()}" if body else self.error.strip()
        if not body:
            body = "(no output)"
        if not self.ok and "exit_code" in self.metadata:
            body = f"exit code {self.metadata['exit_code']}\n{body}"
        return truncate(body, limit)


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the head and tail, drop the middle.

    Command output puts the invocation at the top and the error at the bottom;
    a plain head-truncation would throw away the half that explains the failure.
    """
    if len(text) <= limit:
        return text
    keep = (limit - len(TRUNCATION_NOTE)) // 2
    dropped = len(text) - 2 * keep
    return text[:keep] + TRUNCATION_NOTE.format(dropped=dropped) + text[-keep:]


@runtime_checkable
class Tool(Protocol):
    """Something the agent can call."""

    spec: ToolSpec

    def run(self, **kwargs: Any) -> ToolResult: ...


# --- the safety seam ------------------------------------------------------


class Decision(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Review:
    """An interceptor's verdict on one proposed call."""

    decision: Decision
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.DENY


@runtime_checkable
class Interceptor(Protocol):
    """Vets a call before it runs. P3 supplies the real implementation."""

    def review(self, spec: ToolSpec, arguments: dict[str, Any]) -> Review: ...


class PermissiveInterceptor:
    """P2's placeholder: allows everything.

    Deliberately not the default anywhere a shell is involved - see
    :class:`victor.tools.shell.ShellTool`, which carries its own last-ditch
    denylist until P3 lands.
    """

    def review(self, spec: ToolSpec, arguments: dict[str, Any]) -> Review:
        return Review(Decision.ALLOW)


class DryRunInterceptor:
    """Turns every mutating call into a description of what would have run."""

    def __init__(self, inner: Interceptor | None = None) -> None:
        self._inner = inner or PermissiveInterceptor()

    def review(self, spec: ToolSpec, arguments: dict[str, Any]) -> Review:
        if spec.mutating:
            return Review(Decision.DENY, "dry run: mutating tools are not executed")
        return self._inner.review(spec, arguments)


# --- registry -------------------------------------------------------------


class ToolRegistry:
    """The set of tools one agent run may use."""

    def __init__(self, interceptor: Interceptor | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.interceptor = interceptor or PermissiveInterceptor()

    def register(self, tool: Tool) -> Tool:
        if tool.spec.name in self._tools:
            raise ToolError(f"tool {tool.spec.name!r} is already registered")
        self._tools[tool.spec.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions for the chat API, in registration order."""
        return [tool.spec.to_openai() for tool in self._tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Vet, then execute.

        A model that invents a tool name or passes the wrong arguments gets a
        failed :class:`ToolResult` rather than an exception, so it can read the
        message and correct itself on the next step.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                error=f"no such tool {name!r}. Available: {', '.join(self.names)}",
            )

        review = self.interceptor.review(tool.spec, arguments)
        if review.blocked:
            return ToolResult(
                ok=False,
                error=f"blocked: {review.reason}",
                metadata={"decision": str(review.decision)},
            )

        try:
            result = tool.run(**arguments)
        except TypeError as exc:
            return ToolResult(ok=False, error=f"bad arguments for {name!r}: {exc}")
        except VictorError:
            # Aborts and other deliberate failures propagate: a kill switch that
            # a tool could swallow into a failed result would not be a kill
            # switch.
            raise

        # Tell the interceptor what actually happened, so it can journal the
        # action and its undo recipe. Optional, so P2's simple interceptors
        # stay valid.
        note = getattr(self.interceptor, "note_execution", None)
        if callable(note):
            note(tool.spec, arguments, ok=result.ok)
        return result
