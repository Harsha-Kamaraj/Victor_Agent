"""The tools the agent can call."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from .base import (
    MAX_OUTPUT_CHARS,
    Decision,
    DryRunInterceptor,
    Interceptor,
    PermissiveInterceptor,
    Review,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    truncate,
)
from .git import GitTool, is_repository
from .shell import ReadFileTool, ShellTool, describe_environment, screen

__all__ = [
    "Decision",
    "DryRunInterceptor",
    "GitTool",
    "Interceptor",
    "MAX_OUTPUT_CHARS",
    "PermissiveInterceptor",
    "ReadFileTool",
    "Review",
    "ShellTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_registry",
    "describe_environment",
    "is_repository",
    "screen",
    "truncate",
]


def build_registry(
    settings: Settings,
    *,
    cwd: Path | None = None,
    interceptor: Interceptor | None = None,
    shell: bool = True,
) -> ToolRegistry:
    """The standard tool set.

    ``settings.dry_run`` wraps the interceptor rather than disabling tools, so
    a dry run still exercises the full loop - model, tool selection, arguments -
    and only stops at the point of execution.
    """
    workdir = Path(cwd or Path.cwd())
    chosen = interceptor or PermissiveInterceptor()
    if settings.dry_run:
        chosen = DryRunInterceptor(chosen)

    registry = ToolRegistry(chosen)
    if shell:
        registry.register(ShellTool(cwd=workdir))
    registry.register(ReadFileTool(cwd=workdir))
    registry.register(GitTool(cwd=workdir))
    return registry
