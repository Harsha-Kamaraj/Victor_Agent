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
    kill_switch: object | None = None,
    trash: object | None = None,
    desktop: bool | None = None,
    app: str | None = None,
) -> ToolRegistry:
    """The standard tool set, gated by the safety interceptor.

    ``settings.dry_run`` is handled inside :class:`SafetyInterceptor` rather
    than by disabling tools, so a dry run still exercises the full loop - model,
    tool selection, arguments, classification - and stops only at execution.
    """
    from ..safety import SafetyInterceptor, build_confirmer

    workdir = Path(cwd or Path.cwd())
    chosen = interceptor or SafetyInterceptor(
        confirmer=build_confirmer(),
        kill_switch=kill_switch,
        dry_run=settings.dry_run,
        require_confirmation=settings.confirm_destructive,
    )

    registry = ToolRegistry(chosen)
    if shell:
        registry.register(ShellTool(cwd=workdir, kill_switch=kill_switch, trash=trash))
    registry.register(ReadFileTool(cwd=workdir))
    registry.register(GitTool(cwd=workdir))

    # Desktop control is opt-in: see Settings.desktop_control. Registering the
    # tools is what makes them visible to the model, so a run without the flag
    # cannot click anything even by accident - the capability is absent rather
    # than merely discouraged.
    if desktop if desktop is not None else settings.desktop_control:
        from .desktop import build_desktop_tools

        for tool in build_desktop_tools(kill_switch=kill_switch, app=app):
            registry.register(tool)
    return registry
