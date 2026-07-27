"""Command line entry point."""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import Settings, get_settings
from .doctor import Check, Status, run_checks
from .errors import VictorError
from .providers import Router, Workload
from .providers.registry import ROUTING_TABLE
from .quota import QuotaLedger
from .tracing import list_traces, read_trace

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="victor",
    help="Voice-driven computer-use agent for Windows.",
    add_completion=False,
)
trace_app = typer.Typer(help="Inspect session traces.", no_args_is_help=True)
app.add_typer(trace_app, name="trace")

_STATUS_STYLE: dict[Status, tuple[str, str]] = {
    Status.OK: ("green", "OK"),
    Status.WARN: ("yellow", "WARN"),
    Status.FAIL: ("bold red", "FAIL"),
    Status.SKIP: ("dim", "SKIP"),
    Status.PENDING: ("cyan", "PENDING"),
}


def _settings() -> Settings:
    return get_settings()


def _render_checks(checks: list[Check]) -> None:
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column("status", width=8)
    table.add_column("name", style="bold", width=22)
    table.add_column("detail", overflow="fold")

    for check in checks:
        style, label = _STATUS_STYLE[check.status]
        table.add_row(Text(label, style=style), check.name, check.detail)
        if check.hint and check.status in (Status.FAIL, Status.WARN):
            table.add_row("", "", Text(f"-> {check.hint}", style="dim"))
    console.print(table)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", help="Print the version and exit.")
    ] = False,
) -> None:
    # invoke_without_command lets `--version` work without a subcommand; the
    # trade-off is that a bare `victor` reaches here too, so print help itself.
    if version:
        console.print(f"victor {__version__}")
        raise typer.Exit
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit


@app.command()
def doctor(
    network: Annotated[
        bool, typer.Option("--network/--no-network", help="Verify API keys authenticate.")
    ] = True,
) -> None:
    """Verify keys, storage, dependencies and quota before a run."""
    settings = _settings()
    with console.status("running checks..."):
        checks = run_checks(settings, network=network)
    _render_checks(checks)

    failures = [c for c in checks if c.blocking]
    warnings = [c for c in checks if c.status is Status.WARN]
    pending = [c for c in checks if c.status is Status.PENDING]

    console.print()
    summary = (
        f"{len(checks) - len(failures) - len(warnings) - len(pending)} ok, "
        f"{len(warnings)} warn, {len(failures)} fail, {len(pending)} not yet built"
    )
    if failures:
        console.print(Text(f"not ready: {summary}", style="bold red"))
        raise typer.Exit(1)
    console.print(Text(f"ready: {summary}", style="bold green"))


@app.command()
def quota(
    reset: Annotated[
        str | None,
        typer.Option("--reset", help="Clear usage for a model key, or 'all'."),
    ] = None,
) -> None:
    """Show remaining free-tier allowance per model."""
    settings = _settings()
    ledger = QuotaLedger(settings.paths.ensure().quota_file)

    if reset is not None:
        ledger.reset(None if reset == "all" else reset)
        console.print(f"[green]reset[/green] {reset}")
        return

    router = Router(settings, ledger)
    table = Table(title="free tier remaining", title_justify="left")
    table.add_column("workload", style="bold")
    table.add_column("model")
    table.add_column("used today", justify="right")
    table.add_column("status")

    for workload in Workload:
        for spec, status in router.candidates(workload):
            used, tokens, audio = ledger.usage(spec.key)
            if spec.local:
                usage_cell = Text("local", style="dim")
                state = Text("unlimited", style="green")
            else:
                cap = spec.limits.requests_per_day
                usage_cell = Text(f"{used}/{cap}" if cap else str(used))
                if status.allowed:
                    state = Text("available", style="green")
                else:
                    state = Text(status.reason or "unavailable", style="yellow")
            table.add_row(str(workload), spec.model, usage_cell, state)
    console.print(table)


@app.command()
def route(
    workload: Annotated[
        Workload, typer.Argument(help="Which workload to resolve.")
    ] = Workload.TEXT,
) -> None:
    """Show which model would serve a workload right now, and why."""
    settings = _settings()
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    router = Router(settings, ledger)

    for spec, status in router.candidates(workload):
        mark = "[green]->[/green]" if status.allowed else "[dim]  [/dim]"
        reason = "" if status.allowed else f" [yellow]{status.reason}[/yellow]"
        console.print(f"{mark} {spec.key}{reason}")
        if spec.notes:
            console.print(f"     [dim]{spec.notes}[/dim]")

    selection = router.select(workload)
    console.print()
    console.print(f"[bold green]selected[/bold green] {selection.key}")


@app.command()
def models() -> None:
    """List the routing table."""
    table = Table(title="routing table", title_justify="left")
    table.add_column("workload", style="bold")
    table.add_column("#", justify="right")
    table.add_column("model")
    table.add_column("free allowance")

    for workload, specs in ROUTING_TABLE.items():
        for i, spec in enumerate(specs, start=1):
            limits = spec.limits
            if spec.local:
                allowance = "local, unlimited"
            else:
                parts = []
                if limits.requests_per_day:
                    parts.append(f"{limits.requests_per_day:,}/day")
                if limits.requests_per_minute:
                    parts.append(f"{limits.requests_per_minute}/min")
                if limits.audio_seconds_per_day:
                    parts.append(f"{limits.audio_seconds_per_day / 3600:.0f}h audio/day")
                allowance = ", ".join(parts)
            table.add_row(str(workload) if i == 1 else "", str(i), spec.key, allowance)
    console.print(table)


@trace_app.command("list")
def trace_list(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """List recent sessions, newest first."""
    settings = _settings()
    paths = list_traces(settings.paths.traces_dir, limit=limit)
    if not paths:
        console.print("[dim]no traces yet[/dim]")
        return

    table = Table(box=None)
    table.add_column("session", style="bold")
    table.add_column("events", justify="right")
    table.add_column("status")

    for path in paths:
        events = read_trace(path)
        end = next((e for e in reversed(events) if e["kind"] == "session.end"), None)
        state = (end or {}).get("payload", {}).get("status", "incomplete")
        style = {"ok": "green", "error": "red"}.get(state, "yellow")
        table.add_row(path.stem, str(len(events)), Text(state, style=style))
    console.print(table)


@trace_app.command("show")
def trace_show(
    session: Annotated[str, typer.Argument(help="Session id, or 'last'.")] = "last",
) -> None:
    """Print a session's events in order."""
    settings = _settings()
    traces = list_traces(settings.paths.traces_dir, limit=500)
    if not traces:
        console.print("[dim]no traces yet[/dim]")
        raise typer.Exit(1)

    if session == "last":
        path = traces[0]
    else:
        path = next((p for p in traces if p.stem == session or session in p.stem), None)
        if path is None:
            err_console.print(f"[red]no such session:[/red] {session}")
            raise typer.Exit(1)

    for event in read_trace(path):
        duration = event.get("duration_ms")
        suffix = f" [dim]{duration:.0f}ms[/dim]" if duration else ""
        console.print(f"[dim]{event['seq']:>3}[/dim] [bold]{event['kind']}[/bold]{suffix}")
        for key, value in (event.get("payload") or {}).items():
            console.print(f"      [cyan]{key}[/cyan]: {value}")


def main() -> None:
    """Console script entry point with tidy handling for expected failures."""
    try:
        app()
    except VictorError as exc:
        err_console.print(f"[bold red]error:[/bold red] {exc}")
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        err_console.print("[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
