"""Command line entry point."""

from __future__ import annotations

import contextlib
import platform
import sys
from pathlib import Path
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
from .tracing import Trace, list_traces, read_trace

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="victor",
    help="Voice-driven computer-use agent for Windows.",
    add_completion=False,
)
trace_app = typer.Typer(help="Inspect session traces.", no_args_is_help=True)
app.add_typer(trace_app, name="trace")
voice_app = typer.Typer(help="Voice devices and models.", no_args_is_help=True)
app.add_typer(voice_app, name="voice")
journal_app = typer.Typer(help="Review and reverse past actions.", no_args_is_help=True)
app.add_typer(journal_app, name="journal")

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


# --- voice ----------------------------------------------------------------


@voice_app.command("devices")
def voice_devices() -> None:
    """List audio input and output devices."""
    from .voice import list_devices

    devices = list_devices()
    if not devices:
        err_console.print(
            "[yellow]no audio devices[/yellow] - install with: pip install -e '.[voice]'"
        )
        raise typer.Exit(1)

    table = Table(box=None)
    table.add_column("#", justify="right")
    table.add_column("name", style="bold")
    table.add_column("in", justify="right")
    table.add_column("out", justify="right")
    table.add_column("default")

    for d in devices:
        marks = []
        if d["default_input"]:
            marks.append("input")
        if d["default_output"]:
            marks.append("output")
        table.add_row(
            str(d["index"]),
            str(d["name"]),
            str(d["inputs"]),
            str(d["outputs"]),
            Text(", ".join(marks), style="green"),
        )
    console.print(table)


@voice_app.command("install")
def voice_install(
    voice: Annotated[str, typer.Option("--voice", help="Piper voice name.")] = "",
    force: Annotated[bool, typer.Option("--force", help="Re-download.")] = False,
) -> None:
    """Download the Piper voice model (~63 MB, once)."""
    from .voice import DEFAULT_VOICE, PiperSynthesizer

    settings = _settings()
    name = voice or DEFAULT_VOICE
    synth = PiperSynthesizer(settings.paths.ensure().models_dir, name)

    if synth.installed and not force:
        console.print(f"[green]already installed[/green] {name} -> {synth.model_path}")
        return

    with console.status(f"downloading {name}..."):
        path = synth.ensure_installed(force=force)
    size_mb = path.stat().st_size / 1e6
    console.print(f"[green]installed[/green] {name} ({size_mb:.0f} MB) -> {path}")


@app.command()
def say(
    text: Annotated[str, typer.Argument(help="What to say.")],
    backend: Annotated[
        str, typer.Option("--backend", help="piper | system | null")
    ] = "piper",
    quiet: Annotated[
        bool, typer.Option("--quiet", help="Synthesize without playing.")
    ] = False,
) -> None:
    """Speak a line of text through the local synthesizer."""
    from .voice import Player, build_synthesizer
    from .voice import speak as do_speak

    settings = _settings()
    synth = build_synthesizer(settings.paths.ensure().models_dir, prefer=backend)
    stats = do_speak(synth, Player(enabled=not quiet), text)
    console.print(
        f"[dim]{stats.backend}: {stats.audio_seconds:.2f}s audio, "
        f"first sound in {stats.ttfa_ms:.0f}ms, "
        f"synth {stats.synth_ms:.0f}ms (rtf {stats.realtime_factor:.2f})[/dim]"
    )


@app.command()
def listen(
    mode: Annotated[
        str, typer.Option("--mode", help="vad | ptt | fixed")
    ] = "vad",
    seconds: Annotated[
        float, typer.Option("--seconds", help="Recording length in fixed mode.")
    ] = 5.0,
    reply: Annotated[
        bool, typer.Option("--reply/--no-reply", help="Speak an acknowledgement back.")
    ] = True,
) -> None:
    """Record one utterance, transcribe it, and read it back.

    P1 has no agent behind it yet, so the reply is an acknowledgement rather
    than an answer. It exists to prove the full mic -> STT -> TTS round trip.
    """
    from .voice import ListenMode, NoSpeechDetected, build_pipeline

    settings = _settings()
    with Trace.open(settings.paths.ensure().traces_dir, label="listen") as trace:
        pipeline = build_pipeline(settings, trace=trace)
        try:
            warm_ms = pipeline.warm() * 1000
            if warm_ms:
                console.print(f"[dim]voice model loaded in {warm_ms:.0f}ms[/dim]")

            listen_mode = ListenMode(mode)
            if listen_mode is ListenMode.PTT:
                console.print("[bold]recording[/bold] - press Enter to stop")
            else:
                console.print("[bold]listening[/bold] - start speaking")

            try:
                turn = pipeline.listen(listen_mode, seconds=seconds)
            except NoSpeechDetected:
                err_console.print("[yellow]no speech detected[/yellow]")
                raise typer.Exit(4) from None

            console.print()
            console.print(f"[bold cyan]heard:[/bold cyan] {turn.text or '(nothing)'}")
            console.print(
                f"[dim]{turn.endpoint.segment.duration:.2f}s audio "
                f"({turn.endpoint.reason}), stt {turn.transcript.latency_ms:.0f}ms, "
                f"{turn.transcript.model}[/dim]"
            )

            if reply and turn.text:
                stats = pipeline.speak(f"You said: {turn.text}")
                console.print(f"[dim]spoke in {stats.ttfa_ms:.0f}ms to first sound[/dim]")
        finally:
            pipeline.close()


# --- agent ----------------------------------------------------------------


def _render_safety_summary(agent: object) -> None:
    """Print what the safety layer did, if it was in play."""
    interceptor = getattr(getattr(agent, "registry", None), "interceptor", None)
    stats = getattr(interceptor, "stats", None)
    if stats is None or stats.reviewed == 0:
        return
    if stats.confirmed or stats.refused or stats.denied or stats.dry_run:
        console.print(f"[dim]safety: {stats.summary()}[/dim]")


def _render_step(step: object) -> None:
    """Print one think-act cycle as it happens."""
    for call, result in getattr(step, "calls", ()):
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        console.print(f"  [dim]{mark}[/dim] [cyan]{call}[/cyan]")
        detail = (result.error or result.output).strip().splitlines()
        if detail:
            console.print(f"       [dim]{detail[0][:100]}[/dim]")


@app.command(name="do")
def do_task(
    task: Annotated[str, typer.Argument(help="What you want done.")],
    steps: Annotated[int, typer.Option("--steps", help="Maximum think-act cycles.")] = 8,
    speak_reply: Annotated[
        bool, typer.Option("--speak", help="Read the answer aloud.")
    ] = False,
    show_steps: Annotated[
        bool, typer.Option("--steps-visible/--quiet-steps", help="Print tool calls.")
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview actions without running them.")
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Approve every action without asking. Use with care."),
    ] = False,
    desktop: Annotated[
        bool,
        typer.Option("--desktop", help="Let the agent click and type on screen."),
    ] = False,
    app_name: Annotated[
        str | None,
        typer.Option("--app", help="Point the desktop tools at one application."),
    ] = None,
) -> None:
    """Run one task through the agent and print the answer."""
    from .agent import build_agent
    from .safety import AutoConfirmer, SignalKillSwitch

    settings = _settings()
    if dry_run:
        settings = settings.model_copy(update={"dry_run": True})

    with (
        Trace.open(settings.paths.ensure().traces_dir, label="do") as trace,
        SignalKillSwitch() as switch,
    ):
        agent = build_agent(
            settings,
            trace=trace,
            max_steps=steps,
            kill_switch=switch,
            confirmer=AutoConfirmer(True) if yes else None,
            desktop=desktop or None,
            app=app_name,
        )
        if show_steps:
            agent.on_step = _render_step
        try:
            if settings.dry_run:
                console.print(
                    "[yellow]dry run:[/yellow] actions will be previewed, not executed"
                )
            if yes:
                console.print(
                    "[yellow]--yes:[/yellow] every action is pre-approved. "
                    "Irreversible commands are still refused."
                )
            if desktop:
                console.print(
                    "[yellow]--desktop:[/yellow] the agent can click and type on "
                    "your screen. Ctrl-C stops it mid-click."
                )
            console.print("[dim]Ctrl-C stops the run.[/dim]")
            result = agent.run(task)
        finally:
            agent.close()

    _render_safety_summary(agent)

    console.print()
    style = "bold green" if result.ok else "bold yellow"
    console.print(Text(result.answer or "(no answer)", style=style))
    console.print(f"[dim]{result.summary()}[/dim]")

    if speak_reply and result.answer:
        from .voice import Player, build_synthesizer
        from .voice import speak as do_speak

        synth = build_synthesizer(settings.paths.models_dir)
        do_speak(synth, Player(), result.answer)

    if not result.ok:
        raise typer.Exit(1)


@app.command()
def converse(
    turns: Annotated[int, typer.Option("--turns", help="How many exchanges before exiting.")] = 0,
    mode: Annotated[str, typer.Option("--mode", help="vad | ptt")] = "vad",
    steps: Annotated[int, typer.Option("--steps")] = 8,
) -> None:
    """Hold a spoken conversation: listen, think, act, reply out loud.

    ``--turns 0`` means keep going until interrupted.
    """
    from .agent import STT_PROMPT, build_agent
    from .safety import SignalKillSwitch, SpokenConfirmer, is_stop_phrase
    from .voice import ListenMode, NoSpeechDetected, build_pipeline

    settings = _settings()
    listen_mode = ListenMode(mode)

    with Trace.open(settings.paths.ensure().traces_dir, label="converse") as trace:
        pipeline = build_pipeline(settings, trace=trace)
        switch = SignalKillSwitch()
        # Destructive actions are confirmed out loud, since the user's hands
        # are the reason they are talking to a computer in the first place.
        agent = build_agent(
            settings,
            trace=trace,
            max_steps=steps,
            voice=True,
            kill_switch=switch,
            confirmer=SpokenConfirmer(pipeline),
        )
        agent.on_step = _render_step
        try:
            warm_ms = pipeline.warm() * 1000
            console.print(
                f"[dim]voice ready in {warm_ms:.0f}ms. "
                "Say 'stop' or press Ctrl-C to abort.[/dim]"
            )

            turn_index = 0
            while turns == 0 or turn_index < turns:
                turn_index += 1
                switch.reset()
                console.print()
                if listen_mode is ListenMode.PTT:
                    console.print("[bold]recording[/bold] - press Enter to stop")
                else:
                    console.print("[bold]listening[/bold]")

                try:
                    heard = pipeline.listen(listen_mode, prompt=STT_PROMPT)
                except NoSpeechDetected:
                    console.print("[dim]nothing heard, still listening[/dim]")
                    continue

                if not heard.text:
                    continue
                console.print(f"[bold cyan]you:[/bold cyan] {heard.text}")

                if is_stop_phrase(heard.text):
                    switch.trip("spoken 'stop'")
                    pipeline.speak("Stopped.")
                    console.print("[yellow]stopped[/yellow]")
                    continue

                result = agent.run(heard.text)
                console.print(f"[bold green]victor:[/bold green] {result.answer}")
                if result.answer:
                    pipeline.speak(result.answer)
                _render_safety_summary(agent)
        except KeyboardInterrupt:
            console.print("\n[dim]stopped[/dim]")
        finally:
            switch.restore()
            pipeline.close()
            agent.close()


@journal_app.command("list")
def journal_list(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Show recent actions and whether they can be reversed."""
    from .safety import ActionJournal

    settings = _settings()
    entries = ActionJournal(settings.paths.ensure().journal_file).recent(limit)
    if not entries:
        console.print("[dim]no actions recorded yet[/dim]")
        return

    table = Table(box=None)
    table.add_column("id", style="bold")
    table.add_column("when", style="dim")
    table.add_column("action", overflow="fold")
    table.add_column("outcome")
    table.add_column("undo")

    for entry in entries:
        from .safety import summarise_call

        if entry.undone_at:
            undo_cell = Text("undone", style="dim")
        elif entry.undo is not None:
            undo_cell = Text(entry.undo.description, style="green")
        else:
            undo_cell = Text(entry.no_undo_reason or "no", style="yellow")

        if entry.decision == "deny":
            outcome = Text("blocked", style="red")
        elif entry.ok:
            outcome = Text("ran", style="green")
        else:
            outcome = Text("failed", style="yellow")

        table.add_row(
            entry.id,
            entry.ts.replace("T", " ").replace("+00:00", ""),
            summarise_call(entry.tool, entry.arguments),
            outcome,
            undo_cell,
        )
    console.print(table)


@journal_app.command("undo")
def journal_undo(
    entry_id: Annotated[
        str, typer.Argument(help="Entry id from `victor journal list`, or 'last'.")
    ] = "last",
) -> None:
    """Reverse a recorded action, if it has an exact inverse."""
    from .safety import ActionJournal, AutoConfirmer, SafetyInterceptor, undo_entry
    from .tools import build_registry

    settings = _settings()
    journal = ActionJournal(settings.paths.ensure().journal_file)

    if entry_id == "last":
        entry = journal.last_reversible()
        if entry is None:
            console.print("[yellow]nothing recent can be undone[/yellow]")
            console.print(
                "[dim]Deletes and network calls have no inverse. "
                "See `victor journal list` for why.[/dim]"
            )
            raise typer.Exit(1)
    else:
        entry = journal.get(entry_id)
        if entry is None:
            err_console.print(f"[red]no such entry:[/red] {entry_id}")
            raise typer.Exit(1)

    if entry.undo is None:
        err_console.print(f"[yellow]cannot undo:[/yellow] {entry.no_undo_reason}")
        raise typer.Exit(1)

    console.print(f"This will {entry.undo.description}.")
    # The undo itself is an action, so it goes through the same gate. Approving
    # it here is the confirmation; asking twice would be theatre.
    interceptor = SafetyInterceptor(confirmer=AutoConfirmer(True))
    registry = build_registry(settings, interceptor=interceptor)
    result = undo_entry(journal, registry, entry)

    if result.error:
        err_console.print(f"[red]undo failed:[/red] {result.error}")
        raise typer.Exit(1)
    console.print(f"[green]undone[/green] {entry.id}: {entry.undo.description}")


@app.command()
def run(
    text: Annotated[
        str | None,
        typer.Option("--text", help="Drive by typed prompt instead of the microphone."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Full loop, nothing executes.")
    ] = False,
    steps: Annotated[int, typer.Option("--steps")] = 8,
    mode: Annotated[str, typer.Option("--mode", help="vad | ptt")] = "vad",
    yes: Annotated[bool, typer.Option("--yes", help="Approve every action.")] = False,
) -> None:
    """Start the agent.

    With ``--text`` it runs one typed task and exits; without, it opens the
    push-to-talk voice loop. The typed form is the one used constantly during
    development, so voice is never on the critical path.
    """
    if text is not None:
        do_task(
            task=text,
            steps=steps,
            speak_reply=False,
            show_steps=True,
            dry_run=dry_run,
            yes=yes,
        )
        return

    if dry_run:
        console.print("[yellow]dry run:[/yellow] actions will be previewed, not executed")
    converse(turns=0, mode=mode, steps=steps)


@app.command()
def undo(
    last: Annotated[
        int, typer.Option("--last", "-n", help="How many recent actions to reverse.")
    ] = 1,
) -> None:
    """Reverse recent actions. Alias for `victor journal undo`."""
    from .safety import ActionJournal, AutoConfirmer, SafetyInterceptor
    from .safety import undo_last as reverse_last
    from .tools import build_registry

    settings = _settings()
    journal = ActionJournal(settings.paths.ensure().journal_file)
    interceptor = SafetyInterceptor(confirmer=AutoConfirmer(True))
    registry = build_registry(settings, interceptor=interceptor)

    results = reverse_last(journal, registry, last)
    if not results:
        console.print("[yellow]nothing recent can be undone[/yellow]")
        console.print(
            "[dim]Deletes go to the trash and can be restored; network calls "
            "cannot. `victor journal list` says which is which.[/dim]"
        )
        raise typer.Exit(1)

    failed = False
    for result in results:
        if result.error:
            err_console.print(f"[red]undo failed:[/red] {result.error}")
            failed = True
        else:
            console.print(f"[green]undone[/green] {result.entry.id}: {result.steps[0]}")
    if failed:
        raise typer.Exit(1)


@app.command()
def sessions(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """List recorded sessions. Alias for `victor trace list`."""
    trace_list(limit=limit)


@app.command()
def replay(
    session: Annotated[str, typer.Argument(help="Session id, or 'last'.")] = "last",
) -> None:
    """Step through a recorded session. Alias for `victor trace show`."""
    trace_show(session=session)


@app.command(name="install-shim")
def install_shim(
    directory: Annotated[
        str | None,
        typer.Option("--dir", help="Where to write the shim. Defaults to a PATH entry."),
    ] = None,
) -> None:
    """Put `victor` on the global PATH without activating the venv."""
    import os
    import stat
    import sysconfig

    interpreter = Path(sys.executable)
    is_windows = platform.system() == "Windows"

    if directory:
        target_dir = Path(directory)
    elif is_windows:
        # The plan's target: the base Python's Scripts dir, already on PATH.
        target_dir = Path(sysconfig.get_path("scripts", "nt_user"))
    else:
        target_dir = Path.home() / ".local" / "bin"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        err_console.print(f"[red]cannot create {target_dir}:[/red] {exc}")
        raise typer.Exit(1) from exc

    if is_windows:
        shim = target_dir / "victor.cmd"
        shim.write_text(f'@echo off\r\n"{interpreter}" -m victor %*\r\n', encoding="utf-8")
    else:
        shim = target_dir / "victor"
        shim.write_text(f'#!/bin/sh\nexec "{interpreter}" -m victor "$@"\n', encoding="utf-8")
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    console.print(f"[green]installed[/green] {shim}")
    console.print(f"[dim]runs {interpreter} -m victor[/dim]")

    on_path = str(target_dir) in os.environ.get("PATH", "").split(os.pathsep)
    if not on_path:
        console.print(
            f"[yellow]note:[/yellow] {target_dir} is not on your PATH. "
            "Add it, or pass --dir pointing somewhere that is."
        )


@app.command()
def check(
    command: Annotated[str, typer.Argument(help="A shell command to classify.")],
) -> None:
    """Show how the safety layer would classify a command, without running it."""
    from .safety import Risk, classify_shell

    verdict = classify_shell(command)
    colour = {Risk.SAFE: "green", Risk.CONFIRM: "yellow", Risk.DENY: "bold red"}[verdict.risk]
    console.print(f"[{colour}]{verdict.risk}[/{colour}]  {verdict.reason}")
    if verdict.trigger and verdict.trigger != command:
        console.print(f"[dim]triggered by: {verdict.trigger}[/dim]")

    if verdict.risk is not Risk.SAFE:
        from .safety import plan_undo

        undo, why_not = plan_undo("shell", {"command": command})
        if undo is not None:
            console.print(f"[dim]undo available: {undo.description}[/dim]")
        else:
            console.print(f"[dim]cannot be undone: {why_not}[/dim]")


@app.command()
def uia(
    dump: Annotated[
        bool, typer.Option("--dump", help="Print the focused window's element tree.")
    ] = True,
    demo: Annotated[
        bool, typer.Option("--demo", help="Use the built-in fake tree instead of a real window.")
    ] = False,
    app: Annotated[
        str | None,
        typer.Option("--app", help="Target an application by name instead of the frontmost."),
    ] = None,
    apps: Annotated[
        bool, typer.Option("--apps", help="List applications that can be targeted.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Elements to show.")] = 60,
    refresh: Annotated[bool, typer.Option("--refresh", help="Bypass the cache.")] = False,
) -> None:
    """Print the focused window's element tree. Zero API calls.

    This is the project's central claim made checkable: everything printed here
    came from the operating system, locally, for free. Nothing was sent
    anywhere and no quota was spent.
    """
    from .desktop import (
        FakeBackend,
        PerceptionUnavailable,
        TreeReader,
        demo_tree,
        list_applications,
    )

    if apps:
        names = list_applications()
        if not names:
            # "no applications (or not supported)" told you nothing about which
            # of the two it was. Both platforms can answer this now, so an empty
            # list on either really does mean nothing is open.
            if platform.system() not in ("Darwin", "Windows"):
                console.print(
                    f"[dim]listing windows is not supported on {platform.system()}[/dim]"
                )
            else:
                console.print("[dim]no windows are open to target[/dim]")
            return
        for name in names:
            console.print(f"  {name}")
        console.print(f"\n[dim]target one with: victor uia --app '{names[0]}'[/dim]")
        return

    if not dump:
        console.print("[dim]--dump is the only mode so far; it is implied.[/dim]")

    reader = TreeReader(FakeBackend(demo_tree())) if demo else TreeReader(app=app)

    ok, detail = reader.available()
    if not ok:
        err_console.print(f"[yellow]screen perception unavailable:[/yellow] {detail}")
        err_console.print(
            "[dim]Try `victor uia --demo` to see the output shape on any platform.[/dim]"
        )
        raise typer.Exit(5)

    try:
        snapshot = reader.snapshot(refresh=refresh)
    except PerceptionUnavailable as exc:
        err_console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(5) from exc

    console.print(snapshot.render(limit=limit))
    console.print()
    console.print(
        f"[dim]{len(snapshot)} elements in {snapshot.duration_ms:.0f}ms "
        f"via {snapshot.backend} - 0 API calls, 0 quota spent[/dim]"
    )
    if snapshot.truncated:
        console.print(
            "[yellow]the walk hit its limit[/yellow] - raise --limit or narrow the window"
        )


# --- desktop actuation ----------------------------------------------------


def _open_desktop(app_name: str | None):
    """A ready-to-use desktop, or a printed reason and a non-zero exit."""
    from .desktop import Desktop

    desktop = Desktop(app=app_name)
    ok, detail = desktop.available()
    if not ok:
        err_console.print(f"[yellow]desktop control unavailable:[/yellow] {detail}")
        raise typer.Exit(5)
    return desktop


def _gated_desktop_tools(desktop, *, yes: bool = False):
    """The desktop tools, behind the same gate the agent gets.

    ``victor click`` used to call :meth:`Desktop.click` directly, so it reached
    execution without a classifier, a confirmation or a journal entry. That is
    the same shape of hole P5 closed for terminals - an actuation path around
    the safety layer - and it matters more on Windows, where UI Automation's
    Invoke on a file opens it: a click on an installer would have run it.

    So the CLI now builds the same interceptor and journal the agent builds. It
    is a manual driver, but "a person typed it" is not the same as "a person
    understood what it would do", and the classifier is the part that knows the
    difference between notes.txt and setup.exe.
    """
    from .safety import ActionJournal, AutoConfirmer, SafetyInterceptor, build_confirmer
    from .tools import ToolRegistry
    from .tools.desktop import build_desktop_tools

    settings = _settings()
    journal = ActionJournal(settings.paths.ensure().journal_file, session="cli")
    interceptor = SafetyInterceptor(
        confirmer=AutoConfirmer(True) if yes else build_confirmer(),
        journal=journal,
        require_confirmation=settings.confirm_destructive and not yes,
    )
    registry = ToolRegistry(interceptor)
    for tool in build_desktop_tools(desktop=desktop):
        registry.register(tool)
    return registry, interceptor


@app.command()
def click(
    target: Annotated[
        str | None,
        typer.Argument(help="Part of the element's label. Omit if you pass --index."),
    ] = None,
    index: Annotated[
        int | None, typer.Option("--index", "-i", help="Click this element index instead.")
    ] = None,
    app_name: Annotated[
        str | None, typer.Option("--app", help="Target an application by name.")
    ] = None,
    right: Annotated[bool, typer.Option("--right", help="Right-click.")] = False,
    double: Annotated[bool, typer.Option("--double", help="Double-click.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Say what would be clicked, and stop.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip the confirmation prompt. Use with care.")
    ] = False,
) -> None:
    """Click one control, by label or index. Zero API calls.

    The manual counterpart to ``victor uia``: that one proves Victor can see a
    window, this one proves it can act on what it saw, and neither sends
    anything anywhere. Between them they are the whole perception-actuation
    loop with the model taken out, which is what makes them the right thing to
    run when checking Victor on a new machine.

    Gated exactly as the agent is: consequential clicks ask first, and every
    one is journalled.
    """
    desktop = _open_desktop(app_name)

    if index is None and not target:
        err_console.print("[yellow]name part of a label, or pass --index[/yellow]")
        raise typer.Exit(2)

    snapshot = desktop.snapshot(refresh=True)
    if index is None:
        matches = [e for e in snapshot.find(str(target)) if e.actionable]
        if not matches:
            err_console.print(f"[yellow]nothing clickable matches {target!r}[/yellow]")
            console.print(f"[dim]{len(snapshot)} elements in {snapshot.window_title}[/dim]")
            raise typer.Exit(1)
        if len(matches) > 1:
            # Refusing to pick is the same rule the agent path follows: an
            # ambiguous target is a question, not a coin toss.
            console.print(f"[yellow]{len(matches)} elements match {target!r}:[/yellow]")
            for element in matches[:10]:
                console.print(f"  {element.render()}")
            console.print("\n[dim]pick one with --index[/dim]")
            raise typer.Exit(1)
        chosen = matches[0]
    else:
        chosen = snapshot.by_index(index)
        if chosen is None:
            err_console.print(f"[yellow]there is no element {index}[/yellow]")
            raise typer.Exit(1)

    console.print(f"[dim]{snapshot.window_title}[/dim]")
    console.print(f"  {chosen.render()}")

    if dry_run:
        from .safety.classify import classify

        verdict = classify("click", {"index": chosen.index, "label": chosen.label}, mutating=True)
        console.print(f"[dim]{verdict.risk}:[/dim] {verdict.reason}")
        console.print("[yellow]dry run:[/yellow] nothing was clicked")
        return

    registry, _ = _gated_desktop_tools(desktop, yes=yes)
    outcome = registry.run(
        "click",
        {
            "index": chosen.index,
            "label": chosen.label,
            "button": "right" if right else "left",
            "double": double,
        },
    )
    if not outcome.ok:
        err_console.print(f"[red]{outcome.error}[/red]")
        raise typer.Exit(1)
    method = outcome.metadata.get("method") or "?"
    console.print(f"[green]{outcome.output}[/green] [dim]via {method}[/dim]")


@app.command()
def press(
    keys: Annotated[str, typer.Argument(help="A chord, e.g. 'mod+s' or 'mod+a delete'.")],
    app_name: Annotated[
        str | None, typer.Option("--app", help="Focus this application first.")
    ] = None,
    show: Annotated[
        bool, typer.Option("--vocabulary", help="List the key names that work on both platforms.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip the confirmation prompt. Use with care.")
    ] = False,
) -> None:
    """Press a keyboard shortcut. ``mod`` is Ctrl on Windows and Command on macOS.

    Gated as the agent is: shortcuts that discard work ask first, typing into a
    terminal is refused, and each press is journalled.
    """
    from .desktop.keys import UnknownKey, known_keys, parse_sequence

    if show:
        console.print(", ".join(known_keys()))
        console.print(
            "\n[dim]modifiers: ctrl, alt/option, shift, cmd/win - and mod, which is "
            "whichever of ctrl and cmd this platform uses for shortcuts[/dim]"
        )
        return

    try:
        chords = parse_sequence(keys)
    except UnknownKey as exc:
        err_console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(2) from exc

    desktop = _open_desktop(app_name)
    if app_name:
        focused = desktop.focus_app(app_name)
        if not focused.ok:
            err_console.print(f"[yellow]{focused.detail}[/yellow]")
            raise typer.Exit(1)

    console.print(f"[dim]pressing {' then '.join(str(c) for c in chords)}[/dim]")
    registry, _ = _gated_desktop_tools(desktop, yes=yes)
    outcome = registry.run("press_keys", {"keys": keys})
    if not outcome.ok:
        err_console.print(f"[red]{outcome.error}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{outcome.output}[/green]")


# --- memory ---------------------------------------------------------------


def _warm_embedder(memory) -> None:
    """Load the embedding model now, announcing a first-run download."""
    warm = getattr(memory.embedder, "warm", None)
    if warm is None:
        return
    cached = any(memory.store.directory.parent.glob("models/**/*.onnx"))
    if not cached:
        console.print(
            "[dim]first run: downloading the embedding model (~130 MB, once). "
            "It stays local and nothing is sent anywhere.[/dim]"
        )
    warm()


def _open_memory():
    """Victor's memory, or a printed reason and a non-zero exit."""
    from .rag import build_memory
    from .rag.store import EmbedderChanged

    try:
        return build_memory(_settings())
    except EmbedderChanged as exc:
        err_console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(6) from exc


@app.command(name="index")
def index_command(
    path: Annotated[
        Path | None, typer.Argument(help="File or directory to index. Defaults to the cwd.")
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Re-encode everything with the current embedder."),
    ] = False,
) -> None:
    """Read project files into memory. Local, and costs no quota."""
    from .rag import build_memory, index_path, select_embedder
    from .rag.store import EmbedderChanged, VectorStore

    settings = _settings()
    paths = settings.paths.ensure()

    if rebuild:
        embedder = select_embedder(paths.models_dir)
        # Open against whatever the store already declares, so the guard does
        # not fire on the very command that exists to clear it.
        with sqlite_embedder(paths.memory_dir) as (name, dimensions):
            store = VectorStore(paths.memory_dir, embedder_name=name, dimensions=dimensions)
        from .rag.recall import Memory

        memory = Memory(store, embedder)
        with console.status(f"re-encoding with {embedder.name}..."):
            count = memory.rebuild(embedder)
        console.print(f"[green]re-encoded {count} records[/green] with {embedder.name}")
        console.print(f"[dim]{memory.describe()}[/dim]")
        return

    try:
        memory = build_memory(settings)
    except EmbedderChanged as exc:
        err_console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(6) from exc

    root = Path(path or Path.cwd())
    if not root.exists():
        err_console.print(f"[yellow]{root} does not exist[/yellow]")
        raise typer.Exit(2)

    # Warm the embedder before the spinner starts. The first run downloads a
    # ~130 MB ONNX model, and its progress bar fighting a status spinner looks
    # like a hang - which is exactly when a user needs to see progress most.
    _warm_embedder(memory)

    seen: list[Path] = []
    with console.status(f"indexing {root}..."):
        files, stored = index_path(memory, root, on_file=seen.append)

    console.print(f"[green]{files} files read, {stored} new chunks stored[/green]")
    if files and not stored:
        console.print("[dim]nothing new - every chunk was already known[/dim]")
    console.print(f"[dim]{memory.describe()} - 0 API calls, 0 quota spent[/dim]")


@app.command()
def recall(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many to show.")] = 5,
    kind: Annotated[
        str | None, typer.Option("--kind", help="Restrict to 'fix', 'file' or 'note'.")
    ] = None,
    all_scores: Annotated[
        bool,
        typer.Option("--all", help="Show near misses below the relevance floor too."),
    ] = False,
) -> None:
    """Search memory. Offline, instant, and free."""
    import time

    memory = _open_memory()
    started = time.perf_counter()
    found = memory.recall(query, k=limit, kind=kind, threshold=0.0 if all_scores else None)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if not found.found:
        console.print("[dim]nothing above the relevance floor[/dim]")
        console.print(
            f"[dim]{len(memory.store)} records searched in {elapsed_ms:.0f}ms. "
            "Try --all to see near misses.[/dim]"
        )
        return

    for hit in found.hits:
        marker = "[green]" if hit.score >= memory.threshold else "[dim]"
        console.print(f"{marker}{hit.score:.2f}[/] {hit.record.kind:<5} {hit.record.summary}")
        fix = str(hit.record.meta.get("fix", "")).strip()
        if fix:
            for line in fix.splitlines():
                console.print(f"      [cyan]{line}[/cyan]")
        elif hit.record.source:
            console.print(f"      [dim]{hit.record.source}[/dim]")

    console.print(
        f"\n[dim]{len(found.hits)} of {len(memory.store)} records in {elapsed_ms:.0f}ms "
        "- 0 API calls, 0 quota spent[/dim]"
    )


@app.command(name="memory")
def memory_command(
    clear: Annotated[
        bool, typer.Option("--clear", help="Forget everything. Cannot be undone.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Recent records to list.")] = 10,
) -> None:
    """What Victor remembers."""
    memory = _open_memory()

    if clear:
        count = len(memory.store)
        if not typer.confirm(f"Forget all {count} records?"):
            console.print("[dim]left alone[/dim]")
            return
        memory.store.clear()
        console.print(f"[green]forgot {count} records[/green]")
        return

    console.print(memory.describe())
    if not len(memory.store):
        console.print(
            "\n[dim]nothing remembered yet. Memory fills itself: when a command "
            "fails and you later make it pass, the pair is stored. "
            "`victor index` adds project files too.[/dim]"
        )
        return

    table = Table(title="most recent", title_justify="left")
    table.add_column("kind")
    table.add_column("what", overflow="fold")
    table.add_column("when", style="dim")
    for record in memory.store.recent(limit):
        table.add_row(record.kind, record.summary, record.created)
    console.print(table)


@contextlib.contextmanager
def sqlite_embedder(directory: Path):
    """Read back which embedder a store was built with, without opening it."""
    import sqlite3

    db = directory / "memory.sqlite3"
    if not db.exists():
        yield ("hash", 512)
        return
    connection = sqlite3.connect(db)
    try:
        rows = dict(connection.execute("SELECT key, value FROM store_meta").fetchall())
        yield (rows.get("embedder", "hash"), int(rows.get("dimensions", 512)))
    finally:
        connection.close()


@app.command()
def tools() -> None:
    """List the tools the agent can call."""
    from .tools import build_registry

    settings = _settings()
    registry = build_registry(settings)

    table = Table(title="tools", title_justify="left")
    table.add_column("name", style="bold")
    table.add_column("mutating")
    table.add_column("description", overflow="fold")

    for tool in registry:
        flag = (
            Text("yes", style="yellow") if tool.spec.mutating else Text("no", style="green")
        )
        table.add_row(tool.spec.name, flag, tool.spec.description)
    console.print(table)
    console.print(
        "\n[dim]Mutating calls are classified before they run: read-only commands pass "
        "silently, writes ask for confirmation, and irreversible ones are refused. "
        "Try `victor check '<command>'` to see a verdict without running anything.[/dim]"
    )


# --- benchmarks -----------------------------------------------------------


@app.command()
def bench(
    voice: Annotated[
        bool, typer.Option("--voice", help="Measure the voice legs (the default).")
    ] = True,
    runs: Annotated[int, typer.Option("--runs", "-n")] = 5,
    stt: Annotated[
        bool, typer.Option("--stt", help="Also measure STT (spends audio quota).")
    ] = False,
    playback: Annotated[
        bool, typer.Option("--playback", help="Play audio while timing.")
    ] = False,
) -> None:
    """Measure latency on this machine. Numbers come from real runs."""
    if not voice:
        console.print("[dim]only the voice legs are measurable so far; --voice is implied.[/dim]")
    from .voice.bench import bench_pipeline, summarise
    from .voice.tts import PiperSynthesizer

    settings = _settings()
    synth = PiperSynthesizer(settings.paths.ensure().models_dir)
    with console.status(f"benchmarking ({runs} runs)..."):
        results = bench_pipeline(settings, runs=runs, synthesizer=synth, stt=stt)
    console.print(summarise(results))
    if not playback:
        console.print(
            "\n[dim]TTS measured without playback; add --playback to include "
            "device write time.[/dim]"
        )


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
