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
bench_app = typer.Typer(help="Measured latency benchmarks.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")
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


@bench_app.command("voice")
def bench_voice(
    runs: Annotated[int, typer.Option("--runs", "-n")] = 5,
    stt: Annotated[
        bool, typer.Option("--stt", help="Also measure STT (spends audio quota).")
    ] = False,
    playback: Annotated[
        bool, typer.Option("--playback", help="Play audio while timing.")
    ] = False,
) -> None:
    """Measure VAD, TTS and optionally STT latency on this machine."""
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
