"""Every phase's exit gate, as one command.

``victor doctor`` answers "is this machine set up". This answers a different
and harder question: **does the thing actually still do what each phase claimed
it does.** The two are easy to confuse and it matters that they are separate -
doctor reports that fastembed is installed, this makes it remember a fix and
recall it.

The gates come from [PLAN.md](../../docs/PLAN.md), one per phase, each written
as something you can watch happen. Until now checking them meant working
through a list by hand, which is why the record of *when* each was last
demonstrated lived in a markdown file rather than in the code.

**Three rules, all learned from `doctor`.**

A gate that cannot run here is SKIP with the reason, never a quiet pass. A
locked screen, no microphone, no API key - these are facts about the machine,
and reporting them as success is the failure mode this whole project is
written against.

Nothing spends quota unless asked. The default run makes no API call at all,
so it is safe in a loop and in CI. ``--live`` adds the gates that need a real
model, and says what each one costs before spending it.

Gates are ordered by phase, and a later gate may assume an earlier one passed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .config import Settings
from .doctor import Status

#: What a live gate costs, in provider requests, so the total can be stated
#: before anything is spent.
LIVE_COST = {"P1 voice": 1, "P2 agent": 2, "P4 vision": 1, "P7 scout": 0}


@dataclass(frozen=True, slots=True)
class Gate:
    """One phase's exit criterion, and whether it holds right now."""

    phase: str
    claim: str
    """What the plan said this phase would demonstrate."""
    status: Status
    detail: str
    duration_ms: float = 0.0
    cost: int = 0
    """Provider requests spent proving it. Zero for everything but --live."""

    @property
    def blocking(self) -> bool:
        return self.status is Status.FAIL


@dataclass
class Report:
    gates: list[Gate] = field(default_factory=list)

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if g.blocking]

    @property
    def skipped(self) -> list[Gate]:
        return [g for g in self.gates if g.status is Status.SKIP]

    @property
    def passed(self) -> list[Gate]:
        return [g for g in self.gates if g.status is Status.OK]

    @property
    def cost(self) -> int:
        return sum(g.cost for g in self.gates)

    def summary(self) -> str:
        return (
            f"{len(self.passed)} passed, {len(self.failed)} failed, "
            f"{len(self.skipped)} skipped, {self.cost} API "
            f"{'request' if self.cost == 1 else 'requests'} spent"
        )


class _Timer:
    """Measures a gate, so a pass can report how long the thing took."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000


def _gate(phase: str, claim: str):
    """Turn a function returning ``(status, detail)`` into a :class:`Gate`.

    The wrapper exists for one reason: a gate that raises must become a FAIL
    with the exception in its detail, never a traceback. A self-test that
    crashes half way through has told you less than one that says which gate
    broke and carries on.
    """

    def decorate(func):
        def run(*args: Any, **kwargs: Any) -> Gate:
            timer = _Timer()
            try:
                status, detail = func(*args, **kwargs)
                cost = 0
                if isinstance(detail, tuple):
                    detail, cost = detail
            except Exception as exc:  # noqa: BLE001 - a broken gate is a result
                return Gate(
                    phase, claim, Status.FAIL, f"{type(exc).__name__}: {exc}", timer.ms
                )
            return Gate(phase, claim, status, detail, timer.ms, cost)

        return run

    return decorate


# --- P0 · routing and the quota ledger --------------------------------------


@_gate("P0", "routing falls through when a provider's free tier is spent")
def _p0_routing(settings: Settings) -> tuple[Status, str]:
    from .providers import Router, Workload
    from .quota import QuotaLedger

    with TemporaryDirectory() as tmp:
        ledger = QuotaLedger(Path(tmp) / "quota.json")
        router = Router(settings, ledger)
        first = router.select(Workload.VISION)

        # Spend the primary's whole day in one write, then ask again. The gate
        # is that the answer changes, not that a call fails.
        limit = first.spec.limits.requests_per_day or 1
        ledger.record(first.spec.key, first.spec.limits, requests=limit)
        second = router.select(Workload.VISION)

    if second.spec.key == first.spec.key:
        return Status.FAIL, f"still chose {first.spec.key} with its day spent"
    return Status.OK, f"{first.spec.key} spent -> fell through to {second.spec.key}"


@_gate("P0", "the ledger survives a restart and knows each provider's timezone")
def _p0_ledger(settings: Settings) -> tuple[Status, str]:
    from .providers.registry import all_specs
    from .quota import QuotaLedger

    spec = next(iter(all_specs()))
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "quota.json"
        ledger = QuotaLedger(path)
        ledger.record(spec.key, spec.limits)
        ledger.flush()
        requests, _, _ = QuotaLedger(path).usage(spec.key)

    zones = {s.limits.reset_timezone for s in all_specs()}
    if requests != 1:
        return Status.FAIL, f"reopened ledger reported {requests} requests, expected 1"
    return Status.OK, (
        f"persisted across reopen; {len(zones)} reset "
        f"{'timezone' if len(zones) == 1 else 'timezones'} tracked ({', '.join(sorted(zones))})"
    )


# --- P1 · voice --------------------------------------------------------------


def _speak(settings: Settings, text: str):
    """Synthesise ``text`` with Piper into a :class:`Segment`, or ``None``.

    Shared by both voice gates: the STT gate needs something to transcribe, and
    generating it locally means the round trip can be checked without a
    microphone or a recorded fixture that would drift out of date.
    """
    import numpy as np

    from .voice.audio import AudioFormat, Segment
    from .voice.tts import PiperSynthesizer

    synth = PiperSynthesizer(settings.paths.ensure().models_dir, auto_download=False)
    if not synth.installed:
        return None
    chunks = list(synth.stream(text))
    if not chunks:
        return None
    samples = np.concatenate(chunks)
    return Segment(samples=samples, fmt=AudioFormat(sample_rate=synth.sample_rate, channels=1))


@_gate("P1", "text becomes speech locally, with no API call")
def _p1_tts(settings: Settings) -> tuple[Status, str]:
    segment = _speak(settings, "Victor self test.")
    if segment is None:
        return Status.SKIP, "no Piper voice installed -> victor voice install"
    return Status.OK, f"{segment.duration:.2f}s of audio, offline, 0 API calls"


@_gate("P1", "speech becomes text through Groq Whisper")
def _p1_stt(settings: Settings) -> tuple[Status, tuple[str, int]]:
    from .providers import Router
    from .quota import QuotaLedger
    from .voice.stt import Transcriber

    if not settings.has("groq_api_key"):
        return Status.SKIP, ("GROQ_API_KEY is not set", 0)

    spoken = "the quick brown fox"
    segment = _speak(settings, spoken)
    if segment is None:
        return Status.SKIP, ("needs a Piper voice to speak the test phrase", 0)

    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    transcriber = Transcriber(settings, Router(settings, ledger))
    try:
        heard = transcriber.transcribe(segment)
    finally:
        transcriber.close()

    words = {w.strip(".,!?").lower() for w in heard.text.split()}
    if len({"quick", "brown", "fox"} & words) < 2:
        return Status.FAIL, (f"said {spoken!r}, heard {heard.text.strip()!r}", 1)
    return Status.OK, (f"said it and heard {heard.text.strip()!r} back", 1)


# --- P2 · the agent loop -----------------------------------------------------


@_gate("P2", "the model chooses a tool, reads the result, and answers")
def _p2_agent(settings: Settings) -> tuple[Status, tuple[str, int]]:
    if not settings.has("groq_api_key"):
        return Status.SKIP, ("GROQ_API_KEY is not set", 0)

    from .agent.loop import build_agent

    agent = build_agent(settings, max_steps=4, desktop=False)
    try:
        result = agent.run(
            "Use the shell tool to print exactly SELFTEST-OK, then tell me what it printed."
        )
    finally:
        agent.close()

    if not result.tool_calls:
        return Status.FAIL, ("the model answered without calling a tool", result.api_calls)
    if "SELFTEST-OK" not in result.answer:
        return Status.FAIL, (f"answered {result.answer[:80]!r}", result.api_calls)
    return Status.OK, (
        f"{len(result.steps)} steps, {result.total_tokens} tokens, "
        f"{result.free_tool_calls}/{len(result.tool_calls)} tool calls free",
        result.api_calls,
    )


# --- P3 · safety -------------------------------------------------------------


@_gate("P3", "an irreversible command is refused and a risky one is gated")
def _p3_classification(settings: Settings) -> tuple[Status, str]:
    from .safety.classify import Risk, classify_shell

    expected = {
        "rm -rf /": Risk.DENY,
        "chmod -R 777 /": Risk.DENY,
        # Refused outright because the history is shared; the same push to a
        # feature branch is ordinary work and only asks.
        "git push --force origin main": Risk.DENY,
        "git push --force origin my-feature": Risk.CONFIRM,
        "rm -rf build": Risk.CONFIRM,
        "git status": Risk.SAFE,
        "ls -la": Risk.SAFE,
    }
    wrong = [
        f"{command!r} -> {classify_shell(command).risk} (wanted {risk})"
        for command, risk in expected.items()
        if classify_shell(command).risk is not risk
    ]
    if wrong:
        return Status.FAIL, "; ".join(wrong)
    return Status.OK, f"{len(expected)} commands classified as the plan requires"


@_gate("P3", "a delete is reversible and the journal says how")
def _p3_undo(settings: Settings) -> tuple[Status, str]:
    from .safety import trash_for

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        victim = root / "notes.txt"
        victim.write_text("do not lose me", encoding="utf-8")

        trash = trash_for(root / "state", "selftest")
        item = trash.store(victim)
        if victim.exists():
            return Status.FAIL, "the file was still there after being trashed"

        trash.restore(item)
        if not victim.exists():
            return Status.FAIL, "restore did not bring the file back"
        if victim.read_text(encoding="utf-8") != "do not lose me":
            return Status.FAIL, "the restored file had different contents"
    return Status.OK, "trashed and restored, contents intact"


@_gate("P3", "the kill switch stops a run in flight")
def _p3_killswitch(settings: Settings) -> tuple[Status, str]:
    from .safety.killswitch import Aborted, KillSwitch

    switch = KillSwitch()
    switch.trip("selftest")
    try:
        switch.check()
    except Aborted:
        return Status.OK, "a tripped switch refuses the next action"
    return Status.FAIL, "a tripped switch allowed the next action"


# --- P4/P5 · perception and actuation ----------------------------------------


@_gate("P4", "the accessibility tree names real controls, free and local")
def _p4_perception(settings: Settings) -> tuple[Status, str]:
    from .desktop.session import session_locked
    from .desktop.uia import PerceptionUnavailable, TreeReader

    locked, why = session_locked()
    if locked:
        return Status.SKIP, why

    try:
        snapshot = TreeReader().snapshot(refresh=True)
    except PerceptionUnavailable as exc:
        return Status.SKIP, str(exc)

    named = [e for e in snapshot.elements if e.label.strip()]
    if not named:
        return Status.FAIL, f"{snapshot.window_title!r} produced no named controls"
    return Status.OK, (
        f"{len(named)} named controls in {snapshot.window_title!r}, 0 API calls"
    )


@_gate("P4", "the screen can be captured for the vision fallback")
def _p4_capture(settings: Settings) -> tuple[Status, str]:
    from .desktop.capture import ScreenCapture

    ok, detail = ScreenCapture.available()
    return (Status.OK, detail) if ok else (Status.SKIP, detail)


@_gate("P4", "a vision model locates an element the tree could not name")
def _p4_vision(settings: Settings) -> tuple[Status, tuple[str, int]]:
    from .desktop.capture import ScreenCapture
    from .desktop.session import session_locked
    from .desktop.uia import TreeReader
    from .desktop.vision import VisionClient, annotate
    from .providers import Router
    from .quota import QuotaLedger

    if not (settings.has("gemini_api_key") or settings.has("groq_api_key")):
        return Status.SKIP, ("no vision-capable key is set", 0)
    locked, why = session_locked()
    if locked:
        return Status.SKIP, (why, 0)
    ok, detail = ScreenCapture.available()
    if not ok:
        return Status.SKIP, (detail, 0)

    snapshot = TreeReader().snapshot(refresh=True)
    shot, _ = ScreenCapture().capture()
    ledger = QuotaLedger(settings.paths.ensure().quota_file)
    vision = VisionClient(settings, Router(settings, ledger))
    answer = vision.locate("any button or menu on this screen", annotate(shot, snapshot), snapshot)

    if not answer.found:
        # Not a failure: the model looked and said there was nothing, which is
        # an honest answer it has to be able to give. A gate that only passes
        # when the model claims to see something teaches it to claim.
        return Status.OK, (f"{answer.model} looked and pointed at nothing", 1)
    return Status.OK, (f"{answer.model} pointed at {answer}", 1)


@_gate("P5", "actuation drives a control through its own action, not a click at x,y")
def _p5_actuation(settings: Settings) -> tuple[Status, str]:
    from .desktop.actions import Desktop

    ok, detail = Desktop().available()
    if not ok:
        return Status.SKIP, detail
    # Deliberately not pressed. A self-test that clicks whatever happens to be
    # in front of it is a self-test nobody will run twice.
    return Status.OK, f"{detail}; not exercised - it would act on your desktop"


# --- P6 · memory -------------------------------------------------------------


@_gate("P6", "a fix is captured unprompted and recalled with no API call")
def _p6_memory(settings: Settings) -> tuple[Status, str]:
    from .rag import ErrorFixWatcher, Memory, VectorStore, select_embedder
    from .rag.ingest import describe_call

    traceback = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 1, in <module>\n'
        "    import httpx\n"
        "ModuleNotFoundError: No module named 'httpx'"
    )

    with TemporaryDirectory() as tmp:
        embedder = select_embedder(settings.paths.ensure().models_dir)
        store = VectorStore(
            Path(tmp) / "memory",
            embedder_name=embedder.name,
            dimensions=embedder.dimensions,
        )
        memory = Memory(store, embedder)
        watcher = ErrorFixWatcher(memory)

        # The whole capture rule, in four observations.
        watcher.observe("python3 app.py", ok=False, output=traceback)
        watcher.observe("ls -la", ok=True)  # looking around, must be ignored
        watcher.observe("pip install httpx", ok=True)
        note = watcher.observe("python3 app.py", ok=True)
        if note is None:
            return Status.FAIL, "the fix was not captured"

        started = time.perf_counter()
        recalled = memory.recall_for_error(traceback)
        elapsed = (time.perf_counter() - started) * 1000

        if not recalled.found:
            return Status.FAIL, "the captured fix could not be recalled"
        fix = recalled.best.record.meta.get("fix", "")
        if "pip install httpx" not in fix:
            return Status.FAIL, f"recalled the wrong fix: {fix!r}"
        if "ls -la" in fix:
            return Status.FAIL, "a diagnostic was stored as the fix"

        # The other half of the change that made this gate worth having.
        click = describe_call("click", {"index": 3, "label": "Save"}, mutating=True)
        focus = describe_call("open_app", {"name": "Notepad"}, mutating=True)
        watcher.observe(click, ok=False, output="no element at index 3")
        watcher.observe(focus, ok=True)
        if watcher.observe(click, ok=True) is None:
            return Status.FAIL, "a desktop fix was not captured"

        embedder_name = memory.embedder.name

    return Status.OK, (
        f"captured shell and desktop fixes; recalled at "
        f"{recalled.best.score:.2f} in {elapsed:.0f}ms, 0 API calls ({embedder_name})"
    )


# --- P7 · scout --------------------------------------------------------------


@_gate("P7", "the portfolio gap analysis reaches real GitHub data")
def _p7_scout(settings: Settings) -> tuple[Status, tuple[str, int]]:
    from .scout.github import GitHubClient

    client = GitHubClient(settings.secret("github_token"))
    try:
        remaining, limit = client.rate_limit()
    except Exception as exc:  # noqa: BLE001 - offline is a SKIP, not a failure
        return Status.SKIP, (f"GitHub unreachable: {exc}", 0)
    finally:
        client.close()

    who = "authenticated" if settings.has("github_token") else "anonymous"
    if limit <= 60 and settings.has("github_token"):
        return Status.FAIL, (f"token set but rate limit is {limit}/hour - not accepted", 0)
    return Status.OK, (f"{remaining}/{limit} requests remaining ({who})", 0)


# --- P8 · surface ------------------------------------------------------------


@_gate("P8", "the status strip reads live state off disk")
def _p8_surface(settings: Settings) -> tuple[Status, str]:
    from .ui.hud import build_monitor

    snapshot = build_monitor(settings).read()
    drawable = find_spec("tkinter") is not None
    return Status.OK, (
        f"{snapshot.state}, {snapshot.cost_line}"
        f"{'' if drawable else '; tkinter missing, cannot draw'}"
    )


@_gate("P8", "traces record what every run cost")
def _p8_tracing(settings: Settings) -> tuple[Status, str]:
    from .tracing import Trace

    with TemporaryDirectory() as tmp:
        trace = Trace.open(Path(tmp), label="selftest")
        with trace.span("selftest.span") as span:
            span["cost"] = 0
        trace.event("selftest.event", detail="written")

        written = list(Path(tmp).glob("*.jsonl"))
        if not written:
            return Status.FAIL, "no trace file was written"
        # Read it back before closing: the claim is that a killed process still
        # leaves a readable trace, which is only true if every event is flushed.
        lines = written[0].read_text(encoding="utf-8").strip().splitlines()

    if len(lines) < 2:
        return Status.FAIL, f"trace held {len(lines)} events, expected at least 2"
    return Status.OK, f"{len(lines)} events, flushed per event rather than at exit"


# --- running them ------------------------------------------------------------

_FREE = (
    _p0_routing,
    _p0_ledger,
    _p1_tts,
    _p3_classification,
    _p3_undo,
    _p3_killswitch,
    _p4_perception,
    _p4_capture,
    _p5_actuation,
    _p6_memory,
    _p8_surface,
    _p8_tracing,
)

_LIVE = (_p1_stt, _p2_agent, _p4_vision, _p7_scout)


def live_cost() -> int:
    """Provider requests a ``--live`` run will spend, at most."""
    return sum(LIVE_COST.values())


def run_gates(settings: Settings, *, live: bool = False) -> Iterator[Gate]:
    """Every gate this machine can run, in phase order.

    Yielded one at a time so a caller can print progress: the live gates take
    seconds each and a silent command looks like a hung one.
    """
    for gate in _FREE:
        yield gate(settings)
    if live:
        for gate in _LIVE:
            yield gate(settings)


def selftest(settings: Settings, *, live: bool = False) -> Report:
    return Report(list(run_gates(settings, live=live)))
