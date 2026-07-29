"""Pre-flight checks.

``victor doctor`` is the first thing anyone runs and the last thing to lie.
Checks that cover unbuilt phases report PENDING, not OK - a green tick for a
microphone pipeline that does not exist yet would make the whole command
worthless.
"""

from __future__ import annotations

import platform
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec

import httpx

from .config import Settings
from .errors import VictorError
from .providers.registry import all_specs
from .quota import QuotaLedger


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str
    hint: str = ""

    @property
    def blocking(self) -> bool:
        return self.status is Status.FAIL


def _check_runtime() -> Iterator[Check]:
    # No <3.13 branch here: the package metadata requires 3.13, so an older
    # interpreter fails at install time and never reaches this code.
    yield Check("python", Status.OK, ".".join(str(v) for v in sys.version_info[:3]))

    system = platform.system()
    if system in ("Windows", "Darwin"):
        yield Check("platform", Status.OK, f"{system} {platform.release()}")
    else:
        yield Check(
            "platform",
            Status.WARN,
            f"{system} - screen perception unavailable",
            hint="Perception needs an accessibility backend: UI Automation on "
            "Windows, the Accessibility API on macOS. Voice, shell, git, "
            "safety and memory all work here.",
        )


def _check_config(settings: Settings) -> Iterator[Check]:
    required = {"groq_api_key": "GROQ_API_KEY"}
    # Keys whose consumer is not built yet. Reporting a missing one as WARN
    # would imply the capability exists and is merely degraded, which is the
    # kind of overstatement the PENDING convention exists to prevent.
    unused_yet = {"gemini_api_key": ("GEMINI_API_KEY", "vision", "P4")}
    optional = {"github_token": ("GITHUB_TOKEN", "Scout", "P7")}

    for field, env in required.items():
        if settings.has(field):
            yield Check(f"key {env}", Status.OK, "set")
        else:
            yield Check(
                f"key {env}",
                Status.FAIL,
                "missing",
                hint="Text reasoning and speech-to-text both need it. "
                "Free key: https://console.groq.com/keys",
            )

    for field, (env, feature, phase) in {**unused_yet, **optional}.items():
        detail = f"set - unused until {feature} lands in {phase}"
        if not settings.has(field):
            detail = f"not set - only needed once {feature} lands in {phase}"
        yield Check(f"key {env}", Status.SKIP, detail)


def _check_storage(settings: Settings) -> Iterator[Check]:
    paths = settings.paths
    try:
        paths.ensure()
        probe = paths.root / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        yield Check("data dir", Status.FAIL, f"{paths.root}: {exc}")
        return
    yield Check("data dir", Status.OK, str(paths.root))

    try:
        ledger = QuotaLedger(paths.quota_file, autoflush=False)
    except Exception as exc:  # pragma: no cover - defensive
        yield Check("quota ledger", Status.FAIL, str(exc))
        return

    spent = [
        f"{spec.model.split('/')[-1]} {ledger.usage(spec.key)[0]}"
        for spec in all_specs()
        if not spec.local and ledger.usage(spec.key)[0] > 0
    ]
    detail = ", ".join(spent) if spent else "no usage recorded today"
    yield Check("quota ledger", Status.OK, detail)


def _check_dependencies() -> Iterator[Check]:
    """Optional extras, mapped to the phase that needs them."""
    groups = {
        "voice (P1)": ("sounddevice", "numpy", "piper"),
        "desktop (P4/P5)": ("mss", "PIL"),
        "memory (P6)": ("faiss", "fastembed"),
    }
    for label, modules in groups.items():
        missing = [m for m in modules if find_spec(m) is None]
        if not missing:
            yield Check(f"deps {label}", Status.OK, "installed")
        else:
            extra = label.split()[0]
            yield Check(
                f"deps {label}",
                Status.SKIP,
                f"missing {', '.join(missing)}",
                hint=f"pip install -e '.[{extra}]' when you reach that phase",
            )

    backend_module, backend_label = {
        "Windows": ("uiautomation", "uiautomation"),
        "Darwin": ("ApplicationServices", "pyobjc"),
    }.get(platform.system(), ("", ""))
    if backend_module:
        installed = find_spec(backend_module) is not None
        yield Check(
            f"deps {backend_label}",
            Status.OK if installed else Status.SKIP,
            "installed" if installed else "missing",
            hint="" if installed else "pip install -e '.[desktop]'",
        )


def _check_network(settings: Settings, timeout: float = 6.0) -> Iterator[Check]:
    """Verify the keys actually authenticate, not merely that they are set."""
    if settings.has("groq_api_key"):
        yield _probe(
            "groq reachable",
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {settings.secret('groq_api_key')}"},
            timeout=timeout,
        )
    else:
        yield Check("groq reachable", Status.SKIP, "no key to test")

    if settings.has("gemini_api_key"):
        yield _probe(
            "gemini reachable",
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": settings.secret("gemini_api_key") or ""},
            timeout=timeout,
        )
    else:
        yield Check("gemini reachable", Status.SKIP, "no key to test")


def _probe(name: str, url: str, *, headers: dict[str, str], timeout: float) -> Check:
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return Check(name, Status.FAIL, f"unreachable: {type(exc).__name__}")
    if response.status_code == 200:
        return Check(name, Status.OK, "authenticated")
    if response.status_code in (401, 403):
        return Check(name, Status.FAIL, f"key rejected (HTTP {response.status_code})")
    if response.status_code == 429:
        return Check(name, Status.WARN, "rate limited - key is valid but throttled")
    return Check(name, Status.WARN, f"HTTP {response.status_code}")


def _check_voice(settings: Settings) -> Iterator[Check]:
    """P1: real capture, playback and synthesis availability."""
    if find_spec("numpy") is None or find_spec("sounddevice") is None:
        yield Check(
            "microphone",
            Status.SKIP,
            "voice extra not installed",
            hint="pip install -e '.[voice]'",
        )
        yield Check("speakers", Status.SKIP, "voice extra not installed")
    else:
        from .voice.sources import list_devices

        devices = list_devices()
        if not devices:
            yield Check(
                "microphone",
                Status.FAIL,
                "PortAudio found no devices",
                hint="On macOS, grant the terminal microphone access in "
                "System Settings > Privacy & Security.",
            )
            yield Check("speakers", Status.FAIL, "PortAudio found no devices")
        else:
            inputs = [d for d in devices if d["inputs"]]
            outputs = [d for d in devices if d["outputs"]]
            default_in = next((d for d in inputs if d["default_input"]), None)
            default_out = next((d for d in outputs if d["default_output"]), None)

            yield (
                Check("microphone", Status.OK, f"{default_in['name']} (+{len(inputs) - 1} more)")
                if default_in
                else Check("microphone", Status.WARN, f"{len(inputs)} inputs, none default")
            )
            yield (
                Check("speakers", Status.OK, str(default_out["name"]))
                if default_out
                else Check("speakers", Status.WARN, f"{len(outputs)} outputs, none default")
            )

    if find_spec("piper") is None:
        yield Check(
            "tts (piper)",
            Status.SKIP,
            "piper-tts not installed",
            hint="pip install -e '.[voice]'",
        )
    else:
        from .voice.tts import DEFAULT_VOICE, PiperSynthesizer

        synth = PiperSynthesizer(settings.paths.models_dir, DEFAULT_VOICE)
        if synth.installed:
            size_mb = synth.model_path.stat().st_size / 1e6
            yield Check("tts (piper)", Status.OK, f"{DEFAULT_VOICE} ({size_mb:.0f} MB)")
        else:
            yield Check(
                "tts (piper)",
                Status.WARN,
                f"voice {DEFAULT_VOICE} not downloaded",
                hint="victor voice install",
            )


def _check_agent(settings: Settings) -> Iterator[Check]:
    """P2: the loop is wired and has tools, even if it cannot call out yet."""
    from .tools import build_registry, is_repository

    registry = build_registry(settings)
    yield Check("agent tools", Status.OK, ", ".join(registry.names))

    if not settings.has("groq_api_key"):
        yield Check(
            "agent loop",
            Status.WARN,
            "wired, but no text model is reachable without GROQ_API_KEY",
        )
    else:
        yield Check("agent loop", Status.OK, "ReAct loop ready")

    if is_repository():
        yield Check("git repository", Status.OK, "cwd is inside a work tree")
    else:
        yield Check("git repository", Status.SKIP, "cwd is not a git work tree")


def _check_safety(settings: Settings) -> Iterator[Check]:
    """P3: is anything actually standing between the model and the machine?"""
    from .safety import ActionJournal, DenyingConfirmer, build_confirmer

    if settings.dry_run:
        yield Check("safety mode", Status.OK, "dry run - nothing will be executed")
    elif not settings.confirm_destructive:
        yield Check(
            "safety mode",
            Status.WARN,
            "VICTOR_CONFIRM_DESTRUCTIVE=false - writes run without asking",
            hint="Irreversible commands are still refused, but nothing else will "
            "stop to check with you.",
        )
    else:
        yield Check("safety mode", Status.OK, "destructive actions require confirmation")

    confirmer = build_confirmer()
    if isinstance(confirmer, DenyingConfirmer):
        yield Check(
            "confirmation",
            Status.WARN,
            "no terminal to ask on - destructive actions will be refused",
            hint="Run from an interactive terminal, or use `victor converse` "
            "to confirm out loud.",
        )
    else:
        yield Check("confirmation", Status.OK, type(confirmer).__name__)

    journal = ActionJournal(settings.paths.journal_file)
    entries = list(journal)
    reversible = sum(1 for e in entries if e.reversible)
    detail = (
        f"{len(entries)} actions recorded, {reversible} reversible"
        if entries
        else "no actions recorded yet"
    )
    yield Check("action journal", Status.OK, detail)


def _check_perception(settings: Settings) -> Iterator[Check]:
    """P4: can Victor read the screen, and capture it if the tree is not enough."""
    from .desktop import ScreenCapture, TreeReader

    reader = TreeReader()
    ok, detail = reader.available()
    label = f"accessibility tree ({reader.backend.name})"
    if ok:
        yield Check(label, Status.OK, detail)
    elif platform.system() in ("Windows", "Darwin"):
        yield Check(
            label,
            Status.FAIL,
            detail,
            hint="pip install -e '.[desktop]'"
            if "not installed" in detail
            else "try `victor uia --demo` to see the output shape",
        )
    else:
        yield Check(label, Status.SKIP, detail)

    # Actuation reuses the same permission and the same session, so there is
    # nothing new to probe - only whether it is switched on. Reported separately
    # because "Victor can see your screen" and "Victor can click on it" are
    # different things to be told.
    from .desktop import select_actuator

    actuator = select_actuator()
    act_ok, act_detail = actuator.available()
    yield Check(
        f"desktop actuation ({actuator.name})",
        Status.OK if act_ok else Status.WARN,
        f"{act_detail}"
        + (
            ""
            if settings.desktop_control
            else " - off by default; pass --desktop or set VICTOR_DESKTOP_CONTROL=1"
        ),
    )

    capture_ok, capture_detail = ScreenCapture.available()
    yield (
        Check("screen capture", Status.OK, capture_detail)
        if capture_ok
        else Check(
            "screen capture",
            Status.SKIP,
            capture_detail,
            hint="pip install -e '.[desktop]'",
        )
    )

    if settings.has("gemini_api_key"):
        yield Check("vision fallback", Status.OK, "gemini, falling back to groq")
    elif settings.has("groq_api_key"):
        yield Check(
            "vision fallback",
            Status.WARN,
            "groq only - no GEMINI_API_KEY, so the scarcer half of the budget is missing",
            hint="Free key: https://aistudio.google.com/apikey",
        )
    else:
        yield Check("vision fallback", Status.FAIL, "no key for any vision model")


def _check_memory(settings: Settings) -> Iterator[Check]:
    """Whether recall will work, and how well.

    The distinction that matters is semantic versus lexical: without fastembed
    Victor still remembers, but only recognises a traceback it has seen almost
    verbatim. That is a warning rather than a failure - a degraded memory is
    still a memory - and saying which one is live keeps the claim honest.
    """
    if not settings.memory_enabled:
        yield Check("memory", Status.SKIP, "disabled (VICTOR_MEMORY=0)")
        return

    from .rag import build_memory, describe_embedder
    from .rag.store import EmbedderChanged

    try:
        memory = build_memory(settings)
    except EmbedderChanged as exc:
        yield Check("memory", Status.FAIL, str(exc), hint="victor index --rebuild")
        return
    except VictorError as exc:
        yield Check("memory", Status.FAIL, str(exc))
        return

    try:
        semantic = memory.embedder.name == "fastembed"
        yield Check(
            "memory",
            Status.OK if semantic else Status.WARN,
            f"{len(memory.store)} records, {describe_embedder(memory.embedder)}",
            hint="" if semantic else "pip install -e '.[memory]'",
        )
        yield Check("memory index", Status.OK, f"{memory.store.backend}, local, no quota")
    finally:
        memory.close()


def _check_pending() -> Iterator[Check]:
    """Capabilities the README promises that later phases will deliver."""
    yield Check("scout", Status.PENDING, "P7 not implemented")
    yield Check("status HUD", Status.PENDING, "P8 not implemented")


def _check_tooling() -> Iterator[Check]:
    git = shutil.which("git")
    if git:
        yield Check("git", Status.OK, git)
    else:
        yield Check("git", Status.WARN, "not on PATH - git tools will be unavailable")


def run_checks(settings: Settings, *, network: bool = True) -> list[Check]:
    """Run everything. Order is roughly cheapest and most fundamental first."""
    checks: list[Check] = []
    checks += list(_check_runtime())
    checks += list(_check_config(settings))
    checks += list(_check_storage(settings))
    checks += list(_check_tooling())
    checks += list(_check_dependencies())
    checks += list(_check_voice(settings))
    checks += list(_check_agent(settings))
    checks += list(_check_safety(settings))
    checks += list(_check_perception(settings))
    checks += list(_check_memory(settings))
    if network:
        checks += list(_check_network(settings))
    else:
        checks.append(Check("network", Status.SKIP, "--no-network"))
    checks += list(_check_pending())
    return checks
