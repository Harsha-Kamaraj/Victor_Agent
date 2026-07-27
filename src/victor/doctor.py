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
    if system == "Windows":
        yield Check("platform", Status.OK, f"{system} {platform.release()}")
    else:
        yield Check(
            "platform",
            Status.WARN,
            f"{system} - desktop control unavailable",
            hint="UI Automation is a Windows API. Voice, shell and memory work "
            "here; screen perception and actuation do not.",
        )


def _check_config(settings: Settings) -> Iterator[Check]:
    required = {"groq_api_key": "GROQ_API_KEY"}
    recommended = {"gemini_api_key": "GEMINI_API_KEY"}
    optional = {"github_token": "GITHUB_TOKEN"}

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

    for field, env in recommended.items():
        if settings.has(field):
            yield Check(f"key {env}", Status.OK, "set")
        else:
            yield Check(
                f"key {env}",
                Status.WARN,
                "missing - vision falls back to Groq Llama-4-Scout",
                hint="Free key: https://aistudio.google.com/apikey",
            )

    for field, env in optional.items():
        status = Status.OK if settings.has(field) else Status.SKIP
        detail = "set" if settings.has(field) else "not set - P7 Scout will be rate limited"
        yield Check(f"key {env}", status, detail)


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
        "voice (P1)": ("sounddevice", "numpy", "webrtcvad"),
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

    if platform.system() == "Windows":
        if find_spec("uiautomation") is None:
            yield Check(
                "deps uiautomation",
                Status.SKIP,
                "missing",
                hint="pip install -e '.[desktop]'",
            )
        else:
            yield Check("deps uiautomation", Status.OK, "installed")


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


def _check_pending() -> Iterator[Check]:
    """Capabilities the README promises that later phases will deliver."""
    on_windows = platform.system() == "Windows"
    yield Check("microphone", Status.PENDING, "P1 not implemented")
    yield Check("speakers / TTS", Status.PENDING, "P1 not implemented")
    yield Check("agent loop", Status.PENDING, "P2 not implemented")
    yield Check("safety interceptor", Status.PENDING, "P3 not implemented")
    if on_windows:
        yield Check("UIA tree access", Status.PENDING, "P4 not implemented")
    else:
        yield Check("UIA tree access", Status.SKIP, "Windows only")
    yield Check("memory index", Status.PENDING, "P6 not implemented")


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
    if network:
        checks += list(_check_network(settings))
    else:
        checks.append(Check("network", Status.SKIP, "--no-network"))
    checks += list(_check_pending())
    return checks
