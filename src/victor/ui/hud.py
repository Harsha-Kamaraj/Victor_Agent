"""A status strip that sits on top of everything else.

The plan is specific about scope: *"a status strip, not a UI framework - do not
rabbit-hole into PyQt"*. So this is tkinter, which ships with Python, and it is
one borderless always-on-top window with three labels in it.

**It reads state off disk rather than being handed it.** The obvious design is
for the agent to push updates into the HUD, and it is the wrong one twice over.
Tk insists on owning the main thread on macOS, so an agent that also wants the
main thread ends up with a threading problem that has nothing to do with the
feature. And a HUD wired into the agent can only watch runs it was started
with.

Both problems disappear if the HUD is a *monitor*: the quota ledger and the
session traces are already files that the agent maintains, so the strip polls
them. You can start it before or after a task, in a different terminal, and it
shows the same thing either way. The coupling is a directory.

**The quota counter is the story.** Everything else on the strip is context for
it. A number that stays at zero while an agent clicks through a file manager is
the whole claim of the project, made visible without anybody having to trust a
README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import VictorError

POLL_MS = 400
"""Fast enough to feel live, slow enough to cost nothing. The underlying files
change at human speed, so polling harder would only burn battery."""

WIDTH = 560
HEIGHT = 74


class HudUnavailable(VictorError):
    """No display, or no tkinter to draw on it."""

    exit_code = 8


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What the strip is showing right now."""

    state: str = "idle"
    detail: str = ""
    spent: int = 0
    remaining: str = ""
    session: str = ""

    @property
    def cost_line(self) -> str:
        """The number the whole project is an argument about."""
        if self.spent == 0:
            return "0 API calls today"
        return f"{self.spent} API calls today{f' - {self.remaining}' if self.remaining else ''}"


#: Trace event kinds mapped to a word a person would use.
_STATES = {
    "voice.listen": ("listening", "microphone open"),
    "stt.transcribe": ("hearing", ""),
    "agent.run": ("thinking", ""),
    "llm.complete": ("thinking", ""),
    "tool.run": ("acting", ""),
    "safety.confirm": ("waiting", "needs your confirmation"),
    "memory.recall": ("remembering", "found a past fix"),
    "vision.locate": ("looking", "spending vision quota"),
    "agent.answer": ("idle", ""),
}


@dataclass
class Monitor:
    """Reads Victor's state from the files it already writes."""

    quota_file: Path
    traces_dir: Path
    _last_trace: Path | None = field(default=None, repr=False)

    def read(self) -> Snapshot:
        spent, remaining = self._read_quota()
        state, detail, session = self._read_trace()
        return Snapshot(
            state=state, detail=detail, spent=spent, remaining=remaining, session=session
        )

    def _read_quota(self) -> tuple[int, str]:
        """Requests spent today, and which models spent them.

        Only today's buckets count. The ledger keeps a bucket's ``day`` so a
        rollover does not lose history, which means summing everything would
        show yesterday's spending on the strip and make a fresh day look
        expensive.
        """
        try:
            raw = json.loads(self.quota_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0, ""
        if not isinstance(raw, dict):
            return 0, ""

        today = _today_keys()
        spent = 0
        busiest: list[tuple[int, str]] = []
        for key, bucket in (raw.get("buckets") or {}).items():
            if not isinstance(bucket, dict) or str(bucket.get("day", "")) not in today:
                continue
            requests = int(bucket.get("requests", 0) or 0)
            spent += requests
            if requests:
                busiest.append((requests, str(key).split(":")[-1].split("/")[-1]))

        busiest.sort(reverse=True)
        detail = ", ".join(f"{name} {n}" for n, name in busiest[:2])
        return spent, detail

    def _read_trace(self) -> tuple[str, str, str]:
        newest = self._newest_trace()
        if newest is None:
            return "idle", "no session yet", ""
        try:
            lines = newest.read_text(encoding="utf-8").splitlines()
        except OSError:
            return "idle", "", newest.stem

        # Walk backwards to the most recent event this strip knows a word for.
        for line in reversed(lines[-60:]):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(event.get("kind", ""))
            for prefix, (state, detail) in _STATES.items():
                if kind.startswith(prefix):
                    payload = event.get("payload", {})
                    return state, detail or _describe(kind, payload), newest.stem
        return "idle", "", newest.stem

    def _newest_trace(self) -> Path | None:
        try:
            traces = sorted(
                self.traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        except OSError:
            return None
        return traces[0] if traces else None


def _today_keys() -> set[str]:
    """Today's date string in every timezone the ledger buckets by.

    Not one date. Groq's day rolls at UTC midnight and Google's at midnight
    Pacific, so the ledger keys buckets per provider timezone - and for several
    hours each day those two disagree. Comparing against a single "today" would
    make half the ledger look like yesterday and the strip would read zero
    during a run that was spending.

    Derived from the routing table so a new provider in a new timezone is
    counted without anyone remembering to update this.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from ..providers.registry import all_specs

    zones = {spec.limits.reset_timezone for spec in all_specs() if not spec.local}
    zones.add("UTC")
    keys = set()
    for zone in zones:
        try:
            keys.add(datetime.now(ZoneInfo(zone)).strftime("%Y-%m-%d"))
        except Exception:
            continue
    return keys


def _describe(kind: str, payload: dict[str, Any]) -> str:
    if kind.startswith("tool.run"):
        return str(payload.get("tool", ""))
    if kind.startswith("agent.answer"):
        return str(payload.get("answer", ""))[:60]
    return ""


class Hud:
    """The always-on-top strip itself."""

    def __init__(self, monitor: Monitor, *, poll_ms: int = POLL_MS) -> None:
        self.monitor = monitor
        self.poll_ms = poll_ms
        self._root: Any = None
        self._labels: dict[str, Any] = {}
        self._drag = (0, 0)

    def _build(self) -> Any:
        try:
            import tkinter as tk
        except ImportError as exc:  # pragma: no cover - stdlib on supported platforms
            raise HudUnavailable(
                f"tkinter is not available ({exc}). Use `victor hud --text` instead."
            ) from exc

        try:
            root = tk.Tk()
        except Exception as exc:
            raise HudUnavailable(
                f"no display to draw on ({exc}). Use `victor hud --text` instead."
            ) from exc

        root.title("Victor")
        root.overrideredirect(True)  # no title bar: it is a strip, not a window
        root.attributes("-topmost", True)
        root.geometry(f"{WIDTH}x{HEIGHT}+40+40")
        root.configure(bg="#11141a")

        frame = tk.Frame(root, bg="#11141a", padx=14, pady=10)
        frame.pack(fill="both", expand=True)

        self._labels["state"] = tk.Label(
            frame, text="idle", fg="#7dd3a0", bg="#11141a",
            font=("Helvetica", 15, "bold"), anchor="w",
        )
        self._labels["state"].pack(fill="x")

        self._labels["detail"] = tk.Label(
            frame, text="", fg="#8b94a6", bg="#11141a",
            font=("Helvetica", 11), anchor="w",
        )
        self._labels["detail"].pack(fill="x")

        self._labels["cost"] = tk.Label(
            frame, text="0 API calls today", fg="#e6b800", bg="#11141a",
            font=("Helvetica", 12, "bold"), anchor="w",
        )
        self._labels["cost"].pack(fill="x")

        # Borderless windows cannot be moved by the window manager, so the
        # strip has to carry its own drag - otherwise it lands wherever it
        # opens and covers something.
        for widget in (root, frame, *self._labels.values()):
            widget.bind("<Button-1>", self._press)
            widget.bind("<B1-Motion>", self._move)
        root.bind("<Escape>", lambda _event: root.destroy())

        self._root = root
        return root

    def _press(self, event: Any) -> None:
        self._drag = (event.x_root, event.y_root)

    def _move(self, event: Any) -> None:
        root = self._root
        dx = event.x_root - self._drag[0]
        dy = event.y_root - self._drag[1]
        self._drag = (event.x_root, event.y_root)
        root.geometry(f"+{root.winfo_x() + dx}+{root.winfo_y() + dy}")

    def refresh(self) -> Snapshot:
        """Read the files once and repaint. Returns what it painted."""
        snapshot = self.monitor.read()
        if self._labels:
            self._labels["state"].config(text=snapshot.state)
            self._labels["detail"].config(text=snapshot.detail or snapshot.session)
            self._labels["cost"].config(
                text=snapshot.cost_line,
                fg="#7dd3a0" if snapshot.spent == 0 else "#e6b800",
            )
        return snapshot

    def _tick(self) -> None:
        self.refresh()
        if self._root is not None:
            self._root.after(self.poll_ms, self._tick)

    def run(self) -> None:
        """Show the strip until Escape. Blocks; Tk owns the main thread."""
        root = self._build()
        self._tick()
        root.mainloop()


def build_monitor(settings: Any) -> Monitor:
    paths = settings.paths.ensure()
    return Monitor(quota_file=paths.quota_file, traces_dir=paths.traces_dir)
