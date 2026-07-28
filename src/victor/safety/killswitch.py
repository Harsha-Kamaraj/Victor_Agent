"""The kill switch.

Stopping an agent is not the same as killing the process. A ``SIGKILL`` mid-run
leaves the journal missing the action that was in flight, which is precisely the
action you would want recorded. So the switch is cooperative: tripping it sets
a flag, and every place that could start new work checks it first.

Three checkpoints cover everything that takes time:

* between steps of the ReAct loop, before spending another request;
* before a tool executes, after the model has chosen it;
* inside the shell tool's wait loop, so a running subprocess is terminated
  rather than waited out.

That last one is what makes the abort latency bounded by the poll interval
rather than by whatever the command decided to do.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

from ..errors import VictorError

POLL_INTERVAL = 0.05
"""Seconds between checks while waiting on a subprocess."""


class Aborted(VictorError):
    """The kill switch was tripped."""

    exit_code = 130


@dataclass(frozen=True, slots=True)
class Trip:
    """A record of who stopped the run and when."""

    source: str
    at: float

    def age_ms(self) -> float:
        return (time.monotonic() - self.at) * 1000


class KillSwitch:
    """A thread-safe abort flag with observers.

    Not itself tied to any input mechanism - :meth:`trip` is called by a signal
    handler, a hotkey listener, a spoken "stop", or a test.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._trip: Trip | None = None
        self._observers: list[Callable[[Trip], None]] = []

    @property
    def tripped(self) -> bool:
        return self._event.is_set()

    @property
    def trip_record(self) -> Trip | None:
        return self._trip

    def observe(self, callback: Callable[[Trip], None]) -> None:
        """Register something to run the moment the switch trips."""
        self._observers.append(callback)

    def trip(self, source: str = "unknown") -> Trip:
        """Stop the run. Idempotent - the first trip wins."""
        with self._lock:
            if self._trip is not None:
                return self._trip
            record = Trip(source=source, at=time.monotonic())
            self._trip = record
            self._event.set()
        for observer in self._observers:
            # An observer that raises must not stop the abort itself.
            with contextlib.suppress(Exception):
                observer(record)
        return record

    def reset(self) -> None:
        """Re-arm for another run."""
        with self._lock:
            self._event.clear()
            self._trip = None

    def check(self) -> None:
        """Raise :class:`Aborted` if the switch has been tripped."""
        if self._event.is_set():
            source = self._trip.source if self._trip else "unknown"
            raise Aborted(f"stopped by {source}")

    def wait(self, timeout: float) -> bool:
        """Block until tripped or ``timeout`` elapses. True if tripped."""
        return self._event.wait(timeout)


class SignalKillSwitch(KillSwitch):
    """A kill switch wired to Ctrl-C.

    The first Ctrl-C trips the switch and lets the agent unwind cleanly,
    finishing its journal entry and closing the trace. A second Ctrl-C restores
    the default handler and interrupts for real, so a wedged run is still
    escapable.
    """

    def __init__(self, *, install: bool = True) -> None:
        super().__init__()
        self._previous: object = None
        self._installed = False
        if install:
            self.install()

    def install(self) -> None:
        if self._installed:
            return
        try:
            self._previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        except ValueError:
            # Not on the main thread - the caller gets a plain KillSwitch.
            self._installed = False

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.tripped:
            self.restore()
            raise KeyboardInterrupt
        self.trip("Ctrl-C")

    def restore(self) -> None:
        if self._installed and self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)  # type: ignore[arg-type]
            self._installed = False

    def __enter__(self) -> SignalKillSwitch:
        self.install()
        return self

    def __exit__(self, *exc: object) -> None:
        self.restore()


class HotkeyListener:
    """Optional global hotkey, if ``pynput`` is installed.

    A truly system-wide binding needs an OS-level hook, which is a dependency
    and a permissions prompt on macOS. Victor does not require it: Ctrl-C works
    everywhere, and a spoken "stop" works hands-free. This is here so a user who
    wants the global key can have it, and :attr:`available` says plainly whether
    it is active rather than pretending.
    """

    def __init__(self, switch: KillSwitch, combination: str = "<ctrl>+<alt>+x") -> None:
        self.switch = switch
        self.combination = combination
        self._listener = None

    @staticmethod
    def available() -> bool:
        try:
            import pynput  # noqa: F401
        except Exception:
            return False
        return True

    def start(self) -> bool:
        """Begin listening. False if the dependency is missing."""
        if not self.available():
            return False
        from pynput import keyboard

        self._listener = keyboard.GlobalHotKeys(
            {self.combination: lambda: self.switch.trip(f"hotkey {self.combination}")}
        )
        self._listener.start()
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


#: Words that trip the switch when heard, so a stop does not need hands.
STOP_WORDS = frozenset({"stop", "stop stop", "cancel", "abort", "halt", "never mind", "nevermind"})


def is_stop_phrase(text: str) -> bool:
    """True if a transcript is the user asking to stop."""
    cleaned = text.strip().lower().rstrip(".!?")
    return cleaned in STOP_WORDS or cleaned.startswith(("stop ", "cancel ", "abort "))
