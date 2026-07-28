"""Safety and reversibility: classify, gate, record, reverse, abort."""

from .classify import (
    Classification,
    Risk,
    classify,
    classify_git,
    classify_shell,
    describe,
)
from .confirm import (
    AutoConfirmer,
    Confirmer,
    ConfirmRequest,
    DenyingConfirmer,
    SpokenConfirmer,
    TypedConfirmer,
    build_confirmer,
    interpret,
    summarise_call,
)
from .interceptor import SafetyInterceptor, SafetyStats
from .journal import ActionJournal, Entry, Undo, UndoResult, plan_undo
from .killswitch import (
    Aborted,
    HotkeyListener,
    KillSwitch,
    SignalKillSwitch,
    Trip,
    is_stop_phrase,
)
from .trash import DeletePlan, Trash, TrashedItem, parse_delete, trash_for
from .undo import undo_entry, undo_last

__all__ = [
    "Aborted",
    "ActionJournal",
    "AutoConfirmer",
    "Classification",
    "ConfirmRequest",
    "Confirmer",
    "DeletePlan",
    "DenyingConfirmer",
    "Entry",
    "HotkeyListener",
    "KillSwitch",
    "Risk",
    "SafetyInterceptor",
    "SafetyStats",
    "SignalKillSwitch",
    "SpokenConfirmer",
    "Trash",
    "TrashedItem",
    "Trip",
    "TypedConfirmer",
    "Undo",
    "UndoResult",
    "parse_delete",
    "trash_for",
    "build_confirmer",
    "classify",
    "classify_git",
    "classify_shell",
    "describe",
    "interpret",
    "is_stop_phrase",
    "plan_undo",
    "summarise_call",
    "undo_entry",
    "undo_last",
]
