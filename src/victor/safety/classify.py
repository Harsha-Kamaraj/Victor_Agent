"""Risk classification for proposed tool calls.

The governing rule is **fail closed**: anything not recognised as read-only
requires confirmation. A classifier that guesses "probably fine" for commands
it does not understand is worse than no classifier, because it teaches the user
that the prompt means nothing.

The second rule is **avoid alarm fatigue**, which pulls the other way. If every
call needs confirmation, users stop reading and start saying yes, and the
safety layer becomes a latency tax that protects nobody. So the read-only set
is generous and specific: the commands a developer runs dozens of times an hour
to look at things pass silently, and everything that writes, installs, deletes
or reaches the network stops.

Chained commands are classified by their most dangerous segment. ``ls && rm -rf
build`` is a delete, not a listing.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import IntEnum

from ..tools.git import READ_ONLY as GIT_READ_ONLY
from ..tools.shell import screen


class Risk(IntEnum):
    """Ordered so ``max()`` over segments gives the right answer."""

    SAFE = 0
    CONFIRM = 1
    DENY = 2

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Classification:
    """A verdict, and the specific thing that caused it."""

    risk: Risk
    reason: str
    trigger: str = ""
    """The segment or argument responsible, for showing the user."""
    unmatched: bool = False
    """No rule recognised this; the verdict is the fail-closed default.

    Only these are worth spending an LLM call on. A command the rules already
    understand does not get a second opinion - the rules are deterministic,
    auditable and free, and an LLM that could overrule them would make the
    whole classifier only as trustworthy as its weakest layer."""
    source: str = "rule"
    """Which layer produced this: ``rule``, ``llm``, or ``llm-failed``."""

    @property
    def needs_confirmation(self) -> bool:
        return self.risk is Risk.CONFIRM

    def __str__(self) -> str:
        return f"{self.risk}: {self.reason}"


#: Commands that only read. Everything else needs confirmation.
READ_ONLY_COMMANDS = frozenset(
    {
        # inspection
        "ls", "dir", "pwd", "cd", "echo", "printf", "cat", "bat", "head", "tail",
        "less", "more", "nl", "wc", "file", "stat", "realpath", "basename", "dirname",
        "tree", "du", "df", "mount",
        # search
        "grep", "egrep", "fgrep", "rg", "ag", "ack", "locate", "which", "where",
        "whereis", "type", "command",
        # text processing (read-only forms; sed -i is caught below)
        "sort", "uniq", "cut", "tr", "column", "diff", "comm", "join", "paste",
        "jq", "yq", "xxd", "od", "strings",
        # system facts
        "ps", "top", "whoami", "id", "groups", "uname", "hostname", "date", "uptime",
        "env", "printenv", "locale", "arch", "nproc", "sysctl",
        # hashes
        "md5", "md5sum", "sha1sum", "sha256sum", "cksum",
        # version probes
        "true", "false", "sleep", "test",
    }
)

#: Prefixes that wrap another command without changing what it does.
TRANSPARENT_PREFIXES = frozenset({"time", "nohup", "nice", "ionice", "stdbuf", "command", "exec"})

#: Wrappers that escalate: whatever follows runs with more authority.
ESCALATING = frozenset({"sudo", "doas", "runas", "su"})

#: Shell operators that separate one command from the next.
_SPLIT = re.compile(r"\|\||&&|;|\||\n")

#: Redirections that write to a file. `2>&1` is not one of them.
_REDIRECT = re.compile(r"(?<![0-9<>])>{1,2}(?!&)")

#: Flags that turn an otherwise read-only command into a writing one.
_WRITING_FLAGS: dict[str, tuple[tuple[str, str], ...]] = {
    "sed": ((("-i", "--in-place"), "edits files in place"),),
    "find": ((("-delete", "-exec", "-execdir", "-ok"), "runs actions on what it finds"),),
    "grep": (((), ""),),
    "sort": ((("-o",), "writes its output to a file"),),
    "tar": ((("-x", "--extract", "-c", "--create"), "writes an archive or its contents"),),
    "rsync": (((), ""),),
}

#: Commands whose whole purpose is a side effect. Listed so the reason given to
#: the user is specific rather than "not recognised".
KNOWN_MUTATING: dict[str, str] = {
    "rm": "deletes files",
    "rmdir": "removes directories",
    "unlink": "deletes a file",
    "mv": "moves or renames files",
    "cp": "copies over files",
    "dd": "writes raw data",
    "truncate": "resizes files",
    "touch": "creates files",
    "mkdir": "creates directories",
    "ln": "creates links",
    "chmod": "changes permissions",
    "chown": "changes ownership",
    "chgrp": "changes group ownership",
    "kill": "terminates processes",
    "pkill": "terminates processes by name",
    "killall": "terminates processes by name",
    "systemctl": "controls system services",
    "service": "controls system services",
    "launchctl": "controls system services",
    "shutdown": "changes the power state",
    "apt": "installs system packages",
    "apt-get": "installs system packages",
    "yum": "installs system packages",
    "dnf": "installs system packages",
    "pacman": "installs system packages",
    "brew": "installs packages",
    "winget": "installs packages",
    "choco": "installs packages",
    "pip": "installs Python packages",
    "pip3": "installs Python packages",
    "npm": "installs or runs Node packages",
    "yarn": "installs or runs Node packages",
    "pnpm": "installs or runs Node packages",
    "cargo": "builds or installs Rust crates",
    "go": "builds or installs Go packages",
    "docker": "controls containers",
    "podman": "controls containers",
    "kubectl": "controls a Kubernetes cluster",
    "curl": "transfers data over the network",
    "wget": "downloads files",
    "ssh": "runs commands on another machine",
    "scp": "copies files over the network",
    "make": "runs arbitrary build steps",
    "python": "runs arbitrary code",
    "python3": "runs arbitrary code",
    "node": "runs arbitrary code",
    "ruby": "runs arbitrary code",
    "perl": "runs arbitrary code",
    "sh": "runs arbitrary code",
    "bash": "runs arbitrary code",
    "zsh": "runs arbitrary code",
    "powershell": "runs arbitrary code",
    "pwsh": "runs arbitrary code",
    "cmd": "runs arbitrary code",
    "pytest": "runs arbitrary test code",
    "tox": "runs arbitrary test code",
    "git": "may modify the repository",
}


def _tokenize(segment: str) -> list[str]:
    """Split a segment into words, tolerating quoting the shell would reject."""
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _head(tokens: list[str]) -> tuple[str | None, list[str], bool]:
    """Find the command being run, its arguments, and whether it escalates.

    Skips leading ``NAME=value`` assignments and transparent wrappers so
    ``env FOO=1 time ls`` classifies as ``ls``.
    """
    escalated = False
    index = 0
    skipped: str | None = None
    while index < len(tokens):
        token = tokens[index]
        if "=" in token and not token.startswith("-") and token.split("=", 1)[0].isidentifier():
            index += 1
            continue
        base = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base in ESCALATING:
            escalated = True
            index += 1
            continue
        if base in TRANSPARENT_PREFIXES or base == "env":
            skipped = base
            index += 1
            continue
        return base, tokens[index + 1 :], escalated

    # Every token was a wrapper, so the wrapper *is* the command: bare `env`
    # prints the environment, bare `time` prints timings. Both read only.
    return skipped, [], escalated


def classify_shell(command: str) -> Classification:
    """Classify a shell command line."""
    refusal = screen(command)
    if refusal is not None:
        return Classification(Risk.DENY, refusal, command.strip())

    if not command.strip():
        return Classification(Risk.CONFIRM, "empty command", command)

    worst = Classification(Risk.SAFE, "reads only")
    for raw in _SPLIT.split(command):
        segment = raw.strip()
        if not segment:
            continue
        verdict = _classify_segment(segment)
        if verdict.risk > worst.risk:
            worst = verdict
    return worst


def _classify_segment(segment: str) -> Classification:
    if _REDIRECT.search(segment):
        return Classification(Risk.CONFIRM, "redirects output into a file", segment)

    if "$(" in segment or "`" in segment:
        # Command substitution hides a second command from this analysis.
        return Classification(Risk.CONFIRM, "contains a substituted command", segment)

    tokens = _tokenize(segment)
    name, args, escalated = _head(tokens)
    if name is None:
        return Classification(Risk.CONFIRM, "could not identify the command", segment)

    if escalated:
        return Classification(Risk.CONFIRM, f"runs {name} with elevated privileges", segment)

    if name == "git":
        return _classify_git_shell(args, segment)

    for flags, why in _WRITING_FLAGS.get(name, ()):
        if flags and any(a in flags or a.split("=")[0] in flags for a in args):
            return Classification(Risk.CONFIRM, f"{name} {why}", segment)

    if name in READ_ONLY_COMMANDS:
        return Classification(Risk.SAFE, f"{name} reads only")

    if name in KNOWN_MUTATING:
        return Classification(Risk.CONFIRM, f"{name} {KNOWN_MUTATING[name]}", segment)

    # Fail closed. An unrecognised command is not a safe command - but flag it
    # as unmatched so the adjudication layer can offer a second opinion.
    return Classification(
        Risk.CONFIRM,
        f"{name} is not a known read-only command",
        segment,
        unmatched=True,
    )


#: Branch names whose history is shared, so rewriting it hurts other people.
DEFAULT_BRANCHES = frozenset({"main", "master", "develop", "trunk", "release"})

#: The force flags proper. `--force-with-lease` is deliberately absent: it
#: refuses to overwrite work it has not seen, which is the safe way to do this.
_FORCE_FLAGS = frozenset({"--force", "-f"})


def _classify_git_shell(args: list[str], segment: str) -> Classification:
    subcommand = next((a for a in args if not a.startswith("-")), None)
    if subcommand is None:
        return Classification(Risk.SAFE, "git with no subcommand")
    if subcommand in GIT_READ_ONLY:
        return Classification(Risk.SAFE, f"git {subcommand} reads only")
    if subcommand == "push":
        return _classify_git_push(args, segment)
    return Classification(Risk.CONFIRM, f"git {subcommand} may modify the repository", segment)


def _classify_git_push(args: list[str], segment: str) -> Classification:
    """Force-pushing is routine on a feature branch and destructive on a shared one.

    Counting positional arguments is why this is not a regex: ``git push
    --force origin feature-x`` names its target and is confirmable, while
    ``git push --force`` does not, and the current branch cannot be known from
    the command string. The unnamed case is refused rather than guessed at.
    """
    if not any(a in _FORCE_FLAGS for a in args):
        return Classification(Risk.CONFIRM, "git push contacts a remote", segment)

    positional = [a for a in args if not a.startswith("-")]
    refspecs = positional[2:]  # skip "push" and the remote

    if not refspecs:
        return Classification(
            Risk.DENY,
            "force push with no branch named - the current branch may be a shared one",
            segment,
        )
    if any(ref.split(":")[-1] in DEFAULT_BRANCHES for ref in refspecs):
        return Classification(Risk.DENY, "force push to a shared default branch", segment)
    return Classification(
        Risk.CONFIRM, f"force push rewrites history on {', '.join(refspecs)}", segment
    )


#: Button labels that name a consequential act rather than a navigation.
#:
#: Clicking is normally free - moving through a user interface is how you find
#: out what is there, and confirming every click would be the alarm fatigue this
#: module's docstring warns about. But a handful of buttons do something a user
#: would want to be asked about first, and they are recognisable by their label,
#: because interfaces are designed so a *person* can recognise them.
#:
#: Written as whole words on purpose: ``\bdelete\b`` matches the "Delete" button
#: and not the "Deleted Items" folder, and ``\bsend\b`` matches "Send" and not
#: "Sent Mail". Navigating to a folder full of deleted things is not a delete.
_CONSEQUENTIAL_LABEL = re.compile(
    r"\b("
    r"delete|remove|erase|uninstall|discard|revoke|deactivate|"
    r"send|submit|publish|post|share|"
    r"buy|purchase|pay|checkout|order|subscribe|"
    r"format|reset|restore\s+defaults|factory\s+reset|"
    r"shut\s*down|restart|log\s*out|sign\s*out|"
    r"empty\s+(the\s+)?(trash|bin)|move\s+to\s+(the\s+)?(trash|bin)|"
    r"don'?t\s+save|don'?t\s+keep|revert|overwrite|replace"
    r")\b",
    re.I,
)


#: Extensions whose files *run* when opened rather than being displayed.
#:
#: This exists because "clicking navigates the interface" is not true in a file
#: manager. UI Automation's Invoke on an Explorer list item is a double click:
#: it opens the file with its default handler, so clicking `setup.exe` installs
#: something. A document opening in a viewer is the ordinary case and stays
#: silent; a file that executes is worth one question.
EXECUTABLE_SUFFIXES = frozenset(
    {
        # Windows
        "exe", "msi", "msix", "appx", "bat", "cmd", "com", "scr", "pif",
        "ps1", "psm1", "vbs", "vbe", "js", "jse", "wsf", "wsh", "hta",
        "reg", "lnk", "inf", "cpl", "jar",
        # macOS
        "app", "command", "pkg", "dmg", "scpt", "sh", "tool",
    }
)

_FILENAME = re.compile(r"^(?P<stem>[^\\/:*?\"<>|]+)\.(?P<suffix>[A-Za-z0-9]{1,8})$")


def executable_label(label: str) -> str:
    """The suffix if this label ends in a file that would run, else ``""``.

    Spaces are allowed in the stem, because real files have them - "Setup
    Wizard.exe" is a file, and missing it would be the expensive mistake. The
    cost of the other direction is one confirmation on a button whose label
    happens to end that way, and those turn out to be buttons like "Open
    setup.exe" and "Run install.msi", where confirming is the right answer
    anyway.
    """
    match = _FILENAME.match(label.strip())
    if match is None:
        return ""
    suffix = match.group("suffix").lower()
    return suffix if suffix in EXECUTABLE_SUFFIXES else ""


#: Control types that are a file in a list rather than a control in a window.
#: Clicking one of these in a file manager runs or opens whatever it names.
FILE_ITEM_TYPES = frozenset({"ListItem", "TreeItem", "DataItem"})

#: Processes where a list item is a file. Not a general "is this a file manager"
#: test - just the shells whose items launch on activation.
FILE_MANAGERS = frozenset({"explorer.exe", "explorer", "finder"})


def _is_file_item(arguments: dict) -> bool:
    """Is this click on a file in a file manager, rather than a control?

    Both halves are required. A ListItem in a mail client is a message, and
    confirming every one of those would train people to say yes without reading
    - which costs more safety than it buys.
    """
    control_type = str(arguments.get("control_type", "")).strip()
    process = str(arguments.get("process", "")).strip().casefold()
    if not control_type or not process:
        return False
    return control_type in FILE_ITEM_TYPES and process in FILE_MANAGERS


def classify_click(arguments: dict) -> Classification:
    """Classify a click by what it will actually do.

    ``label`` is what the model was shown. ``filename``, when the tool supplies
    it, is what the element actually points at - and the two differ on Windows,
    where Explorer strips known extensions from the name it reports. Preferring
    the filename is the whole reason this function is not a one-liner over the
    label: an installer whose label reads ``setup`` must not be graded the same
    as a button that happens to say ``setup``.
    """
    label = str(arguments.get("label", "")).strip()
    filename = str(arguments.get("filename", "")).strip()
    index = arguments.get("index")
    target = f"element {index} ({label!r})" if label else f"element {index}"

    if str(arguments.get("button", "")) == "right":
        # Checked before the label, because a right click never performs the
        # thing the label names - it opens a menu. Whatever gets picked from
        # that menu arrives here as its own click, with its own label, and is
        # classified then. Asking about a right click on "Delete" would be
        # asking about the wrong action.
        return Classification(Risk.SAFE, "opens a context menu")

    # The real filename first, then the label. On Windows the label is the
    # extension-stripped display name and cannot answer this question at all.
    named = filename or label
    suffix = executable_label(named)
    if suffix:
        return Classification(
            Risk.CONFIRM,
            f"{named} is a .{suffix} - clicking it in a file manager runs it, "
            "and nothing here can read what it does",
            target,
        )

    if not filename and _is_file_item(arguments):
        # A file in a file manager whose extension nobody can see. Activating it
        # runs whatever it is, and "probably a document" is not a safety
        # argument - the cost of asking is one prompt, the cost of guessing
        # wrong is an unknown binary. Supplying the filename returns this to the
        # ordinary rule above, where a .txt stays silent.
        return Classification(
            Risk.CONFIRM,
            f"{label or 'this item'} is a file and its extension is hidden, "
            "so there is no way to tell a document from a program",
            target,
        )

    match = _CONSEQUENTIAL_LABEL.search(label)
    if match:
        return Classification(
            Risk.CONFIRM,
            f"clicking {label!r} - {match.group(0).lower()} is not something I can undo",
            target,
        )
    return Classification(Risk.SAFE, f"clicking {label or target} navigates the interface")


def classify_keys(arguments: dict) -> Classification:
    """Classify a shortcut by whether it discards something."""
    from ..desktop.keys import UnknownKey, parse_sequence

    raw = str(arguments.get("keys", ""))
    try:
        chords = parse_sequence(raw)
    except UnknownKey as exc:
        return Classification(Risk.CONFIRM, f"could not read the shortcut: {exc}", raw)

    for chord in chords:
        if chord.destructive_hint:
            return Classification(
                Risk.CONFIRM, f"{chord} deletes without going through the trash", raw
            )
    return Classification(Risk.SAFE, f"{raw} is an ordinary shortcut")


def classify_open_app(arguments: dict) -> Classification:
    """Opening an application is routine; opening a terminal is a shell."""
    from ..tools.desktop import TERMINAL_APPS

    name = str(arguments.get("name", "")).strip()
    if name.lower().removesuffix(".exe") in TERMINAL_APPS:
        return Classification(
            Risk.CONFIRM,
            f"{name} is a command line. Anything run there bypasses the checks "
            "that shell commands go through",
            name,
        )
    return Classification(Risk.SAFE, f"opening {name} starts an application")


def classify_git(subcommand: str, args: list[str] | None = None) -> Classification:
    """Classify a call to the structured git tool."""
    sub = subcommand.strip()
    rendered = " ".join(["git", sub, *(args or [])])
    if sub in GIT_READ_ONLY:
        return Classification(Risk.SAFE, f"git {sub} reads only")
    if sub in {"push", "pull", "fetch", "clone"}:
        return Classification(Risk.CONFIRM, f"git {sub} contacts a remote", rendered)
    return Classification(Risk.CONFIRM, f"git {sub} modifies the repository", rendered)


def classify(tool_name: str, arguments: dict, *, mutating: bool) -> Classification:
    """Classify any tool call.

    Tools with no dedicated rule fall back on the ``mutating`` flag they
    declared at registration, which is itself fail-closed.
    """
    if tool_name == "shell":
        return classify_shell(str(arguments.get("command", "")))
    if tool_name == "git":
        return classify_git(
            str(arguments.get("subcommand", "")),
            [str(a) for a in arguments.get("args") or []],
        )
    if tool_name == "read_file":
        return Classification(Risk.SAFE, "reads a file")
    if tool_name == "click":
        return classify_click(arguments)
    if tool_name == "press_keys":
        return classify_keys(arguments)
    if tool_name == "open_app":
        return classify_open_app(arguments)
    if tool_name == "type_text":
        # Typing is reversible - select all, delete - and the thing that makes
        # it consequential is the button pressed afterwards, which arrives here
        # as its own call. Submitting is the exception: return is the button.
        if arguments.get("submit"):
            return Classification(
                Risk.SAFE, "types and presses return, which the target may act on"
            )
        return Classification(Risk.SAFE, "types into a field")
    if tool_name in {"screen_read", "scroll"}:
        return Classification(Risk.SAFE, "reads the screen")
    if mutating:
        return Classification(Risk.CONFIRM, f"{tool_name} is declared as mutating")
    return Classification(Risk.SAFE, f"{tool_name} does not modify anything")


def describe(command: str) -> str:
    """A short spoken-friendly rendering of what is about to happen."""
    collapsed = " ".join(command.split())
    return collapsed if len(collapsed) <= 120 else collapsed[:117] + "..."
