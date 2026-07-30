"""P5: keys, actuation, and the tools that expose them.

Every test here runs on any platform. The Windows and macOS actuators are
thin - they translate one call into one OS call - and everything with a
decision in it lives in the façade above them, driven here through
:class:`FakeActuator`. That is the same trade P4 made with ``FakeBackend``, and
it is why a Windows-only bug in the tree walk was catchable on a Mac.
"""

from __future__ import annotations

import pytest

from victor.desktop import keys as keymap
from victor.desktop.actions import (
    Desktop,
    FakeActuator,
    normalise_label,
)
from victor.desktop.elements import Rect
from victor.desktop.uia import FakeBackend, FakeNode, TreeReader
from victor.safety.classify import Risk, classify
from victor.safety.journal import plan_undo
from victor.safety.killswitch import Aborted, KillSwitch
from victor.tools.desktop import build_desktop_tools, looks_like_terminal


def tree(*buttons: str) -> FakeNode:
    """A window of buttons, at plausible rectangles."""
    return FakeNode(
        "Window",
        "Test Window",
        Rect(0, 0, 800, 600),
        children=[
            FakeNode("Button", name, Rect(10, 40 * i, 110, 40 * i + 30))
            for i, name in enumerate(buttons, start=1)
        ],
    )


def desktop(*buttons: str, **kwargs) -> tuple[Desktop, FakeActuator]:
    actuator = FakeActuator()
    reader = TreeReader(FakeBackend(tree(*buttons), title="Test Window"))
    return Desktop(reader, actuator, settle=0.0, **kwargs), actuator


# --- key vocabulary --------------------------------------------------------


def test_mod_is_the_platform_shortcut_key():
    assert keymap.parse("mod+s", system="Darwin").modifiers == ("cmd",)
    assert keymap.parse("mod+s", system="Windows").modifiers == ("ctrl",)


def test_cmd_stays_the_command_key_on_both():
    """`mod` picks a modifier; `cmd` names one. Win+R must stay expressible."""
    assert keymap.parse("cmd+r", system="Windows").modifiers == ("cmd",)
    assert keymap.parse("win+r", system="Windows").modifiers == ("cmd",)


def test_modifier_spellings_collapse():
    for spelling in ("ctrl", "control", "Ctl"):
        assert keymap.parse(f"{spelling}+a", system="Windows").modifiers == ("ctrl",)
    for spelling in ("alt", "option", "opt"):
        assert keymap.parse(f"{spelling}+a", system="Darwin").modifiers == ("alt",)


def test_modifier_order_is_canonical():
    """The same chord has one spelling, whatever order the model wrote it in."""
    a = keymap.parse("shift+ctrl+alt+p", system="Windows")
    b = keymap.parse("alt+shift+ctrl+p", system="Windows")
    assert a == b
    assert str(a) == "ctrl+alt+shift+p"


def test_key_aliases():
    assert keymap.parse("enter").key == "return"
    assert keymap.parse("esc").key == "escape"
    assert keymap.parse("pgdn").key == "pagedown"


def test_plus_is_a_key_not_only_a_separator():
    assert keymap.parse("mod++", system="Darwin") == keymap.Chord("+", ("cmd",))
    assert keymap.parse("+").key == "+"


def test_a_chord_ending_in_a_modifier_is_refused():
    """Silently dropping a modifier turns 'select all and delete' into 'delete'."""
    with pytest.raises(keymap.UnknownKey):
        keymap.parse("ctrl+shift")
    with pytest.raises(keymap.UnknownKey):
        keymap.parse("mod")


def test_unknown_names_raise_rather_than_approximate():
    with pytest.raises(keymap.UnknownKey):
        keymap.parse("hyper+q")
    with pytest.raises(keymap.UnknownKey):
        keymap.parse("")


def test_sequences_split_on_spaces():
    chords = keymap.parse_sequence("mod+a delete", system="Darwin")
    assert [str(c) for c in chords] == ["cmd+a", "delete"]


def test_both_platforms_can_press_every_shared_key():
    """A key name in the vocabulary must work on both machines, or it is a trap."""
    for name in keymap.known_keys():
        chord = keymap.Chord(name)
        assert keymap.mac_keycode(chord)[0] >= 0
        assert keymap.windows_keycodes(chord)[1] > 0


def test_backspace_and_delete_mean_the_same_thing_on_both():
    """macOS labels its backspace key 'delete'; following the keycap would make
    the same chord do different things on the two machines."""
    assert keymap.MAC_KEYCODES["backspace"] == 51  # the key labelled "delete"
    assert keymap.MAC_KEYCODES["delete"] == 117  # forward delete
    assert keymap.WINDOWS_KEYCODES["backspace"] == 0x08
    assert keymap.WINDOWS_KEYCODES["delete"] == 0x2E


def test_mac_flags_combine():
    _, flags = keymap.mac_keycode(keymap.parse("ctrl+shift+a", system="Darwin"))
    assert flags == keymap.MAC_MODIFIER_FLAGS["ctrl"] | keymap.MAC_MODIFIER_FLAGS["shift"]


# --- index resolution ------------------------------------------------------


def test_click_presses_through_the_accessibility_handle():
    desk, actuator = desktop("Compose", "Settings")
    result = desk.click(1, "Settings")
    assert result.ok
    assert result.method == "accessibility"
    assert actuator.calls == [("press", (1, "Settings"))]


def test_click_falls_back_to_the_rectangle_centre():
    """Only when the control offers no action - and the point comes from the OS."""
    desk, actuator = desktop("Compose", "Settings")
    actuator.pressable = False
    result = desk.click(0, "Compose")
    assert result.ok
    assert result.method == "synthetic"
    assert actuator.calls[-1] == ("click_point", (60, 55, "left", 1))


def test_a_moved_index_is_refused_not_guessed():
    desk, actuator = desktop("Compose", "Settings")
    result = desk.click(0, "Settings")
    assert not result.ok
    assert "not 'Settings'" in result.detail
    assert not any(call[0] in {"press", "click_point"} for call in actuator.calls)


def test_a_moved_index_names_where_the_target_went():
    """One step to recover, rather than a retry of the same wrong index."""
    desk, _ = desktop("Compose", "Settings")
    assert "now element 1" in desk.click(0, "Settings").detail


def test_labels_are_compared_leniently():
    """Accelerators, ellipses and whitespace are not real differences."""
    assert normalise_label("&Save As...") == normalise_label("Save As")
    assert normalise_label("  Send   Mail ") == normalise_label("send mail")
    desk, _ = desktop("&Print...")
    assert desk.click(0, "Print").ok


def test_an_index_that_does_not_exist_says_the_range():
    desk, _ = desktop("Compose")
    assert "0 to 0" in desk.click(9, "Compose").detail


def test_a_disabled_control_is_not_clicked():
    actuator = FakeActuator()
    root = FakeNode(
        "Window",
        "W",
        Rect(0, 0, 100, 100),
        children=[FakeNode("Button", "Archive", Rect(0, 0, 50, 20), enabled=False)],
    )
    desk = Desktop(TreeReader(FakeBackend(root)), actuator, settle=0.0)
    result = desk.click(0, "Archive")
    assert not result.ok and "disabled" in result.detail
    assert actuator.calls == []


def test_the_result_describes_the_screen_afterwards():
    """Saves a whole perception round trip, and every round trip is an API call."""
    desk, _ = desktop("Compose", "Settings")
    result = desk.click(0, "Compose")
    assert result.window == "Test Window"
    assert result.element_count == 2
    assert "Now showing" in result.for_model()


# --- typing ----------------------------------------------------------------


def test_typing_into_a_field_writes_the_value_directly():
    desk, actuator = desktop("Search")
    result = desk.type_text("invoices", into=0, expect="Search")
    assert result.ok
    assert actuator.calls == [("set_value", (0, "invoices"))]


def test_typing_falls_back_to_keystrokes_without_a_value_pattern():
    desk, actuator = desktop("Search")
    actuator.pressable = False
    desk.type_text("invoices", into=0, expect="Search")
    assert ("type_text", ("invoices",)) in actuator.calls


def test_submit_presses_return_afterwards():
    desk, actuator = desktop("Search")
    result = desk.type_text("invoices", submit=True)
    assert result.ok and "pressed return" in result.detail
    assert ("key", ("return",)) in actuator.calls


def test_long_text_is_typed_in_pieces():
    """So a stop lands mid-string rather than after the paragraph."""
    desk, actuator = desktop("Note")
    desk.type_text("x" * 100)
    typed = [call for call in actuator.calls if call[0] == "type_text"]
    assert len(typed) > 1
    assert "".join(call[1][0] for call in typed) == "x" * 100


# --- the kill switch -------------------------------------------------------


def test_a_tripped_switch_stops_a_click_before_it_happens():
    switch = KillSwitch()
    actuator = FakeActuator()
    desk = Desktop(
        TreeReader(FakeBackend(tree("Compose"))), actuator, settle=0.0, kill_switch=switch
    )
    switch.trip("test")
    with pytest.raises(Aborted):
        desk.click(0, "Compose")
    assert actuator.calls == []


def test_aborting_releases_modifiers_first():
    """A stop that leaves Ctrl held is worse than the action it prevented."""
    switch = KillSwitch()
    actuator = FakeActuator()
    desk = Desktop(
        TreeReader(FakeBackend(tree("Compose"))), actuator, settle=0.0, kill_switch=switch
    )
    switch.trip("test")
    with pytest.raises(Aborted):
        desk.press_keys("ctrl+c")
    assert actuator.released >= 1


def test_a_stop_lands_between_chords():
    switch = KillSwitch()
    actuator = FakeActuator()
    desk = Desktop(
        TreeReader(FakeBackend(tree("Note"))), actuator, settle=0.0, kill_switch=switch
    )
    real_key = actuator.key

    def trip_after_first(chord):
        switch.trip("mid-sequence")
        return real_key(chord)

    actuator.key = trip_after_first
    with pytest.raises(Aborted):
        # ctrl rather than mod, so the expected chord does not depend on which
        # machine the suite is running on.
        desk.press_keys("ctrl+a delete")
    assert [c for c in actuator.calls if c[0] == "key"] == [("key", ("ctrl+a",))]


def test_modifiers_are_released_after_every_key_sequence():
    desk, actuator = desktop("Note")
    desk.press_keys("mod+s")
    assert actuator.released >= 1


# --- classification --------------------------------------------------------


def test_an_ordinary_click_does_not_ask():
    """Confirming every click is the alarm fatigue the classifier exists to avoid."""
    verdict = classify("click", {"index": 3, "label": "Compose"}, mutating=True)
    assert verdict.risk is Risk.SAFE


@pytest.mark.parametrize(
    "label",
    [
        "Delete",
        "Delete Message",
        "Remove account",
        "Send",
        "Empty Trash",
        "Move to Trash",
        "Uninstall",
        "Buy now",
        "Confirm purchase",
        "Sign out",
        "Don't Save",
        "Factory reset",
        # Met in the wild: TextEdit's "could not be autosaved" alert offers
        # Save Anyway / Save As… / Revert, and Revert throws the edits away.
        "Revert",
    ],
)
def test_a_consequential_button_asks_first(label):
    assert classify("click", {"index": 1, "label": label}, mutating=True).risk is Risk.CONFIRM


@pytest.mark.parametrize("label", ["Deleted Items", "Sent Mail", "Trash can icon", "Reposts"])
def test_navigating_to_a_folder_is_not_the_act_it_is_named_after(label):
    """`\\bdelete\\b` matches the Delete button, not the Deleted Items folder."""
    assert classify("click", {"index": 1, "label": label}, mutating=True).risk is Risk.SAFE


@pytest.mark.parametrize(
    "label", ["setup.exe", "install.msi", "script.ps1", "run.bat", "Victor.app", "thing.cmd"]
)
def test_clicking_a_file_that_would_run_asks_first(label):
    """Invoke on an Explorer list item opens the file - setup.exe installs."""
    verdict = classify("click", {"index": 4, "label": label}, mutating=True)
    assert verdict.risk is Risk.CONFIRM
    assert "runs it" in verdict.reason


@pytest.mark.parametrize(
    "label", ["notes.txt", "report.pdf", "photo.png", "data.csv", "index.html"]
)
def test_clicking_a_document_stays_silent(label):
    """Opening a document in a viewer is the ordinary case."""
    assert classify("click", {"index": 4, "label": label}, mutating=True).risk is Risk.SAFE


def test_a_filename_with_spaces_is_still_a_filename():
    """Real files have spaces, and missing one is the expensive direction.

    A button whose label ends the same way - "Open setup.exe" - is caught too,
    which is correct rather than merely tolerable: it also runs the thing.
    """
    from victor.safety.classify import executable_label

    assert executable_label("Setup Wizard.exe") == "exe"
    assert executable_label("Open setup.exe") == "exe"
    assert executable_label("Downloads") == ""
    assert executable_label("README.md") == ""
    assert executable_label("Version 2.0") == ""


# --- Windows hides the extension the whole rule depends on ------------------
#
# Measured on a stock Windows 11: HideFileExt is 1, so Explorer's accessibility
# name for setup.exe is "setup". Every test above passes a label with an
# extension, which is a label Windows never supplies - so the rule was correct,
# fully tested, and never fired on the platform it was written for.


def test_the_real_filename_beats_the_stripped_label():
    """The label is what the model was shown; the filename is what it points
    at. On Windows those differ, and only one of them can answer the question."""
    verdict = classify(
        "click",
        {"index": 4, "label": "setup", "filename": "setup.exe"},
        mutating=True,
    )
    assert verdict.risk is Risk.CONFIRM
    assert "setup.exe is a .exe" in verdict.reason


def test_a_document_stays_silent_when_the_filename_is_known():
    """The position agreed with Gagan: a .txt click must not nag."""
    verdict = classify(
        "click",
        {
            "index": 4,
            "label": "notes",
            "filename": "notes.txt",
            "control_type": "ListItem",
            "process": "explorer.exe",
        },
        mutating=True,
    )
    assert verdict.risk is Risk.SAFE


def test_a_file_whose_extension_nobody_can_see_asks():
    """The fallback for when UIA carries no filename either. Activating a file
    manager item runs whatever it is, and "probably a document" is not a safety
    argument - the cost of asking is one prompt."""
    verdict = classify(
        "click",
        {"index": 71, "label": "setup", "control_type": "ListItem", "process": "explorer.exe"},
        mutating=True,
    )
    assert verdict.risk is Risk.CONFIRM
    assert "extension is hidden" in verdict.reason


def test_a_list_item_outside_a_file_manager_is_not_a_file():
    """A ListItem in a mail client is a message. Confirming every one of those
    would teach people to say yes without reading, which costs more safety than
    it buys."""
    verdict = classify(
        "click",
        {"index": 8, "label": "Re: lunch", "control_type": "ListItem", "process": "outlook.exe"},
        mutating=True,
    )
    assert verdict.risk is Risk.SAFE


def test_a_toolbar_button_is_not_treated_as_a_hidden_file():
    """Most of what the agent clicks in Explorer is the toolbar, and none of it
    should start asking."""
    verdict = classify(
        "click",
        {"index": 3, "label": "Refresh", "control_type": "Button", "process": "explorer.exe"},
        mutating=True,
    )
    assert verdict.risk is Risk.SAFE


def test_element_recovers_the_filename_explorer_hid():
    """UIA still knows the whole name even when the display name is stripped -
    in the value or the automation id, depending on the view."""
    from victor.desktop import Element, Rect

    hidden_in_value = Element(
        1, "ListItem", "setup", Rect(0, 0, 10, 10), value=r"C:\Users\g\Downloads\setup.exe"
    )
    hidden_in_id = Element(
        2, "ListItem", "install", Rect(0, 0, 10, 10), automation_id="install.bat"
    )
    plain = Element(3, "ListItem", "notes.txt", Rect(0, 0, 10, 10))
    a_button = Element(4, "Button", "Refresh", Rect(0, 0, 10, 10))

    assert hidden_in_value.filename == "setup.exe"
    assert hidden_in_id.filename == "install.bat"
    assert plain.filename == "notes.txt"
    assert a_button.filename == ""


def test_an_unrelated_value_is_not_mistaken_for_the_filename():
    """An Edit box's value is its contents, not a filename. Matching the stem
    against the display name is what keeps this from inventing one."""
    from victor.desktop import Element, Rect

    edit = Element(1, "Edit", "Search", Rect(0, 0, 10, 10), value="report.docx")
    assert edit.filename == ""


def test_a_stripped_label_does_not_satisfy_an_executable():
    """The other half of the hole. Label matching is deliberately forgiving, so
    "setup" was a substring match for "setup.exe" and sailed straight through
    the re-verification that exists to catch exactly this."""
    target, _ = desktop("setup.exe", "notes.txt")
    result = target.click(0, "setup")

    assert not result.ok
    assert "would run" in result.detail
    assert "Name it in full" in result.detail


def test_naming_the_executable_in_full_still_works():
    """The refusal has to be escapable, or the agent cannot ever run an
    installer the user actually asked for."""
    target, actuator = desktop("setup.exe", "notes.txt")
    assert target.click(0, "setup.exe").ok


def test_a_stripped_label_is_still_fine_for_a_document():
    """Only the executable case is tightened. Ordinary labels gain and lose
    decoration between reads, and refusing those would break normal use."""
    target, _ = desktop("notes.txt")
    assert target.click(0, "notes").ok
    """Whatever gets picked from the menu is classified when it is clicked."""
    verdict = classify("click", {"index": 1, "label": "Delete", "button": "right"}, mutating=True)
    assert verdict.risk is Risk.SAFE


def test_ordinary_shortcuts_pass_and_destructive_ones_ask():
    assert classify("press_keys", {"keys": "mod+s"}, mutating=True).risk is Risk.SAFE
    assert classify("press_keys", {"keys": "mod+delete"}, mutating=True).risk is Risk.CONFIRM
    assert classify("press_keys", {"keys": "shift+delete"}, mutating=True).risk is Risk.CONFIRM


def test_an_unreadable_shortcut_fails_closed():
    assert classify("press_keys", {"keys": "hyper+q"}, mutating=True).risk is Risk.CONFIRM


def test_reading_the_screen_is_free():
    assert classify("screen_read", {}, mutating=False).risk is Risk.SAFE
    assert classify("scroll", {"direction": "down"}, mutating=False).risk is Risk.SAFE


def test_opening_a_terminal_asks_because_it_is_a_shell():
    assert classify("open_app", {"name": "Chrome"}, mutating=True).risk is Risk.SAFE
    assert classify("open_app", {"name": "Terminal"}, mutating=True).risk is Risk.CONFIRM
    assert classify("open_app", {"name": "powershell.exe"}, mutating=True).risk is Risk.CONFIRM


def test_a_click_has_no_undo_and_says_so():
    undo, why_not = plan_undo("click", {"index": 1, "label": "Send"})
    assert undo is None
    assert "cannot be un-clicked" in why_not


# --- the tools -------------------------------------------------------------


def tools_over(*buttons: str, title: str = "Test Window", process: str = "test.exe"):
    actuator = FakeActuator()
    backend = FakeBackend(tree(*buttons), title=title, process=process)
    desk = Desktop(TreeReader(backend), actuator, settle=0.0)
    return {tool.spec.name: tool for tool in build_desktop_tools(desktop=desk)}, actuator


def test_screen_read_is_declared_free_and_non_mutating():
    tools, _ = tools_over("Compose")
    spec = tools["screen_read"].spec
    assert spec.mutating is False
    assert tools["screen_read"].run().metadata["cost"] == 0


def test_screen_read_filters():
    tools, _ = tools_over("Compose", "Settings", "Archive")
    output = tools["screen_read"].run(filter="sett").output
    assert "Settings" in output and "Archive" not in output


def test_click_requires_the_label_for_verification():
    tools, _ = tools_over("Compose")
    assert set(tools["click"].spec.parameters["required"]) == {"index", "label"}


def test_a_failed_action_is_a_result_not_an_exception():
    tools, _ = tools_over("Compose")
    result = tools["click"].run(index=0, label="Settings")
    assert result.ok is False
    assert "changed since you looked" in (result.error or "")


@pytest.mark.parametrize(
    ("title", "process"),
    [
        ("harshak — zsh — 80x24", "Terminal"),
        ("Windows PowerShell", "powershell.exe"),
        ("~/code", "iTerm2"),
        ("Command Prompt", "cmd.exe"),
    ],
)
def test_typing_into_a_terminal_is_refused(title, process):
    """The one hole that would undo P3: a shell no classifier ever sees."""
    tools, actuator = tools_over("Prompt", title=title, process=process)
    result = tools["type_text"].run(text="rm -rf /")
    assert result.ok is False
    assert result.metadata["refused"] == "terminal"
    assert "shell tool" in (result.error or "")
    assert actuator.calls == []


def test_shortcuts_into_a_terminal_are_refused_too():
    tools, actuator = tools_over("Prompt", title="zsh", process="Terminal")
    assert tools["press_keys"].run(keys="return").ok is False
    assert actuator.calls == []


def test_typing_into_an_ordinary_window_is_allowed():
    tools, actuator = tools_over("Search", title="Inbox - Mail", process="mail.exe")
    assert tools["type_text"].run(text="hello").ok
    assert any(call[0] == "type_text" for call in actuator.calls)


@pytest.mark.parametrize(
    "name",
    ["/bin/sh", "cmd /c del *", "Chrome; rm -rf ~", "../../evil.exe", 'a" && b'],
)
def test_open_app_takes_a_name_not_a_command_line(name):
    tools, actuator = tools_over("x")
    result = tools["open_app"].run(name=name)
    assert result.ok is False
    assert "plain application name" in (result.error or "")
    assert actuator.calls == []


def test_open_app_accepts_real_application_names():
    tools, actuator = tools_over("x")
    for name in ("Chrome", "File Explorer", "Visual Studio Code", "Notes"):
        assert tools["open_app"].run(name=name).ok
    assert len(actuator.calls) == 4


@pytest.mark.parametrize(
    ("title", "process", "expected"),
    [
        ("Inbox - Mail", "mail.exe", False),
        ("Untitled - Notepad", "notepad.exe", False),
        ("Terminal", "Terminal", True),
        ("victor — python — 120x30", "iTerm2", True),
        ("Documents", "explorer.exe", False),
    ],
)
def test_terminal_detection(title, process, expected):
    assert looks_like_terminal(title, process) is expected


# --- the real Windows actuator ---------------------------------------------
#
# Same trick as the macOS tests below: drive the real WindowsActuator against a
# fake `uiautomation`. Both defects these cover were found on Gagan's machine
# and neither is visible to FakeActuator, which records that a key was pressed
# rather than whether it was ever let go.


class FakeUIAuto:
    """Enough of `uiautomation` to press keys and focus windows."""

    def __init__(self, *, foreground: str = "", exists: bool = True) -> None:
        self.down: list[int] = []
        self.up: list[int] = []
        self.foreground = foreground
        self.exists = exists
        self.fail_on: int | None = None
        self.activated: list[str] = []

    # keyboard
    def PressKey(self, code):  # noqa: N802
        # Record the key as down *before* raising. That is the shape of the
        # real bug: the key physically went down, and something went wrong
        # before the caller noted that it had. A fake that raises first never
        # reproduces it and quietly passes against the broken code.
        self.down.append(code)
        if self.fail_on is not None and code == self.fail_on:
            raise RuntimeError("COM error mid-chord")

    def ReleaseKey(self, code):  # noqa: N802
        self.up.append(code)

    # windows
    def WindowControl(self, **kwargs):  # noqa: N802
        auto = self
        name = kwargs.get("SubName", "")

        class _Window:
            def Exists(self, **_):  # noqa: N802
                return auto.exists

            def SetActive(self):  # noqa: N802
                auto.activated.append(name)

            def SetTopmost(self, value):  # noqa: N802
                pass

        return _Window()

    def GetForegroundControl(self):  # noqa: N802
        class _Control:
            Name = self.foreground

        return _Control()

    @property
    def still_down(self) -> set[int]:
        """Keys pressed and never released - what a user would feel."""
        return {code for code in self.down if self.down.count(code) > self.up.count(code)}


def windows_actuator(**kwargs) -> tuple[object, FakeUIAuto]:
    from victor.desktop.actions import WindowsActuator

    actuator = WindowsActuator()
    auto = FakeUIAuto(**kwargs)
    actuator._auto = auto
    return actuator, auto


def test_a_normal_chord_leaves_no_key_down():
    actuator, auto = windows_actuator()
    actuator.key(keymap.parse("ctrl+shift+n", system="Windows"))
    assert auto.still_down == set()


def test_a_failure_mid_chord_leaves_no_key_down():
    """The one-statement window where a key was down but untracked.

    Reachable in practice: a COM error, or a KeyboardInterrupt - which is how
    the kill switch is triggered. On Windows a stuck Ctrl affects the whole
    machine, not just Victor.
    """
    actuator, auto = windows_actuator()
    auto.fail_on = keymap.WINDOWS_MODIFIER_CODES["shift"]

    with pytest.raises(RuntimeError):
        actuator.key(keymap.parse("ctrl+shift+n", system="Windows"))

    assert auto.still_down == set(), "a modifier was left physically held"
    assert keymap.WINDOWS_MODIFIER_CODES["ctrl"] in auto.up


def test_releasing_does_not_depend_on_the_bookkeeping():
    """Unconditional, to match macOS - a redundant release costs nothing."""
    actuator, auto = windows_actuator()
    auto.down.append(keymap.WINDOWS_MODIFIER_CODES["alt"])  # pressed off-book
    actuator.release_modifiers()
    assert auto.still_down == set()


def test_focus_app_reports_failure_when_the_window_did_not_come_forward():
    """It used to claim success while the user's editor stayed in front."""
    actuator, _ = windows_actuator(foreground="Victor - Visual Studio Code")
    result = actuator.focus_app("Downloads")
    assert result.ok is False
    assert "Visual Studio Code" in result.detail
    assert "foreground" in result.detail


def test_focus_app_succeeds_when_the_window_really_is_in_front():
    actuator, auto = windows_actuator(foreground="Downloads - File Explorer")
    result = actuator.focus_app("Downloads")
    assert result.ok is True
    assert auto.activated == ["Downloads"]


def test_a_missing_window_is_still_reported_as_missing():
    actuator, _ = windows_actuator(exists=False)
    result = actuator.focus_app("Nope")
    assert result.ok is False
    assert "no window matches" in result.detail


def test_a_running_app_that_will_not_focus_is_not_launched_twice():
    """focus_app now fails for two reasons; only one of them means "launch"."""
    actuator, auto = windows_actuator(foreground="Something Else", exists=True)
    result = actuator.launch_app("Downloads")
    assert result.ok is False
    assert "foreground" in result.detail


# --- flag discipline in the real macOS actuator ----------------------------
#
# FakeActuator cannot catch this class of bug: it records that a key was
# pressed, not what state the key carried. So these drive the real MacActuator
# against a fake Quartz, which works on any platform because the actuator only
# touches the module through the handle it cached.


class FakeQuartz:
    """Records every posted event, and what flags it carried."""

    kCGHIDEventTap = "hid"
    kCGScrollEventUnitLine = 1
    kCGEventMouseMoved = "moved"

    def __init__(self) -> None:
        self.posted: list[dict] = []

    def CGEventCreateKeyboardEvent(self, source, code, down):  # noqa: N802
        return {"code": code, "down": down, "flags": None, "text": ""}

    def CGEventSetFlags(self, event, flags):  # noqa: N802
        event["flags"] = flags

    def CGEventKeyboardSetUnicodeString(self, event, length, text):  # noqa: N802
        event["text"] = text

    def CGEventPost(self, tap, event):  # noqa: N802
        self.posted.append(event)


def mac_actuator() -> tuple[object, FakeQuartz]:
    from victor.desktop.actions import MacActuator

    actuator = MacActuator()
    quartz = FakeQuartz()
    actuator._quartz = quartz
    actuator._services = object()
    actuator._source = "source"
    return actuator, quartz


def test_a_chord_key_up_carries_no_flags():
    """The leak that made typing after cmd+a silently do nothing."""
    actuator, quartz = mac_actuator()
    actuator.key(keymap.Chord("a", ("cmd",)))
    down, up = quartz.posted[0], quartz.posted[1]
    assert down["down"] is True
    assert down["flags"] == keymap.MAC_MODIFIER_FLAGS["cmd"]
    assert up["down"] is False
    assert up["flags"] == 0


def test_a_chord_is_followed_by_real_modifier_releases():
    actuator, quartz = mac_actuator()
    actuator.key(keymap.Chord("a", ("cmd",)))
    releases = quartz.posted[2:]
    assert {e["code"] for e in releases} == set(actuator._MODIFIER_KEYCODES)
    assert all(e["down"] is False and e["flags"] == 0 for e in releases)


def test_every_typed_event_clears_the_flags():
    """A new CGEvent inherits the window server's modifier state."""
    actuator, quartz = mac_actuator()
    actuator.type_text("hi")
    assert [e["text"] for e in quartz.posted] == ["h", "h", "i", "i"]
    assert all(e["flags"] == 0 for e in quartz.posted)


def test_typing_sends_one_character_per_event():
    """Apps handling keyDown: take the first character and drop the rest."""
    actuator, quartz = mac_actuator()
    actuator.type_text("8*8")
    assert all(len(e["text"]) == 1 for e in quartz.posted)
    assert "".join(e["text"] for e in quartz.posted[::2]) == "8*8"


# --- is anyone looking at the screen? --------------------------------------


def test_a_locked_mac_screen_is_reported_as_such(monkeypatch):
    """Found live: a locked screen answers every AX question except geometry,
    so every rectangle is empty and the window looks like it has no controls."""
    from victor.desktop import session

    monkeypatch.setattr(session.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        session, "_mac_locked", lambda: (True, "the screen is locked. Unlock it.")
    )
    locked, why = session.session_locked()
    assert locked and "locked" in why


def test_lock_detection_fails_open(monkeypatch):
    """A probe that cannot answer must not block work on a guess."""
    from victor.desktop import session

    monkeypatch.setattr(session.platform, "system", lambda: "Linux")
    assert session.session_locked() == (False, "")


def test_an_unmeasurable_window_is_not_confused_with_an_empty_one():
    """The two look identical to a caller that only counts elements."""
    from victor.desktop.elements import Snapshot

    measured = Snapshot("W", "p", (), rect=Rect(0, 0, 800, 600))
    unmeasured = Snapshot("W", "p", (), rect=Rect(0, 0, 0, 0), note="off-screen")
    assert measured.empty and unmeasured.empty
    assert not measured.note and unmeasured.note
    assert "off-screen" in unmeasured.render()


# --- the vision fallback, and what it costs --------------------------------


class FakeVision:
    """Stands in for VisionClient. Answers with a fixed index, or refuses."""

    def __init__(self, index: int | None = 0, *, exhausted: bool = False) -> None:
        self.index = index
        self.exhausted = exhausted
        self.calls = 0

    def locate(self, request, shot, snapshot=None):
        from victor.desktop.vision import VisionAnswer, VisionUnavailable

        self.calls += 1
        if self.exhausted:
            raise VisionUnavailable("no vision quota left today")
        element = snapshot.by_index(self.index) if snapshot and self.index is not None else None
        return VisionAnswer(
            index=self.index, raw=str(self.index), model="fake/vlm", latency_ms=1.0,
            element=element,
        )


class FakeCapture:
    """A capture that never touches a screen."""

    def capture(self, region=None, *, force=False):
        from victor.desktop.capture import Screenshot

        return Screenshot(data=b"", width=10, height=10, fingerprint="0" * 16), True


def vision_tool(*buttons: str, vision: FakeVision | None = None):
    from victor.tools.desktop import FindOnScreenTool

    actuator = FakeActuator()
    desk = Desktop(TreeReader(FakeBackend(tree(*buttons), title="W")), actuator, settle=0.0)
    return FindOnScreenTool(desk, vision or FakeVision(), FakeCapture())


def test_find_on_screen_reports_what_it_spent(monkeypatch):
    monkeypatch.setattr("victor.desktop.vision.annotate", lambda shot, snap, **kw: shot)
    tool = vision_tool("Compose", "Settings")
    result = tool.run(description="the compose button")
    assert result.ok
    assert result.metadata["cost"] == 1
    assert "index 0" in result.output and "Compose" in result.output


def test_running_out_of_vision_leaves_a_working_agent(monkeypatch):
    monkeypatch.setattr("victor.desktop.vision.annotate", lambda shot, snap, **kw: shot)
    tool = vision_tool("Compose", vision=FakeVision(exhausted=True))
    result = tool.run(description="anything")
    assert result.ok is False
    assert "screen_read still works" in (result.error or "")
    assert result.metadata["quota"] == "exhausted"


def test_the_free_tools_declare_that_they_are_free():
    tools, _ = tools_over("Compose")
    assert tools["click"].run(index=0, label="Compose").metadata["cost"] == 0
    assert tools["screen_read"].run().metadata["cost"] == 0


def test_an_empty_tree_points_at_the_fallback():
    """A canvas is the case vision exists for; saying so saves a step."""
    actuator = FakeActuator()
    blank = FakeNode("Window", "Game", Rect(0, 0, 800, 600), children=[])
    desk = Desktop(TreeReader(FakeBackend(blank, title="Game")), actuator, settle=0.0)
    tools = {t.spec.name: t for t in build_desktop_tools(desktop=desk)}
    output = tools["screen_read"].run().output
    assert "no readable controls" in output and "find_on_screen" in output


def test_an_unmeasurable_window_says_so_instead_of_reporting_nothing():
    """An off-screen window and an empty one look identical without this.

    Found the hard way: Calculator reopened off the left edge of the display,
    macOS refused to report its position, every rectangle came back empty, and
    Victor cheerfully said "0 elements" - which sends you looking at the tree
    walk instead of at the window.
    """
    actuator = FakeActuator()
    offscreen = FakeNode(
        "Window",
        "Calculator",
        Rect(0, 0, 0, 0),
        children=[FakeNode("Button", "Equals", Rect(0, 0, 0, 0))],
    )
    desk = Desktop(TreeReader(FakeBackend(offscreen, title="Calculator")), actuator, settle=0.0)
    snapshot = desk.snapshot(refresh=True)
    assert snapshot.empty
    assert "off-screen" in snapshot.note
    tools = {t.spec.name: t for t in build_desktop_tools(desktop=desk)}
    assert "minimised" in tools["screen_read"].run().output


def test_vision_is_not_offered_when_it_cannot_be_served():
    """A tool that cannot work costs a step to discover, and the step is the cost."""
    actuator = FakeActuator()
    desk = Desktop(TreeReader(FakeBackend(tree("A"))), actuator, settle=0.0)
    without = {t.spec.name for t in build_desktop_tools(desktop=desk)}
    assert "find_on_screen" not in without


def test_the_zero_cost_ratio_counts_what_tools_reported():
    from victor.agent.loop import AgentResult, Outcome, Step
    from victor.tools.base import ToolResult

    def step(*costs: int) -> Step:
        calls = tuple(
            (f"call{i}", ToolResult(ok=True, metadata={"cost": c}))
            for i, c in enumerate(costs)
        )
        return Step(index=0, reply=None, calls=calls)

    result = AgentResult(
        task="t", answer="a", outcome=Outcome.ANSWERED, steps=[step(0, 0, 0), step(0, 1)]
    )
    assert result.free_tool_calls == 4
    assert result.billed_tool_calls == 1
    assert result.zero_cost_ratio == 0.8
    # Two think-act cycles plus the one billed tool call.
    assert result.api_calls == 3
    assert "4/5 tool calls free" in result.summary()


def test_a_run_that_touched_no_tools_is_not_reported_as_free():
    from victor.agent.loop import AgentResult, Outcome

    result = AgentResult(task="t", answer="a", outcome=Outcome.ANSWERED)
    assert result.zero_cost_ratio == 1.0
    assert "no tool calls" in result.summary()


def cli_registry(tmp_path, monkeypatch, *, yes: bool, buttons: tuple[str, ...]):
    """The CLI's own registry, with state pointed at a tmpdir.

    ``VICTOR_DATA_DIR`` rather than ``data_dir=``: the field has an env alias,
    so the keyword argument is silently ignored and the journal lands in the
    developer's real ``~/.victor``. Which is how this helper came to exist.
    """
    from victor.cli import _gated_desktop_tools
    from victor.config import Settings

    monkeypatch.setattr(
        "victor.cli._settings",
        lambda: Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path / "state")),
    )
    desk, _ = desktop(*buttons)
    return _gated_desktop_tools(desk, yes=yes)


def test_the_cli_click_path_is_gated_like_the_agent(tmp_path, monkeypatch):
    """`victor click` used to call Desktop.click directly - no classifier, no
    confirmation, no journal. On Windows that meant clicking an installer ran
    it, because UIA's Invoke on a file opens it."""
    registry, interceptor = cli_registry(
        tmp_path, monkeypatch, yes=False, buttons=("setup.exe", "notes.txt")
    )
    assert "click" in registry
    assert registry.interceptor is interceptor

    # No terminal to confirm on, so the gate fails closed - which is itself the
    # proof that the classifier and confirmer are in the path at all.
    blocked = registry.run("click", {"index": 0, "label": "setup.exe"})
    assert blocked.ok is False
    assert interceptor.stats.denied + interceptor.stats.refused == 1

    allowed = registry.run("click", {"index": 1, "label": "notes.txt"})
    assert allowed.ok is True, "a document click should not have been stopped"


def test_the_gate_sees_the_hidden_extension_the_model_could_not(tmp_path, monkeypatch):
    """End to end for the Windows hole. The tree carries `setup.exe`, Explorer
    would have shown the model `setup`, and the model passes `setup` - so the
    label alone cannot save this. The registry asks the tool what the index
    actually points at, and the classifier gets the filename."""
    registry, interceptor = cli_registry(
        tmp_path, monkeypatch, yes=False, buttons=("setup.exe", "notes.txt")
    )

    blocked = registry.run("click", {"index": 0, "label": "setup"})
    assert blocked.ok is False
    assert interceptor.stats.denied + interceptor.stats.refused == 1

    # And the ordinary case still does not nag.
    assert registry.run("click", {"index": 1, "label": "notes"}).ok is True


def test_a_tool_without_describe_is_unaffected(tmp_path, monkeypatch):
    """The hook is optional - every other tool must keep working untouched."""
    from victor.config import Settings
    from victor.tools import build_registry

    monkeypatch.setenv("VICTOR_DATA_DIR", str(tmp_path))
    registry = build_registry(Settings(_env_file=None, VICTOR_DATA_DIR=str(tmp_path)))
    assert not hasattr(registry.get("shell"), "describe")
    assert registry.run("read_file", {"path": "pyproject.toml"}).ok


def test_the_cli_click_path_journals(tmp_path, monkeypatch):
    registry, interceptor = cli_registry(
        tmp_path, monkeypatch, yes=True, buttons=("setup.exe",)
    )
    registry.run("click", {"index": 0, "label": "setup.exe"})
    assert [e.tool for e in interceptor.journal.recent()] == ["click"]


def test_the_desktop_tools_are_absent_unless_asked_for():
    """The capability is missing rather than discouraged."""
    from victor.config import Settings
    from victor.tools import build_registry

    settings = Settings(_env_file=None)
    assert "click" not in build_registry(settings)
    assert "click" in build_registry(settings, desktop=True)


def test_the_prompt_only_mentions_the_screen_when_it_can_see_one():
    from victor.agent.prompts import system_prompt

    environment = {"platform": "Darwin", "cwd": "/tmp", "shell": "zsh"}
    assert "screen_read" not in system_prompt(environment)
    assert "screen_read" in system_prompt(environment, desktop=True)
