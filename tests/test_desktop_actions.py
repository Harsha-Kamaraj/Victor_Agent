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
    ],
)
def test_a_consequential_button_asks_first(label):
    assert classify("click", {"index": 1, "label": label}, mutating=True).risk is Risk.CONFIRM


@pytest.mark.parametrize("label", ["Deleted Items", "Sent Mail", "Trash can icon", "Reposts"])
def test_navigating_to_a_folder_is_not_the_act_it_is_named_after(label):
    """`\\bdelete\\b` matches the Delete button, not the Deleted Items folder."""
    assert classify("click", {"index": 1, "label": label}, mutating=True).risk is Risk.SAFE


def test_a_right_click_opens_a_menu_rather_than_doing_what_the_label_says():
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
