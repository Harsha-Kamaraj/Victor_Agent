from __future__ import annotations

import httpx
import pytest

from victor.config import Settings
from victor.desktop import (
    Element,
    FakeBackend,
    FakeNode,
    Rect,
    Snapshot,
    TreeReader,
    VisionClient,
    VisionUnavailable,
    demo_tree,
    hamming,
    is_interesting,
)
from victor.desktop.uia import WalkLimits
from victor.providers import Router
from victor.providers.registry import GEMINI_25_FLASH, LLAMA_4_SCOUT
from victor.quota import QuotaLedger

# --- climbing to the owning window ----------------------------------------
#
# UIABackend.focused_window is the one part of the Windows backend with a
# decision in it, and it was wrong in a way no macOS test could see. These
# drive the real backend against a fake `uiautomation`, so they run anywhere.


class FakeUIAControl:
    """Just enough of a uiautomation control to be climbed."""

    def __init__(self, control_type: str, name: str = "") -> None:
        self.ControlTypeName = control_type
        self.Name = name
        self.ProcessId = 4321
        self.parent: FakeUIAControl | None = None

    def GetParentControl(self):  # noqa: N802
        return self.parent


class FakeUIAModule:
    def __init__(self, focused=None, foreground=None) -> None:
        self._focused = focused
        self._foreground = foreground

    def GetFocusedControl(self):  # noqa: N802
        return self._focused

    def GetForegroundControl(self):  # noqa: N802
        return self._foreground


def chain(*nodes: FakeUIAControl) -> FakeUIAControl:
    """Link deepest-first, and return the deepest."""
    for child, parent in zip(nodes, nodes[1:], strict=False):
        child.parent = parent
    return nodes[0]


def explorer_ancestry() -> FakeUIAControl:
    """The ancestry Gagan measured in File Explorer on Windows 11."""
    return chain(
        FakeUIAControl("ListItemControl", "notes.txt"),
        FakeUIAControl("ListControl", "Items View"),
        FakeUIAControl("PaneControl", "Shell Folder View"),
        FakeUIAControl("PaneControl", "Folder Layout Pane"),
        FakeUIAControl("PaneControl", "Explorer Pane"),
        FakeUIAControl("PaneControl", ""),
        FakeUIAControl("PaneControl", "Downloads"),
        FakeUIAControl("WindowControl", "Downloads - File Explorer"),
        FakeUIAControl("PaneControl", "Desktop 1"),
    )


def uia_backend(focused=None, foreground=None, app: str | None = None):
    from victor.desktop.uia import UIABackend

    backend = UIABackend(app_name=app)
    backend._auto = FakeUIAModule(focused=focused, foreground=foreground)
    return backend


def test_the_climb_passes_through_nested_panes_to_the_window():
    """Explorer nests six panes; stopping at the first cost 101 elements."""
    backend = uia_backend(focused=explorer_ancestry())
    window = backend.focused_window()
    assert window.ControlTypeName == "WindowControl"
    assert window.Name == "Downloads - File Explorer"


def test_the_climb_never_reaches_the_desktop_root():
    """The thing the old Pane rule was actually guarding against."""
    window = uia_backend(focused=explorer_ancestry()).focused_window()
    assert window.Name != "Desktop 1"


def test_the_climb_stops_at_the_innermost_window():
    """A dialog is a WindowControl inside a WindowControl - take the dialog."""
    backend = uia_backend(
        focused=chain(
            FakeUIAControl("ButtonControl", "Save"),
            FakeUIAControl("PaneControl", "content"),
            FakeUIAControl("WindowControl", "Save As"),
            FakeUIAControl("WindowControl", "Document - Word"),
        )
    )
    assert backend.focused_window().Name == "Save As"


def test_a_control_already_at_the_window_is_returned_as_is():
    window = FakeUIAControl("WindowControl", "Downloads - File Explorer")
    assert uia_backend(focused=window).focused_window() is window


def test_no_window_ancestor_falls_back_to_the_foreground():
    """Better than a node halfway up somebody's scaffolding."""
    foreground = FakeUIAControl("WindowControl", "Foreground")
    backend = uia_backend(
        focused=chain(
            FakeUIAControl("ButtonControl", "x"),
            FakeUIAControl("PaneControl", "y"),
            FakeUIAControl("PaneControl", "Desktop 1"),
        ),
        foreground=foreground,
    )
    assert backend.focused_window() is foreground


def test_app_targeting_is_honoured_on_windows():
    """`--app` used to be accepted and silently ignored by this backend."""
    wanted = FakeUIAControl("WindowControl", "Downloads - File Explorer")

    class Module(FakeUIAModule):
        def WindowControl(self, **kwargs):  # noqa: N802
            wanted.searched = kwargs
            wanted.Exists = lambda **_: True
            return wanted

    backend = uia_backend(focused=explorer_ancestry(), app="Downloads")
    backend._auto = Module()
    found = backend.focused_window()
    assert found is wanted
    assert found.searched["SubName"] == "Downloads"


def test_an_unmatched_app_name_is_refused_rather_than_ignored():
    from victor.desktop.uia import PerceptionUnavailable

    class Module(FakeUIAModule):
        def WindowControl(self, **kwargs):  # noqa: N802
            missing = FakeUIAControl("WindowControl", "")
            missing.Exists = lambda **_: False
            return missing

        PaneControl = WindowControl

    backend = uia_backend(app="Nope")
    backend._auto = Module()
    with pytest.raises(PerceptionUnavailable, match="no top-level window"):
        backend.focused_window()


def test_window_info_reports_a_process_name_not_a_pid(monkeypatch):
    """A bare PID tells nobody anything, and macOS reports a name here."""
    from victor.desktop import uia as uia_module

    monkeypatch.setattr(uia_module, "_process_name", lambda pid: "explorer.exe")
    backend = uia_backend()
    title, process, _ = backend.window_info(
        FakeUIAControl("WindowControl", "Downloads - File Explorer")
    )
    assert title == "Downloads - File Explorer"
    assert process == "explorer.exe"


def test_window_info_falls_back_to_the_pid_when_the_name_is_unavailable(monkeypatch):
    from victor.desktop import uia as uia_module

    monkeypatch.setattr(uia_module, "_process_name", lambda pid: "")
    _, process, _ = backend_info()
    assert process == "4321"


def backend_info():
    return uia_backend().window_info(FakeUIAControl("WindowControl", "W"))


# --- geometry -------------------------------------------------------------


def test_rect_centre_is_where_a_click_would_land() -> None:
    assert Rect(24, 180, 140, 220).centre == (82, 200)


def test_an_empty_rect_is_not_actionable() -> None:
    element = Element(0, "Button", "Ghost", Rect(0, 0, 0, 0))
    assert not element.actionable


def test_a_disabled_element_is_not_actionable() -> None:
    element = Element(0, "Button", "Archive", Rect(0, 0, 10, 10), enabled=False)
    assert not element.actionable


# --- filtering ------------------------------------------------------------


def test_interactable_controls_are_kept() -> None:
    assert is_interesting("Button", "OK", Rect(0, 0, 10, 10))
    assert is_interesting("Edit", "", Rect(0, 0, 10, 10))


def test_unnamed_scaffolding_is_dropped() -> None:
    """A window has hundreds of anonymous panes; listing them buries the six
    things you can actually act on."""
    assert not is_interesting("Pane", "", Rect(0, 0, 100, 100))
    assert not is_interesting("Group", "   ", Rect(0, 0, 100, 100))


def test_a_labelled_container_is_kept() -> None:
    assert is_interesting("Group", "Recipients", Rect(0, 0, 100, 100))


def test_zero_sized_elements_are_dropped() -> None:
    assert not is_interesting("Button", "Hidden", Rect(5, 5, 5, 5))


# --- the walk -------------------------------------------------------------


def reader(root: FakeNode, **kwargs) -> TreeReader:
    return TreeReader(FakeBackend(root), **kwargs)


def test_the_demo_tree_reads_the_readmes_example() -> None:
    snapshot = reader(demo_tree()).snapshot()

    labels = [e.label for e in snapshot]
    assert "Compose" in labels
    assert "Search mail" in labels
    assert "Settings" in labels
    # The anonymous zero-sized pane is filtered out.
    assert all(e.rect.area > 0 for e in snapshot)


def test_elements_are_numbered_from_zero_without_gaps() -> None:
    snapshot = reader(demo_tree()).snapshot()
    assert [e.index for e in snapshot] == list(range(len(snapshot)))


def test_the_window_itself_is_not_listed() -> None:
    """Index 0 should be something you can act on, not the frame around it."""
    snapshot = reader(demo_tree()).snapshot()
    assert snapshot.window_title == "Fake Window"
    assert all(e.depth > 0 for e in snapshot)


def test_rendering_is_compact_enough_to_send() -> None:
    snapshot = reader(demo_tree()).snapshot()
    for line in snapshot.render().splitlines()[2:]:
        assert len(line) < 120


def test_disabled_controls_are_listed_but_flagged() -> None:
    snapshot = reader(demo_tree()).snapshot()
    archive = snapshot.find("Archive")[0]

    assert not archive.enabled
    assert "disabled" in archive.render()


def test_find_matches_case_insensitively() -> None:
    snapshot = reader(demo_tree()).snapshot()
    assert snapshot.find("compose")
    assert snapshot.find("compose", control_type="Button")
    assert not snapshot.find("compose", control_type="Edit")


def test_depth_is_bounded() -> None:
    """A web page in Edge is thousands of nodes deep; the walk must not follow."""
    deep = FakeNode("Window", "Deep", Rect(0, 0, 100, 100))
    node = deep
    for i in range(50):
        child = FakeNode("Button", f"level{i}", Rect(0, i, 10, i + 5))
        node.children.append(child)
        node = child

    snapshot = reader(deep, limits=WalkLimits(max_depth=5)).snapshot()
    assert snapshot.truncated is False  # depth-limited, not element-limited
    assert len(snapshot) <= 5


def test_element_count_is_bounded_and_reported() -> None:
    wide = FakeNode(
        "Window",
        "Wide",
        Rect(0, 0, 100, 100),
        children=[FakeNode("Button", f"b{i}", Rect(0, i, 10, i + 5)) for i in range(500)],
    )
    snapshot = reader(wide, limits=WalkLimits(max_elements=20)).snapshot()

    assert len(snapshot) == 20
    assert snapshot.truncated
    assert "some elements are missing" in snapshot.render()


def test_the_walk_is_breadth_first() -> None:
    """If the walk is cut short, the controls a user would reach for are the
    ones already collected."""
    root = FakeNode(
        "Window",
        "W",
        Rect(0, 0, 100, 100),
        children=[
            FakeNode(
                "Group",
                "Outer",
                Rect(0, 0, 50, 50),
                children=[FakeNode("Button", "Nested", Rect(0, 0, 10, 10))],
            ),
            FakeNode("Button", "TopLevel", Rect(50, 0, 60, 10)),
        ],
    )
    labels = [e.label for e in reader(root).snapshot()]
    assert labels.index("TopLevel") < labels.index("Nested")


def test_the_snapshot_reports_how_long_it_took() -> None:
    snapshot = reader(demo_tree()).snapshot()
    assert snapshot.duration_ms >= 0
    assert snapshot.backend == "fake"


# --- caching --------------------------------------------------------------


class CountingBackend(FakeBackend):
    def __init__(self, root: FakeNode) -> None:
        super().__init__(root)
        self.walks = 0

    def window_info(self, window):
        self.walks += 1
        return super().window_info(window)


def test_the_tree_is_cached_between_steps() -> None:
    """Re-walking every step would dominate the loop for no benefit."""
    backend = CountingBackend(demo_tree())
    tree = TreeReader(backend, cache_ttl=60)

    tree.snapshot()
    tree.snapshot()
    assert backend.walks == 1


def test_refresh_forces_a_rewalk() -> None:
    backend = CountingBackend(demo_tree())
    tree = TreeReader(backend, cache_ttl=60)

    tree.snapshot()
    tree.snapshot(refresh=True)
    assert backend.walks == 2


def test_invalidate_drops_the_cache() -> None:
    backend = CountingBackend(demo_tree())
    tree = TreeReader(backend, cache_ttl=60)

    tree.snapshot()
    tree.invalidate()
    tree.snapshot()
    assert backend.walks == 2


# --- capture --------------------------------------------------------------


def make_image(colour: tuple[int, int, int], size: tuple[int, int] = (1920, 1080)):
    pytest.importorskip("PIL")
    from PIL import Image

    return Image.new("RGB", size, colour)


def test_capture_downscales_to_the_token_budget() -> None:
    from victor.desktop import ScreenCapture

    capture = ScreenCapture(grabber=lambda region: make_image((10, 20, 30)))
    shot, is_new = capture.capture()

    assert is_new
    assert max(shot.width, shot.height) == 768


def test_an_unchanged_screen_is_served_from_cache() -> None:
    """The plan: an unchanged screen never re-bills a vision call."""
    from victor.desktop import ScreenCapture

    capture = ScreenCapture(grabber=lambda region: make_image((10, 20, 30)))
    first, new_first = capture.capture()
    second, new_second = capture.capture()

    assert new_first and not new_second
    assert second is first
    assert capture.hits == 1


def test_a_changed_screen_is_captured_again() -> None:
    from victor.desktop import ScreenCapture

    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    state = {"n": 0}

    def grabber(region):
        image = Image.new("RGB", (800, 600), (250, 250, 250))
        draw = ImageDraw.Draw(image)
        # A dialog appearing is exactly the change worth paying to look at.
        if state["n"]:
            draw.rectangle([100, 100, 700, 500], fill=(10, 10, 10))
        state["n"] += 1
        return image

    capture = ScreenCapture(grabber=grabber)
    capture.capture()
    _, is_new = capture.capture()
    assert is_new


def test_a_region_is_left_top_width_height() -> None:
    """The two ends of this disagreed. `find_on_screen` passed the window's
    (left, top, width, height) and the grabber unpacked it as
    (left, top, right, bottom), so every windowed capture was the wrong size -
    and off-screen entirely for any window not at the origin. Nothing caught it
    because no test captured a region."""
    from victor.desktop import ScreenCapture

    seen: list = []

    def grabber(region):
        seen.append(region)
        return make_image((7, 7, 7))

    ScreenCapture(grabber=grabber).capture((300, 100, 800, 600))
    assert seen == [(300, 100, 800, 600)]


def test_mss_is_asked_for_the_width_it_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same bug, one layer down: the monitor dict must describe the
    rectangle that was asked for, not one derived by subtracting from it."""
    pytest.importorskip("PIL")
    from victor.desktop import capture as capture_module

    asked: dict = {}

    class FakeShot:
        size = (800, 600)
        bgra = bytes(800 * 600 * 4)

    class FakeMss:
        monitors = [{"left": 0, "top": 0, "width": 1920, "height": 1080}]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grab(self, monitor):
            asked.update(monitor)
            return FakeShot()

    monkeypatch.setitem(
        __import__("sys").modules, "mss", type("m", (), {"mss": lambda: FakeMss()})
    )
    capture_module._grab_mss((300, 100, 800, 600))

    assert asked == {"left": 300, "top": 100, "width": 800, "height": 600}


def test_a_uniform_image_is_recognised_as_blank() -> None:
    """macOS does not refuse a capture without screen recording permission - it
    returns a valid image of nothing. Sent on, that spends one of ~250 daily
    vision requests asking which button is on a black rectangle."""
    pytest.importorskip("PIL")
    from victor.desktop.capture import _is_blank

    assert _is_blank(make_image((0, 0, 0)))
    assert _is_blank(make_image((255, 255, 255)))

    from PIL import Image, ImageDraw

    real = Image.new("RGB", (100, 100), (250, 250, 250))
    ImageDraw.Draw(real).rectangle([10, 10, 50, 50], fill=(10, 10, 10))
    assert not _is_blank(real)


def test_availability_tries_a_capture_rather_than_an_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It reported "screen capture ready" on a machine where every backend
    failed, because it only asked whether mss could be imported."""
    from victor.desktop import ScreenCapture
    from victor.desktop import capture as capture_module

    def explode(region):
        raise capture_module.CaptureUnavailable("the screen came back blank")

    monkeypatch.setattr(capture_module, "_quartz", lambda: None)
    monkeypatch.setattr(capture_module, "_grab_mss", explode)

    ok, detail = ScreenCapture.available()
    assert not ok
    assert "blank" in detail


def test_force_bypasses_the_cache() -> None:
    from victor.desktop import ScreenCapture

    capture = ScreenCapture(grabber=lambda region: make_image((5, 5, 5)))
    capture.capture()
    _, is_new = capture.capture(force=True)
    assert is_new


def test_identical_images_hash_identically() -> None:
    from victor.desktop import perceptual_hash

    assert perceptual_hash(make_image((1, 2, 3))) == perceptual_hash(make_image((1, 2, 3)))


def test_hamming_counts_differing_bits() -> None:
    assert hamming("00", "00") == 0
    assert hamming("00", "0f") == 4


# --- vision ---------------------------------------------------------------


def vision_client(settings: Settings, tmp_path, handler) -> tuple[VisionClient, QuotaLedger]:
    ledger = QuotaLedger(tmp_path / "quota.json")
    router = Router(settings, ledger)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VisionClient(settings, router, client=client), ledger


def a_shot():
    from victor.desktop import Screenshot

    return Screenshot(data=b"\xff\xd8fake", width=768, height=432, fingerprint="ab" * 8)


def a_snapshot() -> Snapshot:
    return Snapshot(
        window_title="Mail",
        process="mail",
        rect=Rect(0, 0, 1440, 900),
        elements=(
            Element(0, "Button", "Compose", Rect(24, 180, 140, 220)),
            Element(1, "Edit", "Search mail", Rect(300, 60, 900, 100)),
        ),
    )


def test_gemini_is_called_with_inline_data(settings: Settings, tmp_path) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = httpx.Request("POST", request.url, content=request.content).content
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "0"}]}}]}
        )

    client, _ = vision_client(settings, tmp_path, handler)
    answer = client.locate("open the compose window", a_shot(), a_snapshot())

    assert GEMINI_25_FLASH.model in seen["url"]
    assert b"inline_data" in seen["body"]
    assert answer.index == 0
    assert answer.element is not None and answer.element.label == "Compose"


def test_groq_vision_uses_the_openai_image_shape(tmp_path) -> None:
    """The fallback crosses providers, so both request formats must work."""
    settings = Settings(_env_file=None, GROQ_API_KEY="g", VICTOR_DATA_DIR=str(tmp_path))
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "1"}}]},
        )

    client, _ = vision_client(settings, tmp_path, handler)
    answer = client.locate("focus the search box", a_shot(), a_snapshot())

    assert b"image_url" in seen["body"]
    assert LLAMA_4_SCOUT.model.encode() in seen["body"]
    assert answer.index == 1


def test_a_spent_budget_degrades_rather_than_crashes(settings: Settings, tmp_path) -> None:
    """Running out of vision must leave a working agent."""
    client, ledger = vision_client(
        settings, tmp_path, lambda r: httpx.Response(200, json={})
    )
    for spec in (GEMINI_25_FLASH, LLAMA_4_SCOUT):
        for _ in range(spec.limits.requests_per_day or 0):
            ledger.record(spec.key, spec.limits)
            ledger._clock = lambda: __import__("time").time() + 10_000

    with pytest.raises(VisionUnavailable, match="no vision quota left today"):
        client.locate("anything", a_shot(), a_snapshot())


def test_a_retired_model_falls_through_to_the_next_one(
    settings: Settings, tmp_path
) -> None:
    """Observed live: Google retired the pinned Gemini model for new accounts,
    every call 404'd with "no longer available to new users", and vision failed
    outright - while Groq's vision model sat behind it working fine and was
    never asked. The router fell through on *ledger* exhaustion only."""
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        if "generativelanguage" in str(request.url):
            return httpx.Response(
                404, json={"error": {"message": "no longer available to new users"}}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "1"}}]}
        )

    client, _ = vision_client(settings, tmp_path, handler)
    answer = client.locate("the search box", a_shot(), a_snapshot())

    assert answer.index == 1
    assert len(asked) == 2, "the fallback was never asked"
    assert LLAMA_4_SCOUT.model in answer.model


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (429, {}),  # rate limited right now
        (401, {}),  # this provider's key is bad; the other one's may not be
        (404, {"error": {"message": "gone"}}),  # model retired
        (503, {}),  # provider having a bad day
    ],
)
def test_any_provider_side_refusal_tries_the_next_model(
    settings: Settings, tmp_path, status: int, body: dict
) -> None:
    """None of these are reasons to stop looking at the screen - they are
    reasons to ask someone else."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "generativelanguage" in str(request.url):
            return httpx.Response(status, json=body)
        return httpx.Response(200, json={"choices": [{"message": {"content": "1"}}]})

    client, _ = vision_client(settings, tmp_path, handler)
    assert client.locate("the search box", a_shot(), a_snapshot()).index == 1


def test_every_model_refusing_says_which_and_why(settings: Settings, tmp_path) -> None:
    """When the whole chain is out, the reason has to name the models - that is
    the difference between "vision is broken" and a diagnosis."""
    client, _ = vision_client(
        settings, tmp_path, lambda r: httpx.Response(404, json={"error": {"message": "gone"}})
    )

    with pytest.raises(VisionUnavailable) as exc:
        client.locate("anything", a_shot(), a_snapshot())

    assert "every vision model refused" in str(exc.value)
    assert GEMINI_25_FLASH.model in str(exc.value)
    assert LLAMA_4_SCOUT.model in str(exc.value)


def test_the_pinned_gemini_model_is_one_a_new_key_can_use(settings: Settings) -> None:
    """A pinned version rots silently and takes the primary vision model with
    it. `gemini-2.5-flash` was retired for new accounts while still being
    listed by the models endpoint, so nothing caught it until a live call."""
    assert GEMINI_25_FLASH.model == "gemini-flash-latest"


def test_none_means_no_element(settings: Settings, tmp_path) -> None:
    client, _ = vision_client(
        settings,
        tmp_path,
        lambda r: httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "NONE"}]}}]}
        ),
    )
    answer = client.locate("the eject button", a_shot(), a_snapshot())

    assert not answer.found
    assert answer.element is None


def test_an_out_of_range_index_is_rejected(settings: Settings, tmp_path) -> None:
    """A model naming an element that does not exist must not reach actuation."""
    client, _ = vision_client(
        settings,
        tmp_path,
        lambda r: httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "97"}]}}]}
        ),
    )
    answer = client.locate("something", a_shot(), a_snapshot())

    assert answer.index is None
    assert "97" in answer.raw


def test_the_prompt_carries_the_numbered_elements(settings: Settings, tmp_path) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8", "replace")
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "0"}]}}]}
        )

    client, _ = vision_client(settings, tmp_path, handler)
    client.locate("compose a message", a_shot(), a_snapshot())

    assert "[0]" in seen["body"]
    assert "Compose" in seen["body"]
    assert "number alone" in seen["body"]


def test_a_vision_call_is_charged(settings: Settings, tmp_path) -> None:
    client, ledger = vision_client(
        settings,
        tmp_path,
        lambda r: httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "0"}]}}]}
        ),
    )
    client.locate("compose", a_shot(), a_snapshot())

    assert ledger.usage(GEMINI_25_FLASH.key)[0] == 1


# --- cross-platform backend selection --------------------------------------


def test_the_backend_matches_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows and macOS each get a real backend; anything else fails clearly."""
    from victor.desktop import select_backend

    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert select_backend().name == "uia"

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert select_backend().name == "ax"

    monkeypatch.setattr("platform.system", lambda: "Linux")
    backend = select_backend()
    assert backend.name == "unsupported"
    ok, detail = backend.available()
    assert not ok
    assert "Linux" in detail


def test_an_unsupported_platform_fails_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rest of Victor works without perception - say so, do not crash."""
    from victor.desktop import PerceptionUnavailable, UnsupportedBackend

    backend = UnsupportedBackend("Plan9")
    with pytest.raises(PerceptionUnavailable, match="Plan9"):
        backend.focused_window()


def test_ax_roles_map_onto_the_shared_vocabulary() -> None:
    """One control-type vocabulary means the prompt and filter are OS-agnostic."""
    from victor.desktop.ax import _map_role

    assert _map_role("AXButton") == "Button"
    assert _map_role("AXTextField") == "Edit"
    assert _map_role("AXStaticText") == "Text"
    assert _map_role("AXRow") == "ListItem"
    # An unmapped role degrades to a readable name rather than being dropped.
    assert _map_role("AXSomethingNew") == "SomethingNew"


def test_window_buttons_get_their_canonical_names() -> None:
    from victor.desktop.ax import SUBROLE_NAMES

    assert SUBROLE_NAMES["AXCloseButton"] == "Close"
    assert SUBROLE_NAMES["AXMinimizeButton"] == "Minimise"


# --- deduplication ---------------------------------------------------------


def test_duplicate_controls_are_listed_once() -> None:
    """Chrome reports its bookmark bar under two parents; UIA does the same.

    Two identical rows with different indices waste context and give the model
    a choice with no right answer.
    """
    shared = Rect(0, 116, 1280, 150)
    root = FakeNode(
        "Window",
        "Dupes",
        Rect(0, 0, 1280, 800),
        children=[
            FakeNode(
                "Group",
                "Left",
                Rect(0, 0, 640, 800),
                children=[FakeNode("ToolBar", "Bookmarks", shared)],
            ),
            FakeNode(
                "Group",
                "Right",
                Rect(640, 0, 1280, 800),
                children=[FakeNode("ToolBar", "Bookmarks", shared)],
            ),
        ],
    )
    labels = [e.label for e in reader(root).snapshot()]
    assert labels.count("Bookmarks") == 1


def test_same_name_at_a_different_position_is_kept() -> None:
    """Two OK buttons in two dialogs are genuinely two things."""
    root = FakeNode(
        "Window",
        "Two dialogs",
        Rect(0, 0, 800, 600),
        children=[
            FakeNode("Button", "OK", Rect(10, 10, 60, 30)),
            FakeNode("Button", "OK", Rect(400, 10, 450, 30)),
        ],
    )
    assert len([e for e in reader(root).snapshot() if e.label == "OK"]) == 2
