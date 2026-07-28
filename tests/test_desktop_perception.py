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
