"""P6: embeddings, the store, auto-capture and recall.

Everything here runs with nothing installed. :class:`HashEmbedder` needs no
model file and the store falls back to a Python scan when FAISS is absent, so
the suite exercises the same code paths on a bare machine that it does on one
with the memory extra - and the two backends are checked against each other
where that matters.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from victor.rag import (
    ErrorFixWatcher,
    HashEmbedder,
    Memory,
    VectorStore,
    chunk_text,
    command_head,
    describe_call,
    fingerprint,
    index_path,
    is_diagnostic,
    iter_files,
    summarise_error,
)
from victor.rag.embed import normalise
from victor.rag.store import EmbedderChanged
from victor.rag.store import MemoryError_ as StoreError

TRACEBACK = """\
Traceback (most recent call last):
  File "app.py", line 3, in <module>
    import httpx
ModuleNotFoundError: No module named 'httpx'
"""


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[Memory]:
    """Closed on the way out, because most of this file uses this fixture.

    A `return` here left one SQLite connection open per test, which POSIX is
    happy to ignore and Windows is not - and it is the same handle whose absence
    made the P6 selftest gate die in cleanup there.
    """
    embedder = HashEmbedder()
    store = VectorStore(
        tmp_path / "memory", embedder_name=embedder.name, dimensions=embedder.dimensions
    )
    yield Memory(store, embedder)
    store.close()


# --- embedding -------------------------------------------------------------


def test_vectors_are_unit_length():
    """The index uses inner product, which only equals cosine on unit vectors."""
    vector = HashEmbedder().encode(["some error text"])[0]
    assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-6)


def test_normalising_a_zero_vector_does_not_divide_by_zero():
    assert normalise([0.0, 0.0]) == [0.0, 0.0]


def test_the_same_text_always_embeds_the_same_way():
    embedder = HashEmbedder()
    assert embedder.encode(["pytest failed"]) == embedder.encode(["pytest failed"])


def test_identical_text_scores_one_and_unrelated_text_does_not():
    embedder = HashEmbedder()
    a, b = embedder.encode([TRACEBACK, "the disk is full and nothing will write"])
    assert sum(x * y for x, y in zip(a, a, strict=True)) == pytest.approx(1.0, abs=1e-6)
    assert sum(x * y for x, y in zip(a, b, strict=True)) < 0.3


def test_empty_input_gives_no_vectors():
    assert HashEmbedder().encode([]) == []


# --- the store -------------------------------------------------------------


def test_a_record_survives_a_reopen(tmp_path: Path):
    """SQLite is authoritative; the index is a cache rebuilt from it."""
    embedder = HashEmbedder()
    kwargs = {"embedder_name": embedder.name, "dimensions": embedder.dimensions}
    directory = tmp_path / "memory"

    store = VectorStore(directory, **kwargs)
    store.add("connection refused", embedder.encode(["connection refused"])[0], kind="fix")
    store.close()

    with VectorStore(directory, **kwargs) as reopened:
        assert len(reopened) == 1
        hits = reopened.search(embedder.encode(["connection refused"])[0], k=1)
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_deleting_the_index_loses_nothing(tmp_path: Path):
    """Drift between a vector index and its sidecar is a rebuild, not a loss."""
    embedder = HashEmbedder()
    kwargs = {"embedder_name": embedder.name, "dimensions": embedder.dimensions}
    store = VectorStore(tmp_path / "memory", **kwargs)
    store.add("disk full", embedder.encode(["disk full"])[0], kind="fix")
    store.index_path.unlink(missing_ok=True)
    store.close()

    with VectorStore(tmp_path / "memory", **kwargs) as reopened:
        assert len(reopened) == 1


def test_remembering_the_same_thing_twice_stores_it_once(memory: Memory):
    assert memory.remember_fix(error=TRACEBACK, fix="pip install httpx") is not None
    assert memory.remember_fix(error=TRACEBACK, fix="pip install httpx") is None
    assert len(memory.store) == 1


def test_whitespace_does_not_make_a_new_memory():
    """The same traceback captured twice differs in indentation often enough."""
    assert fingerprint("a  b\nc", "fix") == fingerprint("a b c", "fix")


def test_a_wrong_sized_vector_is_refused(memory: Memory):
    with pytest.raises(StoreError):
        memory.store.add("x", [0.1, 0.2], kind="fix")


def test_switching_embedder_is_refused_rather_than_answered_wrongly(tmp_path: Path):
    """Vectors from two models are not comparable; searching returns nonsense."""
    embedder = HashEmbedder()
    VectorStore(
        tmp_path / "m", embedder_name=embedder.name, dimensions=embedder.dimensions
    ).close()
    with pytest.raises(EmbedderChanged, match="index --rebuild"):
        VectorStore(tmp_path / "m", embedder_name="fastembed", dimensions=384)


def test_rebuilding_re_encodes_without_re_reading_the_files(memory: Memory):
    """The text never left SQLite, so a rebuild needs no original source."""
    memory.remember_fix(error=TRACEBACK, fix="pip install httpx")
    wider = HashEmbedder(dimensions=256)
    assert memory.rebuild(wider) == 1
    assert memory.store.dimensions == 256
    assert memory.recall(TRACEBACK, kind="fix").found


def test_search_filters_by_kind(memory: Memory):
    memory.remember_fix(error=TRACEBACK, fix="pip install httpx")
    memory.remember_note("httpx is a http client", source="notes.md")
    assert memory.recall(TRACEBACK, kind="fix").best.record.kind == "fix"
    assert not memory.recall("a totally unrelated sentence", kind="fix").found


# --- recall ---------------------------------------------------------------


def test_an_empty_memory_recalls_nothing(memory: Memory):
    assert not memory.recall("anything at all").found


def test_recall_is_silent_below_the_floor(memory: Memory):
    """A store always returns its nearest neighbour; nearest is not relevant.

    An injected wrong memory is worse than none - it arrives as prior
    experience and the model treats it as evidence.
    """
    memory.remember_fix(error=TRACEBACK, fix="pip install httpx")
    assert not memory.recall("the printer is on fire and the cat is missing").found


def test_lowering_the_floor_shows_the_near_miss(memory: Memory):
    """What `victor recall --all` does, so a bad floor is diagnosable."""
    memory.remember_fix(error=TRACEBACK, fix="pip install httpx")
    assert memory.recall("printer on fire", threshold=0.0).found


def test_the_recalled_block_reads_as_a_report_not_an_order(memory: Memory):
    memory.remember_fix(error=TRACEBACK, fix="pip install httpx")
    block = memory.recall_for_error(TRACEBACK).for_model()
    assert "pip install httpx" in block
    assert "not an instruction" in block


def test_the_recalled_block_is_bounded(memory: Memory):
    """A remembered fix that crowds out the task is a loss, not a help."""
    memory.remember_fix(error=TRACEBACK, fix="x" * 5000)
    assert len(memory.recall_for_error(TRACEBACK).for_model()) <= 900


# --- what counts as a fix --------------------------------------------------


def test_a_fix_is_captured_when_the_command_recovers(memory: Memory):
    watcher = ErrorFixWatcher(memory)
    assert watcher.observe("pytest -x", ok=False, output=TRACEBACK) is None
    assert watcher.observe("pip install httpx", ok=True) is None
    note = watcher.observe("pytest -x", ok=True)

    assert note is not None and "pytest" in note
    assert memory.recall_for_error(TRACEBACK).best.record.meta["fix"] == "pip install httpx"


def test_looking_around_is_not_fixing(memory: Memory):
    """`pytest` fails, `ls` succeeds - "the fix is ls" would be recalled later."""
    watcher = ErrorFixWatcher(memory)
    watcher.observe("pytest", ok=False, output=TRACEBACK)
    for command in ("ls -la", "cat app.py", "grep httpx app.py", "pwd"):
        watcher.observe(command, ok=True)
    watcher.observe("pytest", ok=True)

    assert watcher.captured == 0
    assert len(memory.store) == 0


def test_an_unrelated_success_does_not_close_the_failure(memory: Memory):
    """Only the command that failed can prove it was fixed."""
    watcher = ErrorFixWatcher(memory)
    watcher.observe("pytest", ok=False, output=TRACEBACK)
    watcher.observe("npm install", ok=True)

    assert watcher.captured == 0
    assert watcher.outstanding == ["pytest"]


def test_a_command_that_never_recovers_stores_nothing(memory: Memory):
    watcher = ErrorFixWatcher(memory)
    watcher.observe("pytest", ok=False, output=TRACEBACK)
    watcher.observe("pip install httpx", ok=True)
    watcher.observe("pytest", ok=False, output=TRACEBACK)

    assert watcher.captured == 0


def test_flapping_without_an_intervention_is_not_a_fix(memory: Memory):
    """Failed, then passed, nothing in between - flaky, or a transient."""
    watcher = ErrorFixWatcher(memory)
    watcher.observe("pytest", ok=False, output=TRACEBACK)
    watcher.observe("pytest", ok=True)

    assert watcher.captured == 0


def test_interventions_are_capped(memory: Memory):
    watcher = ErrorFixWatcher(memory, max_interventions=2)
    watcher.observe("pytest", ok=False, output=TRACEBACK)
    for i in range(6):
        watcher.observe(f"pip install pkg{i}", ok=True)
    watcher.observe("pytest", ok=True)

    fix = memory.recall_for_error(TRACEBACK).best.record.meta["fix"]
    assert len(fix.splitlines()) == 2


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pytest -x tests/", "pytest"),
        ("/usr/bin/pytest", "pytest"),
        ("git push origin main", "git push"),
        ("git status", "git status"),
        ("npm install", "npm install"),
        ("pip install -U httpx", "pip install"),
        ("", ""),
    ],
)
def test_commands_are_identified_by_what_they_run(command, expected):
    """`pytest -x tests/` and `pytest tests/` are the same attempt."""
    assert command_head(command) == expected


@pytest.mark.parametrize("command", ["ls", "cat x", "grep -r foo .", "pwd", "which python"])
def test_diagnostics_are_recognised(command):
    assert is_diagnostic(command)


@pytest.mark.parametrize(
    "command",
    [
        "pip install httpx",
        "npm ci",
        "make build",
        # Starts with a read-only command name and writes a file anyway. The
        # first version of is_diagnostic kept its own list of names and called
        # this diagnostic, so the fix that created the missing module was
        # dropped and nothing was ever remembered.
        "printf 'def greet(): pass' > helper.py",
        "echo PATCHED > config.ini",
    ],
)
def test_changes_are_not_diagnostics(command):
    assert not is_diagnostic(command)


def test_diagnostic_agrees_with_the_safety_classifier():
    """One definition of "this only reads", not two that drift apart."""
    from victor.safety.classify import Risk, classify_shell

    for command in ("ls -la", "cat x > y", "rm -rf build", "git status", "sed -i s/a/b/ f"):
        assert is_diagnostic(command) is (classify_shell(command).risk is Risk.SAFE)


# --- identity for tools that are not the shell -----------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "head"),
    [
        ("shell", {"command": "pytest -x tests/"}, "pytest"),
        ("git", {"subcommand": "push", "args": ["origin", "main"]}, "git push"),
        ("click", {"index": 4, "label": "Save"}, "click Save"),
        # The index moves as the tree re-sorts; the label is what was meant.
        ("click", {"index": 9, "label": "Save"}, "click Save"),
        ("click", {"index": 4}, "click 4"),
        ("press_keys", {"keys": "mod+s"}, "press_keys mod+s"),
        ("open_app", {"name": "Notepad", "launch": True}, "open_app Notepad"),
        ("type_text", {"text": "hello", "label": "Search"}, "type_text Search"),
        ("type_text", {"text": "hello"}, "type_text"),
        ("screen_read", {"filter": "save", "limit": 20}, "screen_read"),
        ("read_file", {"path": "app.py", "start_line": 3}, "read_file app.py"),
    ],
)
def test_every_tool_call_gets_an_identity(tool, arguments, head):
    action = describe_call(tool, arguments)
    assert action is not None
    assert action.head == head


def test_two_different_clicks_are_two_different_attempts(memory: Memory):
    """The failure mode of a shared identity: clicking Cancel would "prove"
    that the failed click on Save had been fixed."""
    watcher = ErrorFixWatcher(memory)
    save = describe_call("click", {"index": 1, "label": "Save"}, mutating=True)
    cancel = describe_call("click", {"index": 2, "label": "Cancel"}, mutating=True)

    watcher.observe(save, ok=False, output="element 1 is not enabled")
    watcher.observe(cancel, ok=True)

    assert watcher.captured == 0
    assert watcher.outstanding == ["click Save"]


def test_a_desktop_fix_is_captured(memory: Memory):
    """The failure this whole change exists for: the click was aimed at the
    wrong window, focusing the right one fixed it, and until now nothing in
    Victor remembered that."""
    watcher = ErrorFixWatcher(memory)
    click = describe_call("click", {"index": 3, "label": "Save"}, mutating=True)
    focus = describe_call("open_app", {"name": "Notepad"}, mutating=True)

    assert watcher.observe(click, ok=False, output="no element at index 3") is None
    assert watcher.observe(focus, ok=True) is None
    note = watcher.observe(click, ok=True)

    assert note is not None and "click Save" in note
    best = memory.recall_for_error("no element at index 3").best
    assert best.record.meta["fix"] == "open_app Notepad"
    # Stored without a shell prompt: it was never a command, and a `$` in front
    # of it invites the model to try running it.
    assert not best.record.text.startswith("$")


def test_re_reading_the_screen_is_not_a_fix(memory: Memory):
    """screen_read and scroll are declared non-mutating, so they are the
    desktop's version of `ls` - looking, not fixing."""
    watcher = ErrorFixWatcher(memory)
    click = describe_call("click", {"index": 3, "label": "Save"}, mutating=True)

    watcher.observe(click, ok=False, output="stale index")
    watcher.observe(describe_call("screen_read", {}, mutating=False), ok=True)
    watcher.observe(describe_call("scroll", {"direction": "down"}, mutating=False), ok=True)
    watcher.observe(click, ok=True)

    assert watcher.captured == 0


def test_a_read_only_git_subcommand_is_not_an_intervention(memory: Memory):
    """`git push` fails, `git status` succeeds, `git push` works. Status did
    nothing - the tool's own MUTATING set is what says so."""
    watcher = ErrorFixWatcher(memory)
    push = describe_call("git", {"subcommand": "push", "args": ["origin", "main"]})
    status = describe_call("git", {"subcommand": "status"})

    watcher.observe(push, ok=False, output="rejected: fetch first")
    watcher.observe(status, ok=True)
    watcher.observe(push, ok=True)

    assert watcher.captured == 0


def test_a_git_fix_is_captured(memory: Memory):
    watcher = ErrorFixWatcher(memory)
    push = describe_call("git", {"subcommand": "push", "args": ["origin", "main"]})
    pull = describe_call("git", {"subcommand": "pull", "args": ["--rebase"]})

    watcher.observe(push, ok=False, output="rejected: fetch first")
    watcher.observe(pull, ok=True)
    note = watcher.observe(push, ok=True)

    assert note is not None
    assert memory.recall_for_error("rejected: fetch first").best.record.meta["fix"] == (
        "git pull --rebase"
    )


def test_an_unknown_tool_splits_rather_than_merges():
    """A tool added later has no entry in the table. Splitting too finely
    remembers nothing; merging too coarsely remembers the wrong fix."""
    one = describe_call("future_tool", {"target": "a"})
    two = describe_call("future_tool", {"target": "b"})

    assert one.head != two.head
    assert describe_call("future_tool", {"target": "a"}).head == one.head


# --- summarising and chunking ---------------------------------------------


def test_a_long_traceback_keeps_both_ends():
    """The exception identifies the problem; the head says what was attempted."""
    text = "start of the problem\n" + "\n".join(
        f'  File "/x/{i}.py", line {i}, in f' for i in range(400)
    ) + "\nValueError: the actual problem"
    summary = summarise_error(text, limit=500)

    assert "start of the problem" in summary
    assert "ValueError: the actual problem" in summary
    assert len(summary) <= 500


def test_a_short_error_is_kept_whole():
    assert summarise_error("ValueError: nope") == "ValueError: nope"


def test_chunks_do_not_cut_lines_in_half():
    text = "\n".join(f"line {i} with some content on it" for i in range(200))
    chunks = chunk_text(text, size=200, overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.splitlines():
            assert line == "" or line.startswith("line ")


def test_chunks_overlap_so_a_fact_on_a_boundary_survives():
    text = "\n".join(f"line {i}" for i in range(60))
    chunks = chunk_text(text, size=100, overlap=40)

    assert len(chunks) > 1
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        tail = earlier.splitlines()[-1]
        assert tail in later, f"{tail!r} fell into the gap between two chunks"


def test_chunking_empty_text_gives_nothing():
    assert chunk_text("") == []


# --- indexing files --------------------------------------------------------


def test_indexing_skips_the_directories_nobody_wants(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    for junk in ("node_modules", "__pycache__", ".git", ".venv"):
        (tmp_path / junk).mkdir()
        (tmp_path / junk / "x.py").write_text("noise", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    found = [p.name for p in iter_files(tmp_path)]
    assert found == ["main.py"]


def test_oversized_files_are_skipped(tmp_path: Path):
    (tmp_path / "big.py").write_text("x" * 5000, encoding="utf-8")
    (tmp_path / "small.py").write_text("ok", encoding="utf-8")
    assert [p.name for p in iter_files(tmp_path, max_bytes=1000)] == ["small.py"]


def test_indexing_a_project_stores_chunks(tmp_path: Path, memory: Memory):
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (project / "b.md").write_text("# Notes\nsome prose here\n", encoding="utf-8")

    files, stored = index_path(memory, project)
    assert files == 2
    assert stored == 2

    # Re-indexing an unchanged project stores nothing.
    assert index_path(memory, project) == (2, 0)


def test_an_indexed_file_can_be_recalled(tmp_path: Path, memory: Memory):
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.md").write_text(
        "the deployment key lives in the vault under prod-deploy", encoding="utf-8"
    )
    index_path(memory, project)

    found = memory.recall("deployment key vault prod-deploy", kind="file")
    assert found.found
    assert "prod-deploy" in found.best.record.text


# --- the agent's error path ------------------------------------------------


def _bare_agent(memory, settings, watcher=None, desktop=False):
    """An Agent with only the parts ``_remember`` touches.

    Built with ``__new__`` on purpose: the point is to exercise the memory hook
    against a store that misbehaves, without a model, a provider or a run.
    """
    from victor.agent.loop import Agent
    from victor.tools import build_registry
    from victor.tracing import Trace

    agent = Agent.__new__(Agent)
    agent.memory = memory
    agent.watcher = watcher
    agent.recalled = 0
    agent.messages = []
    agent.registry = build_registry(settings, desktop=desktop)
    agent.trace = Trace.disabled()
    return agent


def test_memory_never_breaks_a_run(settings):
    """A corrupt store must degrade the run to "no recall", not fail it."""
    from victor.agent.llm import ToolCall
    from victor.tools.base import ToolResult

    class Exploding:
        def recall_for_error(self, error):
            raise RuntimeError("store is corrupt")

    agent = _bare_agent(Exploding(), settings)
    call = ToolCall(id="1", name="shell", arguments={"command": "pytest"})
    agent._remember(call, ToolResult(ok=False, error="boom"))  # must not raise
    assert agent.recalled == 0


def test_a_failing_desktop_call_consults_memory(memory: Memory, settings):
    """The gap this closes. `_remember` returned early for anything that was
    not the shell, so a stored desktop fix could never be recalled - the agent
    made the same mistake on Tuesday that it had solved on Monday."""
    from victor.agent.llm import ToolCall
    from victor.tools.base import ToolResult

    memory.remember_fix(
        error="click Save\nno element at index 3",
        fix="open_app Notepad",
        command="click Save",
    )
    agent = _bare_agent(memory, settings, desktop=True)

    call = ToolCall(id="1", name="click", arguments={"index": 3, "label": "Save"})
    agent._remember(call, ToolResult(ok=False, error="no element at index 3"))

    assert agent.recalled == 1
    assert "open_app Notepad" in agent.messages[-1]["content"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"decision": "deny"},  # the safety layer refused it
        {"refused": "repeat"},  # the loop refused it as a repeat
        {"refused": "terminal"},  # a desktop tool refused to type into a shell
    ],
)
def test_a_call_that_never_ran_is_not_remembered(memory: Memory, settings, metadata):
    """A block is not a failure. Recorded as one, the next success looks like
    its fix, and the store fills with advice for problems nobody had."""
    from victor.agent.llm import ToolCall
    from victor.rag import ErrorFixWatcher
    from victor.tools.base import ToolResult

    watcher = ErrorFixWatcher(memory)
    agent = _bare_agent(memory, settings, watcher=watcher)

    call = ToolCall(id="1", name="shell", arguments={"command": "rm -rf /"})
    agent._remember(call, ToolResult(ok=False, error="blocked", metadata=metadata))

    assert watcher.outstanding == []
    assert agent.recalled == 0


# --- choosing an embedder without paying for one --------------------------
#
# `select_embedder` used to answer "fastembed, if it is importable", which is
# not the same question as "fastembed, if it is usable here". Pointed at a fresh
# cache it downloaded 130 MB - and `victor selftest` pointed it at a temporary
# directory, so the gate claiming "0 API calls" fetched a model on every run and
# the suite inherited that. A download has no timeout to fail over from, so a
# stalled one hung instead of falling back.


def test_a_bare_onnx_file_is_not_the_embedding_model(tmp_path: Path) -> None:
    """The piper voice is a bare ``.onnx`` in the same directory.

    A `models/**/*.onnx` glob matched it, so anyone who had run
    `victor voice install` - which the quickstart tells them to do first - was
    told nothing before a 130 MB download.
    """
    from victor.rag.embed import FastEmbedEmbedder

    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"not an embedder")
    assert not FastEmbedEmbedder(tmp_path).installed


def test_a_downloaded_model_is_recognised(tmp_path: Path) -> None:
    """fastembed names the directory after the quantised mirror it really
    fetches, not after ``BAAI/bge-small-en-v1.5``."""
    from victor.rag.embed import FastEmbedEmbedder

    snapshot = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model_optimized.onnx").write_bytes(b"weights")

    assert FastEmbedEmbedder(tmp_path).installed


def test_declining_a_download_falls_back_instead_of_fetching(tmp_path: Path) -> None:
    """``auto_download=False`` must never reach the network, whether or not
    fastembed is importable on this machine."""
    from victor.rag.embed import select_embedder

    assert select_embedder(tmp_path, auto_download=False).name == "hash"


def test_a_cached_model_is_still_preferred_when_downloads_are_declined(
    tmp_path: Path,
) -> None:
    """Declining the download must not mean declining the good embedder - the
    point is to skip the fetch, not to degrade a machine that already has it."""
    pytest.importorskip("fastembed")
    from victor.rag.embed import select_embedder

    snapshot = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model_optimized.onnx").write_bytes(b"weights")

    assert select_embedder(tmp_path, auto_download=False).name == "fastembed"


def test_a_store_that_refuses_to_open_does_not_hold_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``__init__`` that raises hands the caller no object to close.

    ``EmbedderChanged`` is the case that matters: its remedy is
    `victor index --rebuild`, which has to replace the very files the failed
    open was still holding. `build_agent` swallows this error and carries on, so
    the handle stayed open for the life of the process - invisible on POSIX,
    fatal to the remedy on Windows.

    Watching the connection close rather than trying to delete the file,
    because deleting an open file succeeds on this machine whatever the code
    does. `sqlite3.Connection` is immutable, so the spy arrives as the
    connection factory instead of a patched method.
    """
    import sqlite3

    embedder = HashEmbedder()
    VectorStore(
        tmp_path / "m", embedder_name=embedder.name, dimensions=embedder.dimensions
    ).close()

    closes: list[int] = []

    class Watched(sqlite3.Connection):
        def close(self) -> None:
            closes.append(id(self))
            super().close()

    real = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: real(*a, factory=Watched, **kw)
    )

    with pytest.raises(EmbedderChanged):
        VectorStore(tmp_path / "m", embedder_name="fastembed", dimensions=384)

    assert closes, "the refused store kept its SQLite connection open"
