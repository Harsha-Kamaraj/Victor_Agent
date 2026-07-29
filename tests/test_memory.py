"""P6: embeddings, the store, auto-capture and recall.

Everything here runs with nothing installed. :class:`HashEmbedder` needs no
model file and the store falls back to a Python scan when FAISS is absent, so
the suite exercises the same code paths on a bare machine that it does on one
with the memory extra - and the two backends are checked against each other
where that matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from victor.rag import (
    ErrorFixWatcher,
    HashEmbedder,
    Memory,
    VectorStore,
    chunk_text,
    command_head,
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
def memory(tmp_path: Path) -> Memory:
    embedder = HashEmbedder()
    store = VectorStore(
        tmp_path / "memory", embedder_name=embedder.name, dimensions=embedder.dimensions
    )
    return Memory(store, embedder)


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

    reopened = VectorStore(directory, **kwargs)
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

    reopened = VectorStore(tmp_path / "memory", **kwargs)
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


def test_memory_never_breaks_a_run(memory: Memory):
    """A corrupt store must degrade the run to "no recall", not fail it."""
    from victor.agent.loop import Agent
    from victor.tools.base import ToolResult

    class Exploding:
        def recall_for_error(self, error):
            raise RuntimeError("store is corrupt")

    agent = Agent.__new__(Agent)
    agent.memory = Exploding()
    agent.watcher = None
    agent.recalled = 0
    from victor.tracing import Trace

    agent.trace = Trace.disabled()

    from victor.agent.llm import ToolCall

    call = ToolCall(id="1", name="shell", arguments={"command": "pytest"})
    agent._remember(call, ToolResult(ok=False, error="boom"))  # must not raise
    assert agent.recalled == 0
