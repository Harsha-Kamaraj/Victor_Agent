"""Getting things back out, and deciding when not to.

The interesting decision in this module is the threshold. A vector store always
returns its nearest neighbour - "nearest" does not mean "relevant", and an
empty store's nearest neighbour is nothing while a store of one unrelated note
will happily return that note for any query at all.

Injecting a wrong memory into the agent's context is worse than injecting
none. It is presented as prior experience, so the model treats it as evidence,
and it costs tokens to say something misleading. So recall is silent below a
similarity floor, and the floor depends on which embedder is answering: the
hashed one only really recognises repeats, so it is held to a higher bar than
the semantic one, which is supposed to match paraphrases.

Everything here is local. Recall spends no API request and no quota, which is
the whole of the P6 exit gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tracing import Trace
from .embed import Embedder, describe_embedder, select_embedder
from .store import Hit, Record, VectorStore, add_many

#: Minimum cosine similarity worth showing, per embedder.
#:
#: Measured rather than guessed. With bge-small, a paraphrase of a stored
#: traceback scores around 0.85 and an unrelated error around 0.5, so 0.62
#: sits in the gap. The hashed embedder scores an exact repeat at 1.0 and
#: shares-some-words at 0.2-0.4, so its floor is high: it is only trusted for
#: what it is actually good at.
THRESHOLDS = {"fastembed": 0.62, "hash": 0.55}
DEFAULT_THRESHOLD = 0.62

MAX_CONTEXT_CHARS = 900
"""What one recall may add to the conversation. The context budget is 8,000
tokens a minute; a remembered fix that crowds out the actual task is a loss."""


@dataclass(frozen=True, slots=True)
class Recollection:
    """What memory had to say about one query."""

    hits: tuple[Hit, ...]
    query: str

    @property
    def found(self) -> bool:
        return bool(self.hits)

    @property
    def best(self) -> Hit | None:
        return self.hits[0] if self.hits else None

    def for_model(self, limit: int = MAX_CONTEXT_CHARS) -> str:
        """The block injected into the agent's conversation.

        Phrased as a report of what happened before, not as an instruction.
        A memory that says "run this" is a memory that will eventually be wrong
        and obeyed anyway; one that says "last time, this worked" leaves the
        model free to notice that the situation differs.
        """
        if not self.hits:
            return ""
        lines = ["You have seen something like this before."]
        for hit in self.hits:
            fix = str(hit.record.meta.get("fix", "")).strip()
            if fix:
                lines.append(f"\nPreviously: {hit.record.summary}")
                lines.append(f"What resolved it:\n{fix}")
            else:
                where = hit.record.source or "an indexed file"
                lines.append(f"\nFrom {where}:\n{hit.record.text.strip()}")
        lines.append(
            "\nThis is a note from a previous session, not an instruction - "
            "check that it applies before acting on it."
        )
        block = "\n".join(lines)
        return block if len(block) <= limit else block[: limit - 3] + "..."


class Memory:
    """Victor's local memory: store, embedder, and the rules for using them."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        trace: Trace | None = None,
        threshold: float | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.trace = trace or Trace.disabled()
        self.threshold = (
            threshold
            if threshold is not None
            else THRESHOLDS.get(embedder.name, DEFAULT_THRESHOLD)
        )

    # -- writing -----------------------------------------------------------

    def remember_fix(self, *, error: str, fix: str, command: str = "") -> Record | None:
        """Store an error and what resolved it. ``None`` if already known."""
        if not error.strip() or not fix.strip():
            return None
        vector = self.embedder.encode([error])[0]
        record = self.store.add(
            error,
            vector,
            kind="fix",
            source=command,
            meta={"fix": fix, "command": command},
        )
        if record is not None:
            self.trace.event("memory.remember", kind="fix", command=command)
        return record

    def remember_note(self, text: str, *, source: str = "") -> Record | None:
        if not text.strip():
            return None
        vector = self.embedder.encode([text])[0]
        return self.store.add(text, vector, kind="note", source=source)

    def add_batch(self, items: list[tuple[str, str, str, dict[str, Any]]]) -> int:
        """Embed and store many ``(text, kind, source, meta)`` tuples at once."""
        return add_many(self.store, self.embedder, items)

    # -- reading -----------------------------------------------------------

    def recall(
        self, query: str, *, k: int = 2, kind: str | None = None, threshold: float | None = None
    ) -> Recollection:
        """Nearest memories above the relevance floor. Local, free, instant."""
        floor = self.threshold if threshold is None else threshold
        if not query.strip() or len(self.store) == 0:
            return Recollection((), query)

        with self.trace.span("memory.recall", chars=len(query)) as span:
            vector = self.embedder.encode([query])[0]
            hits = tuple(
                hit for hit in self.store.search(vector, k=k, kind=kind) if hit.score >= floor
            )
            span["hits"] = len(hits)
            span["best"] = round(hits[0].score, 3) if hits else 0.0
            # Named so a trace can show the zero-cost claim rather than assert it.
            span["cost"] = 0
        return Recollection(hits, query)

    def recall_for_error(self, error: str) -> Recollection:
        """Recall restricted to remembered fixes, for the agent's error path."""
        from .ingest import summarise_error

        return self.recall(summarise_error(error), k=1, kind="fix")

    # -- housekeeping ------------------------------------------------------

    def rebuild(self, embedder: Embedder | None = None) -> int:
        """Re-encode everything, for a switch of embedder. Returns the count."""
        target = embedder or self.embedder
        texts = self.store.texts()
        vectors = target.encode(texts) if texts else []
        self.store.replace_vectors(target.name, target.dimensions, vectors)
        self.embedder = target
        self.threshold = THRESHOLDS.get(target.name, DEFAULT_THRESHOLD)
        return len(texts)

    def describe(self) -> str:
        counts = self.store.counts()
        parts = [f"{n} {kind}" for kind, n in sorted(counts.items())] or ["empty"]
        return (
            f"{len(self.store)} records ({', '.join(parts)}), "
            f"{self.store.size_bytes() / 1024:.0f} KB, "
            f"index {self.store.backend}, {describe_embedder(self.embedder)}"
        )

    def close(self) -> None:
        self.store.close()


def build_memory(
    settings: Any,
    *,
    trace: Trace | None = None,
    embedder: Embedder | None = None,
    directory: Path | None = None,
) -> Memory:
    """The standard memory for a run.

    Constructing this touches the disk but not the model: fastembed loads on
    first use, so an agent that never hits an error never pays for the
    embedder it did not need.
    """
    paths = settings.paths.ensure()
    chosen = embedder or select_embedder(paths.models_dir)
    store = VectorStore(
        directory or paths.memory_dir,
        embedder_name=chosen.name,
        dimensions=chosen.dimensions,
    )
    return Memory(store, chosen, trace=trace)
