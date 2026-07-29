"""Where memories live.

Two stores, one of which is authoritative. **SQLite holds everything** - the
text, the metadata and the vector itself. FAISS holds a copy of the vectors and
answers queries fast.

That split is deliberate. The classic failure of a vector store beside a
metadata store is drift: the index says hit number 41 and the sidecar no longer
agrees what 41 is, usually after a crash between two writes. Here the index is a
*cache* that can be thrown away and rebuilt from SQLite at any time, so drift is
not a corruption to recover from - it is a rebuild.

It costs one float array of duplication per record, about 1.5 KB, and buys a
store that cannot lie to you.

``IndexFlatIP`` is itself exact brute force - FAISS just does it in C rather
than Python. So the fallback used when FAISS is not installed is not an
approximation of the real thing; it returns the same neighbours in the same
order, more slowly. On the scale this store operates at - thousands of records,
not millions - the difference is milliseconds.
"""

from __future__ import annotations

import array
import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path
from typing import Any

from ..errors import VictorError

SCHEMA_VERSION = 1


class MemoryError_(VictorError):
    """The memory store cannot be used as asked."""

    exit_code = 6


class EmbedderChanged(MemoryError_):
    """The store was built with a different embedder than the one in use.

    Worth its own class because the failure it prevents is silent: vectors from
    two different models are not comparable, and searching one with the other
    returns confident nonsense rather than an error.
    """


@dataclass(frozen=True, slots=True)
class Record:
    """One remembered thing."""

    id: int
    kind: str
    text: str
    source: str = ""
    created: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        collapsed = " ".join(self.text.split())
        return collapsed if len(collapsed) <= 100 else collapsed[:97] + "..."


@dataclass(frozen=True, slots=True)
class Hit:
    """A record and how close it was. ``score`` is cosine similarity."""

    record: Record
    score: float

    def __str__(self) -> str:
        return f"[{self.score:.2f}] {self.record.summary}"


def fingerprint(text: str, kind: str) -> str:
    """Content hash, so remembering the same thing twice stores it once.

    Whitespace-insensitive: the same traceback captured from two runs differs in
    indentation often enough that treating those as distinct would fill the
    store with copies of one memory.
    """
    normalised = " ".join(text.split()).lower()
    return blake2b(f"{kind}\x00{normalised}".encode(), digest_size=16).hexdigest()


def _pack(vector: Sequence[float]) -> bytes:
    return array.array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array.array("f")
    values.frombytes(blob)
    return list(values)


class VectorStore:
    """Records, their vectors, and nearest-neighbour search over them."""

    def __init__(self, directory: Path, *, embedder_name: str, dimensions: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "memory.sqlite3"
        self.index_path = self.directory / "vectors.faiss"
        self.embedder_name = embedder_name
        self.dimensions = dimensions

        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._index: Any = None
        self._ids: list[int] = []
        self._vectors: list[list[float]] = []
        self._create_schema()
        self._check_embedder()
        self._load_vectors()

    # -- schema ------------------------------------------------------------

    def _create_schema(self) -> None:
        with self._lock, self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    created TEXT NOT NULL,
                    meta TEXT NOT NULL DEFAULT '{}',
                    vector BLOB NOT NULL
                )
                """
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS records_kind ON records(kind)")

    def _check_embedder(self) -> None:
        """Refuse to search vectors made by a different model."""
        with self._lock:
            rows = dict(self._db.execute("SELECT key, value FROM store_meta").fetchall())
        stored = rows.get("embedder")
        if stored is None:
            with self._lock, self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
                    [
                        ("embedder", self.embedder_name),
                        ("dimensions", str(self.dimensions)),
                        ("schema", str(SCHEMA_VERSION)),
                    ],
                )
            return
        if stored != self.embedder_name or int(rows.get("dimensions", 0)) != self.dimensions:
            raise EmbedderChanged(
                f"this memory was built with {stored!r} "
                f"({rows.get('dimensions')}d) and you are now using "
                f"{self.embedder_name!r} ({self.dimensions}d). Vectors from two "
                "models are not comparable. Run `victor index --rebuild` to "
                "re-encode what is stored - the text is all still here."
            )

    # -- vectors -----------------------------------------------------------

    def _load_vectors(self) -> None:
        with self._lock:
            rows = self._db.execute("SELECT id, vector FROM records ORDER BY id").fetchall()
            self._ids = [row["id"] for row in rows]
            self._vectors = [_unpack(row["vector"]) for row in rows]
            self._index = None
            # Built even when there is nothing to put in it: an empty
            # IndexFlatIP costs nothing, and it means `victor memory` reports
            # the backend it will actually use rather than "brute-force"
            # because the store happens to be new.
            self._build_index()

    def _build_index(self) -> Any:
        """Build the FAISS cache from what SQLite holds, if FAISS is here."""
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np
        except ImportError:
            self._index = None
            return None
        index = faiss.IndexFlatIP(self.dimensions)
        if self._vectors:
            index.add(np.asarray(self._vectors, dtype="float32"))
        self._index = index
        return index

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "brute-force"

    # -- writing -----------------------------------------------------------

    def add(
        self,
        text: str,
        vector: Sequence[float],
        *,
        kind: str = "note",
        source: str = "",
        meta: dict[str, Any] | None = None,
    ) -> Record | None:
        """Store one record. Returns ``None`` if it was already known."""
        if len(vector) != self.dimensions:
            raise MemoryError_(
                f"expected a {self.dimensions}d vector, got {len(vector)}d"
            )
        digest = fingerprint(text, kind)
        created = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock:
            cursor = self._db.execute(
                "SELECT id FROM records WHERE fingerprint = ?", (digest,)
            )
            if cursor.fetchone() is not None:
                return None
            with self._db:
                cursor = self._db.execute(
                    "INSERT INTO records (fingerprint, kind, text, source, created, meta, "
                    "vector) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        digest,
                        kind,
                        text,
                        source,
                        created,
                        json.dumps(meta or {}),
                        _pack(vector),
                    ),
                )
            record_id = int(cursor.lastrowid or 0)
            self._ids.append(record_id)
            self._vectors.append(list(vector))
            if self._index is not None:
                import numpy as np

                self._index.add(np.asarray([list(vector)], dtype="float32"))

        return Record(
            id=record_id,
            kind=kind,
            text=text,
            source=source,
            created=created,
            meta=meta or {},
        )

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM records")
        self._ids.clear()
        self._vectors.clear()
        self._index = None
        self.index_path.unlink(missing_ok=True)

    def replace_vectors(
        self, embedder_name: str, dimensions: int, vectors: list[list[float]]
    ) -> None:
        """Re-encode in place, for ``victor index --rebuild``.

        Possible only because the text never left SQLite: switching embedder is
        a re-encode rather than a re-crawl of the original files, which may have
        moved or changed since.
        """
        with self._lock:
            rows = self._db.execute("SELECT id FROM records ORDER BY id").fetchall()
            if len(rows) != len(vectors):
                raise MemoryError_(
                    f"{len(rows)} records but {len(vectors)} vectors - refusing to rebuild"
                )
            with self._db:
                for row, vector in zip(rows, vectors, strict=True):
                    self._db.execute(
                        "UPDATE records SET vector = ? WHERE id = ?",
                        (_pack(vector), row["id"]),
                    )
                self._db.executemany(
                    "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
                    [("embedder", embedder_name), ("dimensions", str(dimensions))],
                )
        self.embedder_name = embedder_name
        self.dimensions = dimensions
        self._load_vectors()

    # -- reading -----------------------------------------------------------

    def search(self, vector: Sequence[float], k: int = 5, *, kind: str | None = None) -> list[Hit]:
        """The ``k`` nearest records, most similar first."""
        if not self._vectors or k <= 0:
            return []
        query = list(vector)
        if len(query) != self.dimensions:
            raise MemoryError_(f"expected a {self.dimensions}d query, got {len(query)}d")

        # Over-fetch when filtering by kind: the index does not know about
        # kinds, so asking it for exactly k and then discarding some would
        # quietly return fewer than asked for.
        want = k if kind is None else min(len(self._vectors), k * 5)
        scored = self._nearest(query, want)

        hits: list[Hit] = []
        for position, score in scored:
            record = self.get(self._ids[position])
            if record is None or (kind is not None and record.kind != kind):
                continue
            hits.append(Hit(record=record, score=score))
            if len(hits) == k:
                break
        return hits

    def _nearest(self, query: list[float], k: int) -> list[tuple[int, float]]:
        if self._index is not None:
            import numpy as np

            scores, positions = self._index.search(
                np.asarray([query], dtype="float32"), min(k, len(self._vectors))
            )
            return [
                (int(pos), float(score))
                for pos, score in zip(positions[0], scores[0], strict=True)
                if pos >= 0
            ]

        # Same computation, in Python. Vectors are unit length, so the inner
        # product is the cosine similarity.
        scored = [
            (position, sum(a * b for a, b in zip(query, vector, strict=True)))
            for position, vector in enumerate(self._vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def get(self, record_id: int) -> Record | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM records WHERE id = ?", (record_id,)
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def texts(self) -> list[str]:
        """Every stored text, in id order. Used to re-encode on rebuild."""
        with self._lock:
            rows = self._db.execute("SELECT text FROM records ORDER BY id").fetchall()
        return [row["text"] for row in rows]

    def recent(self, limit: int = 20, *, kind: str | None = None) -> list[Record]:
        query = "SELECT * FROM records"
        params: list[Any] = []
        if kind is not None:
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT kind, COUNT(*) AS n FROM records GROUP BY kind"
            ).fetchall()
        return {row["kind"]: row["n"] for row in rows}

    def __len__(self) -> int:
        return len(self._ids)

    def size_bytes(self) -> int:
        return self.db_path.stat().st_size if self.db_path.exists() else 0

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _row_to_record(row: sqlite3.Row) -> Record:
    try:
        meta = json.loads(row["meta"])
    except (json.JSONDecodeError, TypeError):
        meta = {}
    return Record(
        id=row["id"],
        kind=row["kind"],
        text=row["text"],
        source=row["source"],
        created=row["created"],
        meta=meta if isinstance(meta, dict) else {},
    )


def add_many(
    store: VectorStore,
    embedder: Any,
    items: Iterable[tuple[str, str, str, dict[str, Any]]],
) -> int:
    """Embed and store ``(text, kind, source, meta)`` tuples. Returns new count."""
    batch = list(items)
    if not batch:
        return 0
    vectors = embedder.encode([text for text, _, _, _ in batch])
    stored = 0
    for (text, kind, source, meta), vector in zip(batch, vectors, strict=True):
        if store.add(text, vector, kind=kind, source=source, meta=meta) is not None:
            stored += 1
    return stored
