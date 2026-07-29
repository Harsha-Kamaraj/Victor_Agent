"""Turning text into vectors, locally.

The plan specifies ``fastembed`` with ``BAAI/bge-small-en-v1.5`` - a ~130 MB
ONNX model that runs on CPU. That is the right default: it is genuinely
semantic, it costs nothing per call, and it never leaves the machine, which is
what lets P6 claim "recalls the prior fix offline, with zero API calls".

But a 130 MB download is a hard dependency for a feature whose whole point is
that it works when nothing else does, and it makes the test suite depend on a
model file. So there are two embedders behind one protocol, and the difference
between them is stated rather than hidden:

:class:`FastEmbedEmbedder` understands meaning. "connection refused" and "could
not connect to the server" land near each other.

:class:`HashEmbedder` does not. It is a hashed bag of words - it matches text
that repeats, and near-misses that share vocabulary, and nothing else. That is
weaker than it sounds and also *exactly* the P6 exit gate: the same traceback
twice. It is the fallback rather than the default because "I remember this
error" is worth much more when it also fires on the paraphrase.

Which one is in use is reported by ``victor memory``, never guessed at.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..errors import VictorError

MODEL_NAME = "BAAI/bge-small-en-v1.5"
HASH_DIMENSIONS = 512
"""Enough to keep unrelated tracebacks apart without making the store huge.

At 512 floats a record costs ~2 KB of vector. A thousand remembered errors is
2 MB, which is the right order for something that lives in ``~/.victor``."""


class EmbeddingUnavailable(VictorError):
    """No embedder can be constructed on this machine."""

    exit_code = 6


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit vectors. One implementation per backend."""

    name: str
    dimensions: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def normalise(vector: list[float]) -> list[float]:
    """Scale to unit length, so a dot product *is* cosine similarity.

    Both the FAISS index and the fallback use inner product, which only equals
    cosine on normalised vectors. Doing it here rather than at search time means
    a vector is normalised once, when it is created, instead of on every query.
    """
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return vector
    return [value / magnitude for value in vector]


class FastEmbedEmbedder:
    """``BAAI/bge-small-en-v1.5`` via fastembed. Semantic, local, free."""

    name = "fastembed"
    dimensions = 384  # bge-small's output width

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingUnavailable(
                f"fastembed is not installed ({exc}). pip install -e '.[memory]'"
            ) from exc
        kwargs: dict[str, Any] = {"model_name": MODEL_NAME}
        if self.cache_dir is not None:
            # Keep the ONNX file inside Victor's data directory rather than in
            # a hidden cache somewhere else, so `victor memory` can report its
            # size and deleting ~/.victor really does remove everything.
            kwargs["cache_dir"] = str(self.cache_dir)
        self._model = TextEmbedding(**kwargs)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        return [normalise([float(v) for v in vector]) for vector in model.embed(texts)]

    def warm(self) -> None:
        """Force the model download and load now, rather than mid-task."""
        self.encode(["warm"])


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+")


class HashEmbedder:
    """A hashed bag of words. Lexical, not semantic - and honest about it.

    Deterministic, instant, and needs nothing installed, so the whole store,
    recall and auto-capture stack is testable without a model file. It finds
    text that repeats. It will not find a paraphrase, and the code that decides
    whether a recall is worth showing knows that.
    """

    name = "hash"

    def __init__(self, dimensions: int = HASH_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _WORD.findall(text.lower())
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            # The sign bit spreads collisions in both directions, so two
            # unrelated words landing in one bucket tend to cancel rather than
            # reinforce - the standard hashing-trick correction.
            sign = 1.0 if digest[4] & 1 else -1.0
            # Sub-linear in count: a traceback that repeats one word forty times
            # should not be forty times more about that word.
            vector[bucket] += sign * (1.0 + math.log(count))
        return normalise(vector)


def select_embedder(cache_dir: Path | None = None) -> Embedder:
    """The best embedder available here, preferring meaning over vocabulary."""
    from importlib.util import find_spec

    if find_spec("fastembed") is not None:
        return FastEmbedEmbedder(cache_dir)
    return HashEmbedder()


def describe_embedder(embedder: Embedder) -> str:
    """One line for ``victor memory`` and ``victor doctor``."""
    if embedder.name == "fastembed":
        return f"{MODEL_NAME} ({embedder.dimensions}d, local ONNX) - semantic"
    return (
        f"hashed bag of words ({embedder.dimensions}d) - matches repeated text "
        "only. pip install -e '.[memory]' for semantic recall"
    )
