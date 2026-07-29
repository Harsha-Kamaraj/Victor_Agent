"""P6: local memory - embeddings, a vector store, and recall.

Victor's answer to repeating diagnostic work. When a shell command fails and is
later made to succeed, the pair is captured without anyone deciding to capture
it; when a similar failure appears again, the earlier fix is recalled and put in
front of the model.

All of it is local. Embedding is an ONNX model on the CPU, search is an exact
inner-product scan, and storage is SQLite - so recall spends no API request and
no quota, which is what P6's exit gate measures.
"""

from .embed import (
    MODEL_NAME,
    Embedder,
    EmbeddingUnavailable,
    FastEmbedEmbedder,
    HashEmbedder,
    describe_embedder,
    select_embedder,
)
from .ingest import (
    ErrorFixWatcher,
    chunk_text,
    command_head,
    index_path,
    is_diagnostic,
    iter_files,
    summarise_error,
)
from .recall import Memory, Recollection, build_memory
from .store import EmbedderChanged, Hit, Record, VectorStore, fingerprint

__all__ = [
    "MODEL_NAME",
    "Embedder",
    "EmbedderChanged",
    "EmbeddingUnavailable",
    "ErrorFixWatcher",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "Hit",
    "Memory",
    "Record",
    "Recollection",
    "VectorStore",
    "build_memory",
    "chunk_text",
    "command_head",
    "describe_embedder",
    "fingerprint",
    "index_path",
    "is_diagnostic",
    "iter_files",
    "select_embedder",
    "summarise_error",
]
