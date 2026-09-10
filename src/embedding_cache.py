"""Shared helpers for building and caching InMemoryVectorStore instances.

Embeddings are computed once per dataset and cached to disk as a compact
numpy array plus a JSON sidecar of texts/metadata, then loaded straight from
cache on every run after the first. The built-in InMemoryVectorStore.dump()/
load() are not used here because they serialize every float of every vector
as JSON text, which blows up to gigabytes at any real scale; numpy's binary
format keeps that compact.

This module is schema-agnostic: it only knows how to embed a list of
Documents and cache/reload the result. Dataset-specific loading (how to turn
a JSON file, a PDF, etc. into Documents) lives in its own module — see
vector_store.py and pdf_vector_store.py.
"""

import json
import os
import time
from typing import Callable

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# Every embedding call in this app must go through get_embeddings() so the
# model stays pinned to this one value.
EMBEDDING_MODEL = "text-embedding-3-small"

EMBED_BATCH_SIZE = 500

_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """Shared embeddings client, pinned to EMBEDDING_MODEL."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def build_and_cache_vector_store(
    documents: list[Document],
    vectors_path: str,
    records_path: str,
    label: str = "documents",
) -> InMemoryVectorStore:
    """Embed `documents` with get_embeddings(), reporting batch progress, and
    cache the resulting vectors/records to vectors_path/records_path."""
    total = len(documents)
    print(
        f"No cached embeddings found for {label} — embedding {total} chunks "
        f"with '{EMBEDDING_MODEL}' now (one-time cost; results will be cached "
        f"to disk so future runs can skip this step)."
    )

    vector_store = InMemoryVectorStore(embedding=get_embeddings())

    start_time = time.monotonic()
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = documents[start:start + EMBED_BATCH_SIZE]
        vector_store.add_documents(batch)
        done = min(start + EMBED_BATCH_SIZE, total)
        elapsed = time.monotonic() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(
            f"Embedding progress: {done}/{total} ({done / total:.0%}) — "
            f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
        )

    os.makedirs(os.path.dirname(vectors_path), exist_ok=True)

    entries = list(vector_store.store.values())
    vectors = np.array([entry["vector"] for entry in entries], dtype=np.float32)
    records_out = [
        {"id": entry["id"], "text": entry["text"], "metadata": entry["metadata"]}
        for entry in entries
    ]

    np.save(vectors_path, vectors)
    with open(records_path, "w") as f:
        json.dump(records_out, f)

    total_elapsed = time.monotonic() - start_time
    print(
        f"Done embedding {label} in {total_elapsed:.0f}s. Cached to "
        f"'{os.path.dirname(vectors_path)}'."
    )
    return vector_store


def load_cached_vector_store(vectors_path: str, records_path: str) -> InMemoryVectorStore:
    """Load a vector store previously cached by build_and_cache_vector_store."""
    vectors = np.load(vectors_path)
    with open(records_path, "r") as f:
        records = json.load(f)

    vector_store = InMemoryVectorStore(embedding=get_embeddings())
    for record, vector in zip(records, vectors):
        vector_store.store[record["id"]] = {
            "id": record["id"],
            "vector": vector.tolist(),
            "text": record["text"],
            "metadata": record["metadata"],
        }
    return vector_store


def load_or_build_vector_store(
    documents_fn: Callable[[], list[Document]],
    vectors_path: str,
    records_path: str,
    label: str = "documents",
) -> InMemoryVectorStore:
    """Load a cached vector store if present, otherwise build one from
    documents_fn() (only called on a cache miss, since it can be expensive
    — e.g. parsing/chunking a large source file) and cache it for next time.
    """
    print(f"Checking for cached {label} embeddings...")
    if os.path.exists(vectors_path) and os.path.exists(records_path):
        print(
            f"Found cached embeddings for {label} — skipping re-embedding "
            f"and loading them into memory."
        )
        return load_cached_vector_store(vectors_path, records_path)
    return build_and_cache_vector_store(documents_fn(), vectors_path, records_path, label=label)
