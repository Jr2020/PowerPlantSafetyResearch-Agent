"""In-memory vector store for severe_injury_data.json.

Each narrative is short (~200 chars average), so no chunking is needed here —
one Document per record. See embedding_cache.py for the shared build/cache
logic, and pdf_vector_store.py for a source that does need configurable
chunking.
"""

import json
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from embedding_cache import load_or_build_vector_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

SEVERE_INJURY_JSON_PATH = os.path.join(DATA_DIR, "severe_injury_data.json")
SEVERE_INJURY_CACHE_DIR = os.path.join(DATA_DIR, "severe_injury_embeddings_cache")
SEVERE_INJURY_VECTORS_PATH = os.path.join(SEVERE_INJURY_CACHE_DIR, "vectors.npy")
SEVERE_INJURY_RECORDS_PATH = os.path.join(SEVERE_INJURY_CACHE_DIR, "records.json")


def _load_severe_injury_documents() -> list[Document]:
    with open(SEVERE_INJURY_JSON_PATH, "r") as f:
        records = json.load(f)

    documents = []
    for record in records:
        metadata = {k: v for k, v in record.items() if k != "Final Narrative"}
        documents.append(Document(page_content=record["Final Narrative"], metadata=metadata))
    return documents


def load_severe_injury_vector_store() -> InMemoryVectorStore:
    """Load the severe injury vector store: from the on-disk cache if present,
    otherwise embed everything once and write the cache for next time."""
    return load_or_build_vector_store(
        _load_severe_injury_documents,
        SEVERE_INJURY_VECTORS_PATH,
        SEVERE_INJURY_RECORDS_PATH,
        label="severe injury narratives",
    )
