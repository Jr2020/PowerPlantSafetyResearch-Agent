"""In-memory vector store for arbitrary PDF documents.

Unlike severe_injury_data.json (short narratives, no chunking needed), a PDF
is one large text blob that has to be split into chunks before embedding, and
the right chunk size/overlap varies by document (a dense spec sheet vs. a
narrative report). chunk_size/chunk_overlap are exposed as call-time
parameters for that reason. See embedding_cache.py for the shared build/cache
logic used by both this and vector_store.py.
"""

import os

import pymupdf
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_cache import get_embeddings, load_or_build_vector_store

PDF_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "pdf_embeddings_cache"
)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def _cache_paths(pdf_path: str, chunk_size: int, chunk_overlap: int) -> tuple[str, str]:
    # Keyed by filename + chunk params, not just filename: changing the chunk
    # size for a PDF that's already cached should re-embed, not silently
    # reuse chunks built with the old size.
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_dir = os.path.join(PDF_CACHE_ROOT, f"{stem}__cs{chunk_size}_ov{chunk_overlap}")
    return (
        os.path.join(cache_dir, "vectors.npy"),
        os.path.join(cache_dir, "records.json"),
    )


def _load_pdf_documents(pdf_path: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = []
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            for chunk_index, chunk in enumerate(splitter.split_text(text)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_number,
                            "chunk": chunk_index,
                        },
                    )
                )
    return documents


def load_pdf_vector_store(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> InMemoryVectorStore:
    """Load a PDF's vector store: from the on-disk cache if present, otherwise
    chunk and embed the PDF once and write the cache for next time."""
    vectors_path, records_path = _cache_paths(pdf_path, chunk_size, chunk_overlap)
    return load_or_build_vector_store(
        lambda: _load_pdf_documents(pdf_path, chunk_size, chunk_overlap),
        vectors_path,
        records_path,
        label=os.path.basename(pdf_path),
    )


def load_pdf_vector_stores(
    pdf_paths: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> InMemoryVectorStore:
    """Load and merge multiple PDFs into one combined vector store searchable
    as a single corpus. Each PDF is still cached independently by
    load_pdf_vector_store, so adding a new PDF to the list only embeds that
    new file — it doesn't re-embed the ones already cached."""
    combined = InMemoryVectorStore(embedding=get_embeddings())
    for pdf_path in pdf_paths:
        store = load_pdf_vector_store(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        combined.store.update(store.store)
    return combined
