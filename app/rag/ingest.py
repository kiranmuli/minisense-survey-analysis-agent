"""
Step 4b — RAG ingest: chunk -> embed -> store.

CHUNKING STRATEGY (and why):
  The FAQ is a structured document: an intro paragraph plus a series of
  self-contained "Q: ... A: ..." entries. The most meaningful retrieval unit is
  therefore ONE Q&A PAIR (or the intro block) — each already answers a single
  topic. So we split on blank lines into logical blocks rather than using
  fixed-size windows, which could sever a question from its answer or merge two
  unrelated topics. Very short blocks (like the title line) are merged forward.
  For an arbitrary document with no such structure, we fall back to LangChain's
  RecursiveCharacterTextSplitter (sentence-aware, fixed-ish size with overlap).

  Trade-off: structure-aware chunks are highly precise for a clean FAQ but
  assume the document has clear block boundaries; the recursive fallback keeps
  us robust when it doesn't.

STORAGE:
  Chunks are embedded with nomic-embed-text and stored in a local ChromaDB
  collection using COSINE distance (right metric for these embeddings).

Run it:
    python -m app.rag.ingest
"""

from __future__ import annotations

import re

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import CHROMA_DIR, settings
from app.rag.embed import embed_texts

COLLECTION_NAME = "faq_chunks"
MIN_CHARS = 40           # blocks shorter than this get merged into the next one
FALLBACK_CHUNK_SIZE = 500
FALLBACK_OVERLAP = 80


def _normalize(block: str) -> str:
    """Collapse wrapped lines and extra spaces so each chunk is clean, flat text."""
    return re.sub(r"\s+", " ", block).strip()


def chunk_document(text: str) -> list[str]:
    """
    Split the FAQ into logical blocks (intro + each Q&A pair).

    Falls back to recursive character splitting if the document has no blank-line
    structure to key off.
    """
    # Split on one-or-more blank lines -> logical blocks.
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks = [_normalize(b) for b in raw_blocks if _normalize(b)]

    if len(blocks) <= 1:
        # No structure to exploit -> sentence-aware recursive splitter.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=FALLBACK_CHUNK_SIZE,
            chunk_overlap=FALLBACK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]

    # Merge tiny blocks (e.g. the title) into the following block for context.
    merged: list[str] = []
    carry = ""
    for b in blocks:
        b = (carry + " " + b).strip() if carry else b
        if len(b) < MIN_CHARS:
            carry = b        # too short alone — carry it into the next block
        else:
            merged.append(b)
            carry = ""
    if carry:                # leftover short tail — attach to the last chunk
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)
    return merged


def get_client() -> "chromadb.ClientAPI":
    """Persistent Chroma client rooted at the project's chroma_db/ folder."""
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def ingest() -> int:
    """Chunk the FAQ, embed the chunks, and (re)build the Chroma collection."""
    text = settings.FAQ_TXT.read_text(encoding="utf-8")
    chunks = chunk_document(text)
    print(f"Chunked FAQ into {len(chunks)} blocks.")

    embeddings = embed_texts(chunks)
    print(f"Embedded {len(embeddings)} chunks (dim={len(embeddings[0])}).")

    client = get_client()
    # Delete any previous version so re-ingesting is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet — fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for these embeddings
    )
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": settings.FAQ_TXT.name, "index": i} for i in range(len(chunks))],
    )
    print(f"Stored {collection.count()} chunks in Chroma at {CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    ingest()
