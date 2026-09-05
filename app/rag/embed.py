"""
Step 4a — Embedding helper.

One job: turn text into vectors. Both the ingest step (embedding FAQ chunks)
and the retrieve step (embedding the query) call `embed_texts` so they always
use the SAME model — a hard requirement for vector search to be meaningful.

Default backend is Ollama's `nomic-embed-text` (local, free). Set
EMBEDDING_BACKEND=openai in .env to switch to text-embedding-3-small instead.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import settings

# Build a client for whichever embedding backend is configured.
if settings.EMBEDDING_BACKEND == "openai":
    _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    _model = "text-embedding-3-small"
else:  # "ollama" (default) — nomic-embed-text via the OpenAI-compatible API
    _client = OpenAI(base_url=settings.OLLAMA_BASE_URL, api_key="ollama")
    _model = settings.OLLAMA_EMBED_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings into a list of float vectors (order preserved)."""
    resp = _client.embeddings.create(model=_model, input=texts)
    return [item.embedding for item in resp.data]


def embed_one(text: str) -> list[float]:
    """Convenience: embed a single string into one vector."""
    return embed_texts([text])[0]
