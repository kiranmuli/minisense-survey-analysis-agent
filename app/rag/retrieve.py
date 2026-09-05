"""
Step 4c — RAG retrieve: query -> top-k chunks.

Given a natural-language query, embed it with the SAME model used at ingest,
search the Chroma collection, and return the most relevant FAQ chunks with a
similarity score (1.0 = identical, lower = less related).
"""

from __future__ import annotations

from app.rag.embed import embed_one
from app.rag.ingest import COLLECTION_NAME, get_client


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Return the top-k FAQ chunks for `query` as dicts:
        {"text": str, "source": str, "score": float}
    Scores are cosine similarity (1 - cosine distance), rounded.
    """
    client = get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "FAQ collection not found. Run `python -m app.rag.ingest` first."
        ) from exc

    query_vec = embed_one(query)
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns lists-of-lists (one per query). We sent a single query.
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    out: list[dict] = []
    for text, meta, dist in zip(docs, metas, dists):
        out.append({
            "text": text,
            "source": meta.get("source", "product_faq.txt"),
            "score": round(1.0 - dist, 3),  # cosine distance -> similarity
        })
    return out


if __name__ == "__main__":
    # Quick manual check.
    for q in ["How long will I have to wait?", "What is your CSAT target?"]:
        print(f"\nQ: {q}")
        for c in retrieve(q, top_k=2):
            print(f"  [{c['score']}] {c['text'][:90]}...")
