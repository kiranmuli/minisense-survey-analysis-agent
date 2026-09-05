"""
Step 4d — RAGAgent.

The thin agent wrapper the orchestrator routes to for document grounding.
It takes a structured TaskSpec, runs vector retrieval over the FAQ, and returns
a typed RAGAgentResult (query + retrieved chunks). The orchestrator later injects
these chunks into the final prompt so answers are grounded in the FAQ.

We use the question text directly as the search query. (A more advanced version
could ask the LLM to rewrite the question into a tighter search query first;
noted as a possible improvement, kept simple here for speed and transparency.)
"""

from __future__ import annotations

from app.models import RAGAgentResult, RetrievedChunk, TaskSpec
from app.rag.retrieve import retrieve


def run(spec: TaskSpec) -> RAGAgentResult:
    """TaskSpec in -> RAGAgentResult out."""
    query = spec.question
    hits = retrieve(query, top_k=spec.top_k)
    chunks = [RetrievedChunk(**h) for h in hits]
    return RAGAgentResult(query=query, chunks=chunks)
