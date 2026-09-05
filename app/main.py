"""
Step 7 — FastAPI application.

The HTTP interface over the whole pipeline. One endpoint does the work:

    POST /ask   { "question": "..." }  ->  AskResponse
        Runs the orchestrator and returns the final narrative answer PLUS all
        structured evidence (metrics, comparison, retrieved FAQ chunks) and the
        plan the orchestrator chose — so a reviewer can see exactly how the
        answer was produced.

    GET  /health   -> readiness check (data file + vector store present)
    GET  /         -> basic info

Run it:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import CHROMA_DIR, settings
from app.logging_config import log
from app.models import AskResponse
from app.orchestrator import answer_question

app = FastAPI(
    title="MiniSense — Survey Analysis Agent",
    description="Multi-agent + RAG system that answers business questions about survey feedback.",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log EVERY endpoint hit: method, path, status, and duration."""
    start = time.perf_counter()
    log.info(f"--> {request.method} {request.url.path}")
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    log.info(f"<-- {request.method} {request.url.path} {response.status_code} ({ms:.0f} ms)")
    return response


class AskRequest(BaseModel):
    """Body for POST /ask."""
    question: str = Field(..., min_length=3,
                          examples=["What are the top complaints for GreenLeaf Bistro "
                                    "this month and how do they compare to last month?"])


@app.get("/")
def root() -> dict:
    """Basic service info + pointer to the docs."""
    return {
        "service": "MiniSense Survey Analysis Agent",
        "llm_backend": settings.LLM_BACKEND,
        "model": settings.OLLAMA_MODEL if settings.LLM_BACKEND == "ollama" else settings.OPENAI_MODEL,
        "docs": "/docs",
        "ask": "POST /ask with {\"question\": \"...\"}",
    }


@app.get("/health")
def health() -> dict:
    """Readiness: confirm the dataset and vector store exist before serving."""
    data_ok = settings.SURVEY_JSON.exists()
    rag_ok = CHROMA_DIR.exists()
    status = "ok" if (data_ok and rag_ok) else "degraded"
    hints = []
    if not data_ok:
        hints.append("Run `python data/generate_data.py` to create the dataset.")
    if not rag_ok:
        hints.append("Run `python -m app.rag.ingest` to build the vector store.")
    return {"status": status, "data_present": data_ok, "vector_store_present": rag_ok, "hints": hints}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a natural-language business question via the multi-agent pipeline."""
    log.info(f"/ask question: {req.question!r}")
    try:
        resp = answer_question(req.question)
        log.info(f"/ask answered (plan={[s.agent.value for s in resp.plan]})")
        return resp
    except Exception as exc:
        # Surface a clean 500 with the reason rather than a raw stack trace.
        log.exception("/ask failed")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc
