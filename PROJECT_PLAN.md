# MiniSense — Survey Analysis Agent · Project Plan

A multi-agent + RAG system that answers business questions about survey feedback.
This file is the single source of truth for **what we're building, in what order, and why**.
We complete one step, tick its box, and make one commit per step.

---

## 🎯 Goal (from the assessment)

Build a runnable AI system where:
- An **Orchestrator agent** takes a natural-language business question, splits it into sub-tasks, routes them to **sub-agents**, and synthesizes a final narrative answer.
- A **RAG pipeline** grounds answers in a product FAQ document.
- A **README** explains the design + a 300–500 word fine-tuning strategy.

Deliverable: GitHub repo with README, runnable code, and a design writeup.

---

## 🧱 Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Standard for AI/agent tooling |
| LLM runtime | **Ollama (local)** | Free, no API key, already installed |
| LLM model | **`qwen3:8b`** + `/no_think` | Accurate tool calls + clean JSON output |
| LLM SDK | `openai` SDK → Ollama's OpenAI-compatible endpoint | Same code works for Ollama *or* real OpenAI |
| Agents | LangGraph | Orchestrator → sub-agent graph |
| Structured I/O | Pydantic | Typed task specs & outputs (required) |
| Embeddings | **`nomic-embed-text`** (local via Ollama) | Free, already installed |
| Vector store | ChromaDB | Simplest local vector DB |
| Chunking | LangChain text splitters | Sentence-aware chunking |
| Data | pandas + Faker | Metrics + fake data generation |
| API | FastAPI + uvicorn | `/ask` endpoint |
| Config | python-dotenv | Load settings from `.env` |

> **Note:** We run fully local via Ollama (no cost, no key). `qwen3:8b` is a "thinking"
> model, so every system prompt ends with `/no_think` to suppress reasoning tokens and
> get clean structured output. Switching to real OpenAI later is a one-line `.env` change
> (`LLM_BACKEND=openai` + `OPENAI_API_KEY=...`).

---

## 🗺️ End-to-End Flow

```
User question
      │
      ▼
┌─────────────────────┐
│   Orchestrator      │  1. Understand the question
│   (Planner agent)   │  2. Build structured TaskSpec(s)
└─────────┬───────────┘  3. Route to sub-agents
          │
          ├──────────────┬──────────────┬───────────────┐
          ▼              ▼              ▼               ▼
   ┌───────────┐  ┌───────────┐  ┌─────────────┐  ┌───────────┐
   │ DataAgent │  │ RAGAgent  │  │ Comparison  │  │ Summary   │
   │ metrics   │  │ FAQ chunks│  │ Agent       │  │ Agent     │
   │ (tools)   │  │ (vector)  │  │ period diff │  │ narrative │
   └─────┬─────┘  └─────┬─────┘  └──────┬──────┘  └─────┬─────┘
         │              │               │               │
         └──────────────┴───────────────┴───────────────┘
                        │ structured results (Pydantic/JSON)
                        ▼
              ┌─────────────────────┐
              │   Orchestrator      │  Synthesize final
              │   synthesis         │  business-language answer
              └─────────┬───────────┘
                        ▼
                Final narrative answer
```

---

## ✅ Step-by-Step Checklist (one commit per step)

- [x] **Step 0 — Project setup**
  - venv, `requirements.txt`, `.gitignore`, `.env.example`, `app/config.py`, folder skeleton
  - commit: `chore: project setup and dependencies`

- [x] **Step 1 — Generate data** (Appendix A & B)
  - `data/generate_data.py` → `survey_responses.json` (60k records, rating-correlated free-text, month-over-month drift)
  - expanded `data/product_faq.txt` to ~500 words (GreenLeaf Bistro)
  - commit: `feat: fake survey dataset + product FAQ`

- [x] **Step 2 — Pydantic models + metric tools**
  - `app/models.py` (TaskSpec + typed agent outputs + AskResponse)
  - `app/tools/metrics.py` (`compute_csat`, avg rating, counts, distribution, top themes; defensive coercion; `run_metrics` dispatcher)
  - commit: `feat: structured models and metric tools`

- [x] **Step 3 — DataAgent**
  - `app/llm.py` (shared client + `/no_think` helper)
  - `app/agents/data_agent.py` — LLM calls `get_survey_metrics` tool; deterministic fallback; returns typed `DataAgentResult`
  - commit: `feat: DataAgent with tool calling`

- [x] **Step 4 — RAG pipeline** (Part 2)
  - `app/rag/embed.py` (nomic-embed-text via Ollama)
  - `app/rag/ingest.py` (structure-aware Q&A chunking + cosine Chroma store; recursive fallback)
  - `app/rag/retrieve.py` (top-k cosine search with similarity scores)
  - `app/agents/rag_agent.py` (typed `RAGAgentResult`)
  - commit: `feat: RAG ingest + retrieve + RAGAgent`

- [x] **Step 5 — ComparisonAgent + SummaryAgent**
  - `app/agents/comparison_agent.py` (deterministic two-period diff + notable-change detection)
  - `app/agents/summary_agent.py` (LLM narrative grounded strictly in structured evidence)
  - commit: `feat: comparison and summary agents`

- [x] **Step 6 — Orchestrator (LangGraph)** (Part 1)
  - `app/orchestrator.py` — LLM `build_plan` tool + deterministic guards → LangGraph route (data/rag/comparison) → synthesize
  - hardened: pre-computed pp deltas in comparison evidence; UTF-8 CLI output
  - commit: `feat: orchestrator graph wiring all agents`

- [x] **Step 7 — FastAPI endpoint**
  - `app/main.py` — `POST /ask` (returns answer + evidence + plan), `GET /health`, `GET /`
  - hardened SummaryAgent: explicit absence markers + strict no-hallucination rules (fixed invented comparison/FAQ)
  - commit: `feat: FastAPI /ask endpoint`

- [x] **Step 7.5 — Dockerize**
  - `Dockerfile`, `docker-compose.yml` (Ollama + app), `docker/entrypoint.sh`, `.dockerignore`, `.gitattributes`
  - trimmed unused heavy deps (sentence-transformers, langchain) to slim the image
  - `docker compose up --build` → self-contained stack (auto-pulls models, generates data, builds index, serves)
  - commit: `feat: dockerize with compose (ollama + app)`

- [x] **Step 8 — Evaluation + README**
  - README: architecture (mermaid), tech stack, structure, setup/run (local + Docker), API, design decisions, observability, limitations
  - Part 2 evaluation: 3 sample questions with retrieved chunks + answers + honest commentary (caught & fixed a need_faq bug)
  - Part 3 fine-tuning writeup (409 words)
  - commit: `docs: README, evaluation, and fine-tuning design`

---

## 📌 Setup notes (do once)

1. `copy .env.example .env` then add your real `OPENAI_API_KEY`.
2. `.\.venv\Scripts\Activate.ps1` to activate the environment.
3. `pip install -r requirements.txt` (already run during Step 0).

---

## 📝 Progress Log

| Date | Step | Notes |
|---|---|---|
| 2026-09-04 | Step 0 | Setup complete; dependencies installed. |
| 2026-09-04 | Step 0.1 | Switched to local Ollama (no API key). LLM=qwen3:8b, embeddings=nomic-embed-text. Verified chat + tool calling work. |
| 2026-09-05 | Step 1 | Generated 60k survey records + ~500-word FAQ. Verified signal: avg rating 3.85→3.55 Jul→Aug; wait-time complaints 19%→39%. JSON is gitignored (regenerable). |
| 2026-09-05 | Step 2 | Pydantic contracts (TaskSpec, typed results) + metric tools. Verified on 60k rows: CSAT 64.9%, Aug/GreenLeaf top theme Wait Time 27.4%, string-rating coercion safe. |
| 2026-09-05 | Step 3 | DataAgent with real LLM tool calling (qwen3:8b -> get_survey_metrics). Verified Aug/GreenLeaf CSAT 60.1, avg 3.55, top theme Wait Time. Deterministic fallback in place. |
| 2026-09-05 | Step 4 | RAG pipeline: 10 Q&A chunks embedded (nomic, dim=768) into cosine Chroma. Retrieval verified incl. semantic match (phone->app booking). chroma_db/ gitignored. |
| 2026-09-05 | Step 5 | ComparisonAgent (Jul vs Aug: CSAT -9.7pp, Wait Time 19.5%->27.4%) + SummaryAgent. Verified full evidence->narrative with accurate, non-hallucinated numbers. |
| 2026-09-05 | Step 6 | LangGraph orchestrator: plan->data/rag/comparison->summarize. Verified end-to-end on comparison Q (routed 3 agents, business detected, Wait Time +7.9pp correct). Fixed LLM arithmetic drift by pre-computing deltas. |
| 2026-09-05 | Step 7 | FastAPI /ask + /health. Verified 200 via TestClient. Caught & fixed a grounding bug: single-month Q had invented a July comparison + FAQ; hardened SummaryAgent with explicit absence markers + temp 0. |
| 2026-09-05 | Step 7.5 | Dockerized: compose stack (ollama + app), entrypoint auto-pulls models + builds data/index. Trimmed unused torch/langchain deps. Compose validated; image build verified. |
| 2026-09-05 | Extra | Switched default to llama3.2 (~30-90s vs 18 min); hardened DataAgent tool-arg coercion; added full DEBUG trace logging of prompts/tool-calls/metrics/evidence. |
| 2026-09-05 | Step 8 | README + Part 2 evaluation (3 Qs, real chunks/answers, honest commentary) + Part 3 fine-tuning essay (409 words). Fixed need_faq so "target/policy" questions retrieve FAQ. All required deliverables complete. |
