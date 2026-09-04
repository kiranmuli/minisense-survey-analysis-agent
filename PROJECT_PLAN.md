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
| LLM | OpenAI (`gpt-4o-mini`) | Reasoning + function calling |
| Agents | LangGraph | Orchestrator → sub-agent graph |
| Structured I/O | Pydantic | Typed task specs & outputs (required) |
| Embeddings | sentence-transformers (local) | Free, offline after first download |
| Vector store | ChromaDB | Simplest local vector DB |
| Chunking | LangChain text splitters | Sentence-aware chunking |
| Data | pandas + Faker | Metrics + fake data generation |
| API | FastAPI + uvicorn | `/ask` endpoint |
| Config | python-dotenv | Load API key from `.env` |

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

- [ ] **Step 1 — Generate data** (Appendix A & B)
  - `data/generate_data.py` → `survey_responses.json` (50k–100k records)
  - expand `data/product_faq.txt` to ~500 words
  - commit: `feat: fake survey dataset + product FAQ`

- [ ] **Step 2 — Pydantic models + metric tools**
  - `app/models.py` (TaskSpec + typed agent outputs)
  - `app/tools/metrics.py` (`compute_csat`, avg rating, counts, top themes)
  - commit: `feat: structured models and metric tools`

- [ ] **Step 3 — DataAgent**
  - `app/agents/data_agent.py` — loads JSON, calls metric tools (tool calling demo)
  - commit: `feat: DataAgent with tool calling`

- [ ] **Step 4 — RAG pipeline** (Part 2)
  - `app/rag/ingest.py` (chunk + embed + store in Chroma)
  - `app/rag/retrieve.py` (top-k search)
  - `app/agents/rag_agent.py`
  - commit: `feat: RAG ingest + retrieve + RAGAgent`

- [ ] **Step 5 — ComparisonAgent + SummaryAgent**
  - `app/agents/comparison_agent.py`, `app/agents/summary_agent.py`
  - commit: `feat: comparison and summary agents`

- [ ] **Step 6 — Orchestrator (LangGraph)** (Part 1)
  - `app/orchestrator.py` — plan → route → synthesize
  - commit: `feat: orchestrator graph wiring all agents`

- [ ] **Step 7 — FastAPI endpoint**
  - `app/main.py` — `POST /ask`
  - commit: `feat: FastAPI /ask endpoint`

- [ ] **Step 8 — Evaluation + README**
  - 3 sample questions with retrieved chunks + final answers
  - Part 3 fine-tuning writeup (300–500 words)
  - design writeup + run instructions
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
