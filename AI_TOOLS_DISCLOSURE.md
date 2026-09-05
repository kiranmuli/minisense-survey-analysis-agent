# AI Tools, Libraries & External Resources — Disclosure

As requested in the assignment, this document discloses the AI tools, libraries,
models, and external resources used to build this project.

## AI assistance
- **AI coding assistant** — used for pair-programming during development:
  scaffolding, boilerplate, iterating on the agent/RAG design, debugging, and
  drafting documentation. All architecture and design decisions were reviewed,
  understood, and validated by me; every component was tested and verified to
  work as described in the README.

## Language models (run locally via Ollama — no external API)
- **`llama3.2`** — the default chat/reasoning model powering the agents.
- **`qwen3:8b`** — an optional higher-quality alternative (configurable).
- **`nomic-embed-text`** — the embedding model used for RAG retrieval.

> No cloud LLM API (e.g. OpenAI) was used to run the system; everything runs
> locally. The code is written against the OpenAI-compatible interface, so it can
> switch to OpenAI via configuration only (see the README).

## Open-source libraries
| Library | Purpose |
|---|---|
| [Ollama](https://ollama.com) | Local LLM + embedding runtime |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Orchestrator → sub-agent graph |
| [langchain-text-splitters](https://github.com/langchain-ai/langchain) | RAG chunking fallback |
| [ChromaDB](https://www.trychroma.com/) | Local vector store |
| [Pydantic](https://docs.pydantic.dev/) | Typed inter-agent contracts |
| [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | HTTP API |
| [openai](https://github.com/openai/openai-python) (SDK) | OpenAI-compatible client (points at Ollama) |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Configuration via `.env` |

Exact versions are pinned in [`requirements.txt`](requirements.txt).

## External resources
- Official documentation of the libraries listed above.
- The survey dataset and product FAQ are **synthetically generated** by this
  project ([`data/generate_data.py`](data/generate_data.py) and
  [`data/product_faq.txt`](data/product_faq.txt)); no real or third-party data
  was used.
