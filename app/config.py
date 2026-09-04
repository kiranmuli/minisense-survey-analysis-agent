"""
Central configuration for MiniSense.

All environment-dependent values (which LLM backend, model names, file paths)
are read HERE once. The rest of the code imports `settings` and calls
`get_llm_client()` / `get_chat_model()` instead of touching os.environ directly.

We run locally with Ollama by default, but the same code works against real
OpenAI just by flipping LLM_BACKEND=openai in .env — because Ollama exposes an
OpenAI-COMPATIBLE endpoint, so we use the one `openai` SDK for both.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load key=value pairs from .env into the environment (no-op if .env is absent).
load_dotenv()

# Project paths.  __file__ = .../app/config.py
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"  # on-disk vector store location


class Settings:
    """Read-once settings holder shared across the app."""

    # --- Which backend powers the agent "brain" ---
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")  # "ollama" | "openai"

    # --- Ollama (local) ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # --- OpenAI (cloud fallback) ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --- Embeddings backend ---
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "ollama")  # ollama|local|openai

    # --- File paths ---
    SURVEY_JSON: Path = DATA_DIR / "survey_responses.json"
    FAQ_TXT: Path = DATA_DIR / "product_faq.txt"


settings = Settings()


def get_chat_model() -> str:
    """Return the chat model name for the active backend."""
    return settings.OLLAMA_MODEL if settings.LLM_BACKEND == "ollama" else settings.OPENAI_MODEL


def get_llm_client():
    """
    Return an OpenAI-SDK client wired to the active backend.

    - ollama: point base_url at the local Ollama endpoint; api_key is a required
              placeholder that Ollama ignores.
    - openai: normal cloud client; requires a real OPENAI_API_KEY.

    Imported lazily so simple scripts (like data generation) don't need the SDK.
    """
    from openai import OpenAI

    if settings.LLM_BACKEND == "ollama":
        return OpenAI(base_url=settings.OLLAMA_BASE_URL, api_key="ollama")

    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "LLM_BACKEND=openai but OPENAI_API_KEY is empty. Add it to .env, "
            "or set LLM_BACKEND=ollama to run locally."
        )
    return OpenAI(api_key=settings.OPENAI_API_KEY)
