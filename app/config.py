"""
Central configuration for MiniSense.

Everything that depends on environment variables (API keys, model names, file
paths) is read HERE once, so the rest of the code just imports `settings`
instead of reading os.environ all over the place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load key=value pairs from the .env file into the process environment.
# (Does nothing if .env is missing — we fall back to defaults below.)
load_dotenv()

# Project root = the folder that contains this "app" package's parent.
# __file__ = .../app/config.py  ->  .parent = .../app  ->  .parent = project root
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"  # where the vector store lives on disk


class Settings:
    """Simple settings holder. Read once, use everywhere."""

    # --- LLM ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --- RAG ---
    # "local"  -> sentence-transformers (free, offline after first download)
    # "openai" -> text-embedding-3-small (uses your API quota)
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "local")

    # --- File paths (used by data generation + RAG ingest) ---
    SURVEY_JSON: Path = DATA_DIR / "survey_responses.json"
    FAQ_TXT: Path = DATA_DIR / "product_faq.txt"

    def require_api_key(self) -> None:
        """Call this before any real OpenAI request so we fail with a clear message."""
        if not self.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )


# A single shared instance the whole app imports:  from app.config import settings
settings = Settings()
