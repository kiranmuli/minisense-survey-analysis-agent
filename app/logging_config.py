"""
Central logging setup.

One shared logger named "minisense" used across the API and the agent pipeline,
so every request and every pipeline stage produces consistent, timestamped logs
on stdout (which also means `docker compose logs` shows them).

Level is controlled by the LOG_LEVEL env var (default INFO).
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> logging.Logger:
    """Configure and return the shared 'minisense' logger (idempotent)."""
    logger = logging.getLogger("minisense")
    if logger.handlers:  # already configured — don't add duplicate handlers
        return logger

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | minisense | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # don't double-log through the root logger
    return logger


# Import this everywhere:  from app.logging_config import log
log = setup_logging()
