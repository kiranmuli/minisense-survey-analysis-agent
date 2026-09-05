"""
Step 2b — Metric tools.

These are the plain, deterministic functions the DataAgent calls as TOOLS
(the assessment's required "tool calling" demo, e.g. `compute_csat(responses)`).

Design principles:
  * PURE & EXACT: metrics are computed in Python, never guessed by the LLM.
    The model decides *which* metric and *which* filter; the math happens here.
  * DEFENSIVE: small local models sometimes pass ratings as strings ("5") or
    odd values. Every rating is coerced and range-checked so a bad argument
    can never crash the pipeline.
  * DATASET STAYS SERVER-SIDE: we never send 60k rows through the LLM. The tool
    call carries only filters (business, dates); the data is loaded here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from app.config import settings

# A response counts as "satisfied" for CSAT if its rating is >= this.
SATISFIED_THRESHOLD = 4

# Keyword -> theme map used to mine themes from free-text. Rule-based on purpose:
# it's transparent and free. (Part 3 of the assessment proposes replacing this
# with a fine-tuned classifier at scale.)
THEME_KEYWORDS = {
    "Food Quality": ["food", "meal", "menu", "tasted", "delicious", "bland", "cold", "quality", "avocado"],
    "Wait Time": ["wait", "waited", "slow", "minutes", "quick", "efficient", "line", "fast"],
    "Staff": ["staff", "team", "rude", "polite", "attentive", "acknowledge", "friendly", "helpful"],
    "Cleanliness": ["clean", "dirty", "spotless", "restroom", "sticky", "sanit"],
    "Price / Value": ["price", "priced", "overpriced", "value", "expensive", "portion", "cheap"],
    "Ambiance": ["atmosphere", "noisy", "cramped", "lighting", "music", "seating", "ambiance", "relax"],
    "App / Booking": ["app", "booking", "online", "checkout", "crash", "buggy", "reserve"],
}


# ---------------------------------------------------------------------------
# Data loading + filtering
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_responses(path: Optional[str] = None) -> tuple[dict, ...]:
    """
    Load survey responses once and cache them (the JSON is ~15 MB).

    Returns a tuple (immutable) so lru_cache can hold it safely. Callers treat
    each item as a normal dict.
    """
    p = path or str(settings.SURVEY_JSON)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return tuple(data["responses"])


def filter_responses(
    responses,
    business_id: Optional[str] = None,
    start: Optional[str] = None,   # ISO date inclusive
    end: Optional[str] = None,     # ISO date inclusive
) -> list[dict]:
    """Return only responses matching the optional business + date-range filters."""
    out = []
    for r in responses:
        if business_id and r.get("business_id") != business_id:
            continue
        d = r.get("date", "")
        if start and d < start:
            continue
        if end and d > end:
            continue
        out.append(r)
    return out


def _coerce_rating(value) -> Optional[int]:
    """Turn a rating into an int in 1..5, or None if it's junk."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    return iv if 1 <= iv <= 5 else None


# ---------------------------------------------------------------------------
# The metric tools (each takes a list of response dicts)
# ---------------------------------------------------------------------------
def compute_csat(responses) -> float:
    """CSAT = % of responses rated 4 or 5. Returns 0.0 on an empty set."""
    ratings = [r for r in (_coerce_rating(x.get("rating")) for x in responses) if r is not None]
    if not ratings:
        return 0.0
    satisfied = sum(1 for r in ratings if r >= SATISFIED_THRESHOLD)
    return round(100.0 * satisfied / len(ratings), 1)


def average_rating(responses) -> float:
    """Mean 1-5 rating. Returns 0.0 on an empty set."""
    ratings = [r for r in (_coerce_rating(x.get("rating")) for x in responses) if r is not None]
    if not ratings:
        return 0.0
    return round(sum(ratings) / len(ratings), 2)


def response_count(responses) -> int:
    """Number of responses in the set."""
    return len(responses)


def rating_distribution(responses) -> dict[int, int]:
    """Count of responses at each rating 1..5."""
    dist = {i: 0 for i in range(1, 6)}
    for x in responses:
        r = _coerce_rating(x.get("rating"))
        if r is not None:
            dist[r] += 1
    return dist


def top_themes(responses, k: int = 5) -> list[dict]:
    """
    Mine the most-mentioned themes from free-text.

    A response "mentions" a theme if any of that theme's keywords appears in its
    free_text (case-insensitive). Returns the top-k themes as dicts:
        {"theme": ..., "mentions": int, "share_pct": float}
    where share_pct is % of ALL responses in the set that mention the theme.
    """
    total = len(responses) or 1
    counts = {theme: 0 for theme in THEME_KEYWORDS}
    for x in responses:
        text = str(x.get("free_text", "")).lower()
        if not text:
            continue
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                counts[theme] += 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"theme": t, "mentions": n, "share_pct": round(100.0 * n / total, 1)}
        for t, n in ranked[:k]
        if n > 0
    ]


# ---------------------------------------------------------------------------
# Convenience dispatcher — what the DataAgent's tool call ultimately runs.
# ---------------------------------------------------------------------------
# Maps a metric name (what the LLM asks for) to its function.
METRIC_FUNCS = {
    "csat": compute_csat,
    "average_rating": average_rating,
    "response_count": response_count,
    "rating_distribution": rating_distribution,
    "top_themes": top_themes,
}


def run_metrics(
    metrics: list[str],
    business_id: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Load + filter the data, then compute each requested metric.

    This is the single entry point the DataAgent exposes as a tool: the LLM
    passes metric names + filters, and gets back exact numbers.
    """
    subset = filter_responses(load_responses(), business_id=business_id, start=start, end=end)
    result: dict = {}
    for name in metrics:
        func = METRIC_FUNCS.get(name)
        if func is None:
            continue  # ignore unknown metric names rather than erroring
        result[name] = func(subset, top_k) if name == "top_themes" else func(subset)
    result["_scope_count"] = len(subset)  # how many rows the metrics were based on
    return result
