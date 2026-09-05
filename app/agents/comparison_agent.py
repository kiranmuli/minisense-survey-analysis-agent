"""
Step 5a — ComparisonAgent.

Compares TWO time periods and surfaces significant changes. This is where the
month-over-month drift we baked into the data (wait-time complaints rising in
August) becomes a concrete, reportable finding.

Design choice — deterministic, NOT LLM-driven:
  Comparing numbers is arithmetic, so we compute it exactly in Python. Letting
  an LLM "compare" numbers risks hallucinated deltas. The SummaryAgent will turn
  these exact facts into prose later. The orchestrator hands us two periods; we
  compute each period's metrics with the SAME tools the DataAgent uses.
"""

from __future__ import annotations

from app.models import (
    ComparisonAgentResult,
    PeriodMetrics,
    Period,
    TaskSpec,
    ThemeCount,
)
from app.tools import metrics as metric_tools
from app.tools.metrics import THEME_KEYWORDS

# Sensible defaults if the orchestrator didn't specify the two periods.
DEFAULT_PREVIOUS = Period(label="July 2026", start="2026-07-01", end="2026-07-31")
DEFAULT_CURRENT = Period(label="August 2026", start="2026-08-01", end="2026-08-31")

# Threshold below which we don't bother reporting a change as "notable".
MIN_PP = 1.0          # percentage points for CSAT / theme share
MIN_RATING = 0.05     # average-rating points


def _period_metrics(period: Period, business_id: str | None, top_k: int):
    """Compute one period's metrics. Returns (PeriodMetrics, {theme: share_pct})."""
    subset = metric_tools.filter_responses(
        metric_tools.load_responses(),
        business_id=business_id, start=period.start, end=period.end,
    )
    # Full theme shares (all themes) so we can compute deltas even for themes
    # that fall outside a single period's top-k.
    all_themes = metric_tools.top_themes(subset, k=len(THEME_KEYWORDS))
    share_by_theme = {t["theme"]: t["share_pct"] for t in all_themes}

    pm = PeriodMetrics(
        period_label=period.label,
        csat=metric_tools.compute_csat(subset),
        average_rating=metric_tools.average_rating(subset),
        response_count=metric_tools.response_count(subset),
        top_themes=[ThemeCount(**t) for t in all_themes[:top_k]],
    )
    return pm, share_by_theme


def _notable_changes(prev: PeriodMetrics, curr: PeriodMetrics,
                     prev_shares: dict, curr_shares: dict) -> list[str]:
    """Build human-readable statements about the biggest changes."""
    notes: list[str] = []

    # CSAT
    csat_delta = round(curr.csat - prev.csat, 1)
    if abs(csat_delta) >= MIN_PP:
        verb = "rose" if csat_delta > 0 else "fell"
        notes.append(f"CSAT {verb} {abs(csat_delta)} pp ({prev.csat}% -> {curr.csat}%).")

    # Average rating
    rating_delta = round(curr.average_rating - prev.average_rating, 2)
    if abs(rating_delta) >= MIN_RATING:
        verb = "up" if rating_delta > 0 else "down"
        notes.append(
            f"Average rating {verb} {abs(rating_delta)} "
            f"({prev.average_rating} -> {curr.average_rating})."
        )

    # Theme movements — rank every theme by absolute change in share.
    movers = []
    for theme in set(prev_shares) | set(curr_shares):
        p = prev_shares.get(theme, 0.0)
        c = curr_shares.get(theme, 0.0)
        delta = round(c - p, 1)
        if abs(delta) >= MIN_PP:
            movers.append((abs(delta), theme, p, c, delta))
    movers.sort(reverse=True)
    for _, theme, p, c, delta in movers[:3]:
        verb = "rose" if delta > 0 else "fell"
        # Include the exact pp delta so the SummaryAgent never has to subtract.
        notes.append(f"'{theme}' mentions {verb} {abs(delta)} pp (from {p}% to {c}% of responses).")

    if not notes:
        notes.append("No significant changes between the two periods.")
    return notes


def run(spec: TaskSpec) -> ComparisonAgentResult:
    """TaskSpec (with two periods) in -> ComparisonAgentResult out."""
    # Resolve the two periods and order them chronologically (previous, current).
    periods = spec.compare_periods or [DEFAULT_PREVIOUS, DEFAULT_CURRENT]
    periods = sorted(periods, key=lambda p: p.start)
    previous_p, current_p = periods[0], periods[-1]

    prev, prev_shares = _period_metrics(previous_p, spec.business_id, spec.top_k)
    curr, curr_shares = _period_metrics(current_p, spec.business_id, spec.top_k)

    return ComparisonAgentResult(
        current=curr,
        previous=prev,
        csat_change=round(curr.csat - prev.csat, 1),
        avg_rating_change=round(curr.average_rating - prev.average_rating, 2),
        notable_changes=_notable_changes(prev, curr, prev_shares, curr_shares),
    )
