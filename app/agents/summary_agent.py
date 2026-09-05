"""
Step 5b — SummaryAgent.

The only agent whose product is PROSE. It receives the structured outputs of the
other sub-agents (metrics, comparison, retrieved FAQ chunks) and writes the final
business-language answer.

Guardrails:
  * It must use ONLY the provided evidence — exact numbers and FAQ excerpts — so
    the narrative can't drift from the computed facts.
  * Output is a short paragraph (not raw numbers, not bullet dumps), matching the
    assessment's "coherent narrative" requirement.
"""

from __future__ import annotations

from typing import Optional

from app.llm import MODEL, client, nothink
from app.models import (
    ComparisonAgentResult,
    DataAgentResult,
    RAGAgentResult,
    SummaryAgentResult,
)


def _format_evidence(
    data: Optional[DataAgentResult],
    rag: Optional[RAGAgentResult],
    comparison: Optional[ComparisonAgentResult],
) -> str:
    """Turn the typed sub-agent results into a compact text block for the prompt."""
    lines: list[str] = []

    if data:
        m = data.metrics
        lines.append(f"[METRICS] scope: {data.scope}")
        if m.csat is not None:
            lines.append(f"  CSAT: {m.csat}%")
        if m.average_rating is not None:
            lines.append(f"  Average rating: {m.average_rating}")
        if m.response_count is not None:
            lines.append(f"  Responses: {m.response_count}")
        if m.top_themes:
            themes = ", ".join(f"{t.theme} ({t.share_pct}%)" for t in m.top_themes)
            lines.append(f"  Top themes: {themes}")

    if comparison:
        lines.append("[COMPARISON]")
        lines.append(f"  Previous: {comparison.previous.period_label} "
                     f"CSAT {comparison.previous.csat}%, avg {comparison.previous.average_rating}")
        lines.append(f"  Current:  {comparison.current.period_label} "
                     f"CSAT {comparison.current.csat}%, avg {comparison.current.average_rating}")
        for note in comparison.notable_changes:
            lines.append(f"  - {note}")

    if rag and rag.chunks:
        lines.append("[FAQ CONTEXT]")
        for c in rag.chunks:
            lines.append(f"  ({c.score}) {c.text}")

    return "\n".join(lines) if lines else "No evidence was gathered."


def run(
    question: str,
    data: Optional[DataAgentResult] = None,
    rag: Optional[RAGAgentResult] = None,
    comparison: Optional[ComparisonAgentResult] = None,
) -> SummaryAgentResult:
    """Synthesize the final narrative answer from the gathered evidence."""
    evidence = _format_evidence(data, rag, comparison)

    system = nothink(
        "You are SummaryAgent for a survey-analytics tool. Write a concise, "
        "business-language answer (one paragraph, 3-6 sentences) to the user's "
        "question. Use ONLY the evidence provided: quote the exact metrics and "
        "reflect period-over-period changes and their direction. Weave in relevant "
        "FAQ context where it adds business meaning. Do NOT invent numbers or "
        "facts that are not in the evidence, and do NOT do your own arithmetic — "
        "quote figures and changes (e.g. percentage-point deltas) exactly as given. "
        "Write plainly, no bullet points."
    )
    user = f"Question: {question}\n\nEvidence:\n{evidence}\n\nWrite the final answer."

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,  # a little warmth for readable prose, still grounded
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        # Fallback: return a plain, factual summary assembled from the evidence.
        answer = f"(LLM unavailable: {exc})\n{evidence}"

    return SummaryAgentResult(answer=answer)
