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
    else:
        # Make the ABSENCE explicit so the model can't invent a comparison.
        lines.append("[COMPARISON] None. No period-over-period comparison was performed. "
                     "Do NOT mention any previous period or any change over time.")

    if rag and rag.chunks:
        lines.append("[FAQ CONTEXT]")
        for c in rag.chunks:
            lines.append(f"  ({c.score}) {c.text}")
    else:
        lines.append("[FAQ CONTEXT] None. No FAQ context was retrieved. "
                     "Do NOT reference or quote the FAQ.")

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
        "business-language answer (one paragraph, 2-5 sentences) to the user's "
        "question, using ONLY the evidence provided. Strict rules: "
        "(1) Never invent numbers or facts not in the evidence. "
        "(2) Do NOT do your own arithmetic — quote figures and deltas exactly as given. "
        "(3) Discuss a change between periods ONLY if a [COMPARISON] section with data "
        "is present; if it says None, do not mention any previous period or trend. "
        "(4) Reference the FAQ ONLY if a [FAQ CONTEXT] section with excerpts is present; "
        "if it says None, do not mention the FAQ. "
        "Write plainly, no bullet points."
    )
    user = f"Question: {question}\n\nEvidence:\n{evidence}\n\nWrite the final answer."

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,  # deterministic, grounded — no creative padding
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        # Fallback: return a plain, factual summary assembled from the evidence.
        answer = f"(LLM unavailable: {exc})\n{evidence}"

    return SummaryAgentResult(answer=answer)
