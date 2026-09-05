"""
Step 3 — DataAgent  (the required "tool calling from within an agent" demo).

What it does:
  Given a structured TaskSpec, the DataAgent asks the LLM to fetch exact survey
  metrics BY CALLING A TOOL. The LLM never does the arithmetic itself — it only
  decides *which* metrics to request; the numbers come from real Python
  (app/tools/metrics.run_metrics). The agent then returns a typed DataAgentResult.

Why tool calling (and not just calling run_metrics directly)?
  The assessment explicitly asks to demonstrate an agent invoking a tool. This
  is the canonical, honest pattern: the model reasons about intent, the tool
  guarantees correctness. We still keep a deterministic FALLBACK so a flaky
  local model can never break the pipeline.
"""

from __future__ import annotations

import json

from app.llm import MODEL, client, nothink
from app.logging_config import log
from app.models import DataAgentResult, MetricResult, TaskSpec, ThemeCount
from app.tools import metrics as metric_tools

# The tool the LLM is allowed to call, described in OpenAI function-calling format.
GET_METRICS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_survey_metrics",
        "description": (
            "Compute EXACT survey metrics over the survey dataset, with optional "
            "filters. Always use this tool to obtain numbers; never estimate them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "csat",
                            "average_rating",
                            "response_count",
                            "rating_distribution",
                            "top_themes",
                        ],
                    },
                    "description": "Which metrics to compute.",
                },
                "business_id": {
                    "type": "string",
                    "description": "Business id b01..b05. Omit to include all businesses.",
                },
                "start": {"type": "string", "description": "Inclusive start date YYYY-MM-DD."},
                "end": {"type": "string", "description": "Inclusive end date YYYY-MM-DD."},
                "top_k": {"type": "integer", "description": "How many themes to return."},
            },
            "required": ["metrics"],
        },
    },
}

# Default metrics if the TaskSpec didn't name any.
DEFAULT_METRICS = ["csat", "average_rating", "response_count", "top_themes"]


def _scope_text(business_name: str | None, business_id: str | None,
                start: str | None, end: str | None, rows: int) -> str:
    """Human-readable description of what the metrics were computed over."""
    who = business_name or (business_id or "all businesses")
    when = f"{start} to {end}" if (start or end) else "all dates"
    return f"{who} | {when} | {rows:,} responses"


def _execute_tool(args: dict, spec: TaskSpec) -> dict:
    """
    Run the real metric computation.

    Filters (business/date) come from the TaskSpec — the orchestrator is the
    source of truth for *what data* to look at. The LLM's tool call chooses
    *which metrics*. We merge: metrics/top_k from the model (falling back to the
    spec), filters from the spec (falling back to the model's args).
    """
    # Always compute the core metrics as a floor, then add anything the plan or
    # the model's tool call additionally asked for. This prevents a weaker model
    # from accidentally dropping a key metric (e.g. csat) via its tool arguments.
    requested = args.get("metrics") or []
    # Small models sometimes send metrics as a string ("csat" or "csat,avg")
    # instead of a list — coerce defensively so concatenation can't crash.
    if isinstance(requested, str):
        requested = [m.strip() for m in requested.split(",") if m.strip()]
    metrics = list(dict.fromkeys(DEFAULT_METRICS + list(spec.metrics or []) + list(requested)))
    # top_k may arrive as a string ("5") from a weaker model — coerce to int.
    try:
        top_k = int(args.get("top_k") or spec.top_k)
    except (TypeError, ValueError):
        top_k = spec.top_k

    business_id = spec.business_id or args.get("business_id") or None
    start = (spec.period.start if spec.period else None) or args.get("start")
    end = (spec.period.end if spec.period else None) or args.get("end")

    raw = metric_tools.run_metrics(
        metrics=metrics, business_id=business_id, start=start, end=end, top_k=top_k
    )
    raw["_used"] = {"business_id": business_id, "start": start, "end": end, "metrics": metrics}
    return raw


def _to_result(raw: dict, spec: TaskSpec) -> DataAgentResult:
    """Wrap the raw metric dict into the typed DataAgentResult contract."""
    themes = raw.get("top_themes")
    metric_obj = MetricResult(
        csat=raw.get("csat"),
        average_rating=raw.get("average_rating"),
        response_count=raw.get("response_count"),
        rating_distribution=raw.get("rating_distribution"),
        top_themes=[ThemeCount(**t) for t in themes] if themes else None,
    )
    used = raw.get("_used", {})
    scope = _scope_text(spec.business_name, used.get("business_id"),
                        used.get("start"), used.get("end"), raw.get("_scope_count", 0))
    return DataAgentResult(scope=scope, metrics=metric_obj)


def run(spec: TaskSpec) -> DataAgentResult:
    """Entry point: TaskSpec in -> DataAgentResult out (via LLM tool calling)."""
    wanted = spec.metrics or DEFAULT_METRICS
    system = nothink(
        "You are DataAgent, a precise survey-analytics worker. To answer, you "
        "MUST call the get_survey_metrics tool to fetch exact numbers. Do not "
        "compute metrics yourself."
    )
    # Tell the model exactly which metrics/filters the plan wants — it should
    # reflect these in its tool call.
    filt = []
    if spec.business_id:
        filt.append(f"business_id={spec.business_id}")
    if spec.period:
        filt.append(f"dates {spec.period.start}..{spec.period.end}")
    user = (
        f"Question: {spec.question}\n"
        f"Compute these metrics: {wanted}.\n"
        f"Filters: {', '.join(filt) if filt else 'none (all data)'}.\n"
        f"Call get_survey_metrics now."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            tools=[GET_METRICS_TOOL],
            temperature=0,
        )
        tool_calls = resp.choices[0].message.tool_calls
        if tool_calls:
            # Parse the model's chosen arguments (robust to malformed JSON).
            try:
                args = json.loads(tool_calls[0].function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            raw = _execute_tool(args, spec)
            return _to_result(raw, spec)
    except Exception as exc:  # network/model hiccup -> fall through to fallback
        log.warning(f"[DataAgent] tool-calling path failed ({exc}); using deterministic fallback.")

    # ---- Deterministic fallback: run straight from the TaskSpec ----
    raw = _execute_tool({}, spec)
    return _to_result(raw, spec)
