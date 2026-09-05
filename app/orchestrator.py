"""
Step 6 — Orchestrator (Planner) built on LangGraph.

This is the top level of the two-level agent architecture. It:

  1. PLANS   — uses the LLM (via a `build_plan` tool call) to read the question
               and decide which sub-agents are needed and with what filters,
               producing structured TaskSpec objects (never raw text).
  2. ROUTES  — a LangGraph StateGraph runs each needed sub-agent, writing its
               typed result into shared state.
  3. SYNTHESIZES — hands all structured results to the SummaryAgent for the
               final business-language answer.

Robustness: the dataset only covers July & August 2026, so we tell the planner
"this month = August 2026, last month = July 2026", and we back the LLM's plan
with DETERMINISTIC guards (business-name matching, period mapping, metric
validation) plus a keyword-based fallback if the model call fails.

Graph shape (linear; each node acts only if the plan asked for it):
    START -> plan -> data -> rag -> comparison -> summarize -> END
"""

from __future__ import annotations

import json
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents import comparison_agent, data_agent, rag_agent, summary_agent
from app.llm import MODEL, client, nothink
from app.models import (
    AgentType,
    AskResponse,
    ComparisonAgentResult,
    DataAgentResult,
    Period,
    RAGAgentResult,
    TaskSpec,
)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
JULY = Period(label="July 2026", start="2026-07-01", end="2026-07-31")
AUGUST = Period(label="August 2026", start="2026-08-01", end="2026-08-31")

# The businesses in the dataset (name -> id). Used to resolve names in questions.
BUSINESS_NAME_TO_ID = {
    "greenleaf bistro": "b01",
    "quickfit gym": "b02",
    "urban threads boutique": "b03",
    "brightsmile dental": "b04",
    "petpals grooming": "b05",
}
ID_TO_NAME = {v: k.title() for k, v in BUSINESS_NAME_TO_ID.items()}

VALID_METRICS = ["csat", "average_rating", "response_count", "rating_distribution", "top_themes"]
DEFAULT_METRICS = ["csat", "average_rating", "response_count", "top_themes"]

# Words that imply a period-over-period comparison.
COMPARE_HINTS = ["compare", "comparison", " vs", "versus", "last month", "previous",
                 "trend", "change", "changed", "month over month", "month-over-month"]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
BUILD_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "build_plan",
        "description": "Decide how to answer a survey question about the dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "business": {
                    "type": "string",
                    "description": "Business name mentioned, or 'all' if none/multiple.",
                },
                "time_scope": {
                    "type": "string",
                    "enum": ["single", "compare", "all"],
                    "description": "'compare' for month-over-month; 'single' for one month; 'all' otherwise.",
                },
                "month": {
                    "type": "string",
                    "enum": ["July", "August"],
                    "description": "Which month, when time_scope is 'single'. 'this month'=August, 'last month'=July.",
                },
                "need_faq": {
                    "type": "boolean",
                    "description": "Whether FAQ/business context would help answer.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": VALID_METRICS},
                    "description": "Which metrics are relevant.",
                },
            },
            "required": ["time_scope"],
        },
    },
}


def _detect_business(question: str) -> tuple[Optional[str], Optional[str]]:
    """Deterministically find a known business in the question -> (id, name)."""
    q = question.lower()
    for name, bid in BUSINESS_NAME_TO_ID.items():
        # match on full name or the distinctive first word (e.g. "greenleaf")
        if name in q or name.split()[0] in q:
            return bid, name.title()
    return None, None


def _heuristic_plan(question: str) -> dict:
    """Keyword-based plan used if the LLM planner is unavailable."""
    q = question.lower()
    compare = any(h in q for h in COMPARE_HINTS)
    if compare:
        scope = "compare"
    elif "july" in q or "august" in q:
        scope = "single"
    else:
        scope = "all"
    month = "August" if "august" in q else ("July" if "july" in q else "August")
    return {"time_scope": scope, "month": month, "need_faq": True, "metrics": DEFAULT_METRICS}


def _llm_plan(question: str) -> dict:
    """Ask the LLM to build a routing plan via a tool call; fall back on error."""
    system = nothink(
        "You are the Orchestrator/Planner for a survey-analytics tool. The dataset "
        "covers ONLY July 2026 and August 2026. Treat 'this month' as August 2026 and "
        "'last month' as July 2026. Known businesses: GreenLeaf Bistro, QuickFit Gym, "
        "Urban Threads Boutique, BrightSmile Dental, PetPals Grooming. Decide the plan "
        "by calling build_plan."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": question}],
            tools=[BUILD_PLAN_TOOL],
            temperature=0,
        )
        calls = resp.choices[0].message.tool_calls
        if calls:
            return json.loads(calls[0].function.arguments)
    except Exception as exc:
        print(f"[Orchestrator] LLM planning failed ({exc}); using heuristic plan.")
    return _heuristic_plan(question)


def build_specs(question: str, plan: dict) -> list[TaskSpec]:
    """Turn a raw plan dict into validated, structured TaskSpec objects."""
    # --- Business: trust deterministic detection over the LLM's echo ---
    bid, bname = _detect_business(question)
    if not bid and plan.get("business") and plan["business"].lower() != "all":
        key = plan["business"].strip().lower()
        bid = BUSINESS_NAME_TO_ID.get(key)
        bname = key.title() if bid else None

    # --- Metrics: validate; default if empty/invalid ---
    metrics = [m for m in plan.get("metrics", []) if m in VALID_METRICS] or DEFAULT_METRICS

    # --- Comparison intent: LLM plan OR keyword hints (either triggers it) ---
    scope = plan.get("time_scope", "all")
    if scope != "compare" and any(h in question.lower() for h in COMPARE_HINTS):
        scope = "compare"

    need_faq = plan.get("need_faq", True)
    top_k = 5

    specs: list[TaskSpec] = []

    if scope == "compare":
        specs.append(TaskSpec(agent=AgentType.COMPARISON, question=question,
                              business_id=bid, business_name=bname,
                              compare_periods=[JULY, AUGUST], top_k=top_k))
        # Also fetch current-month headline metrics for the narrative.
        specs.append(TaskSpec(agent=AgentType.DATA, question=question, metrics=metrics,
                              business_id=bid, business_name=bname, period=AUGUST, top_k=top_k))
    elif scope == "single":
        period = AUGUST if plan.get("month", "August") == "August" else JULY
        specs.append(TaskSpec(agent=AgentType.DATA, question=question, metrics=metrics,
                              business_id=bid, business_name=bname, period=period, top_k=top_k))
    else:  # "all"
        specs.append(TaskSpec(agent=AgentType.DATA, question=question, metrics=metrics,
                              business_id=bid, business_name=bname, period=None, top_k=top_k))

    if need_faq:
        specs.append(TaskSpec(agent=AgentType.RAG, question=question, top_k=3))

    return specs


# ---------------------------------------------------------------------------
# LangGraph state + nodes
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str
    specs: list                      # list[TaskSpec] the planner produced
    data: Optional[DataAgentResult]
    rag: Optional[RAGAgentResult]
    comparison: Optional[ComparisonAgentResult]
    answer: str


def _find(specs: list, agent: AgentType) -> Optional[TaskSpec]:
    """Return the first spec targeting `agent`, if the plan included one."""
    return next((s for s in specs if s.agent == agent), None)


def plan_node(state: GraphState) -> dict:
    """Node 1: build the plan + structured specs."""
    plan = _llm_plan(state["question"])
    specs = build_specs(state["question"], plan)
    return {"specs": specs}


def data_node(state: GraphState) -> dict:
    spec = _find(state["specs"], AgentType.DATA)
    return {"data": data_agent.run(spec)} if spec else {}


def rag_node(state: GraphState) -> dict:
    spec = _find(state["specs"], AgentType.RAG)
    return {"rag": rag_agent.run(spec)} if spec else {}


def comparison_node(state: GraphState) -> dict:
    spec = _find(state["specs"], AgentType.COMPARISON)
    return {"comparison": comparison_agent.run(spec)} if spec else {}


def summarize_node(state: GraphState) -> dict:
    """Node 5: synthesize the final answer from whatever evidence was gathered."""
    result = summary_agent.run(
        state["question"],
        data=state.get("data"),
        rag=state.get("rag"),
        comparison=state.get("comparison"),
    )
    return {"answer": result.answer}


def _build_graph():
    """Assemble and compile the LangGraph orchestration graph."""
    g = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("data", data_node)
    g.add_node("rag", rag_node)
    g.add_node("comparison", comparison_node)
    g.add_node("summarize", summarize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "data")
    g.add_edge("data", "rag")
    g.add_edge("rag", "comparison")
    g.add_edge("comparison", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


# Compile once at import (cheap; reused across requests).
GRAPH = _build_graph()


def answer_question(question: str) -> AskResponse:
    """Public entry point: NL question in -> full structured AskResponse out."""
    final = GRAPH.invoke({"question": question})
    return AskResponse(
        question=question,
        answer=final.get("answer", ""),
        data=final.get("data"),
        rag=final.get("rag"),
        comparison=final.get("comparison"),
        plan=final.get("specs", []),
    )


if __name__ == "__main__":
    import sys
    # Windows consoles default to cp1252, which can't print characters like the
    # arrow the model sometimes emits. Force UTF-8 so CLI runs never crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    q = " ".join(sys.argv[1:]) or "What are the top complaints for GreenLeaf Bistro this month vs last month?"
    resp = answer_question(q)
    print("Q:", resp.question)
    print("\nANSWER:\n", resp.answer)
    print("\nPLAN:", [s.agent.value for s in resp.plan])
