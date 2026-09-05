"""
Step 2a — Structured contracts (Pydantic models).

The assessment REQUIRES that:
  * the orchestrator sends each sub-agent a *structured task spec* (not raw text)
  * each sub-agent returns *structured output* (not free-form text)

These models are those contracts. Every hand-off in the pipeline is one of the
types below, so data flowing between agents is typed, validated, and JSON-safe.

Flow of types:
    Orchestrator  --TaskSpec-->            each sub-agent
    DataAgent     --DataAgentResult-->     orchestrator
    RAGAgent      --RAGAgentResult-->      orchestrator
    Comparison    --ComparisonAgentResult->orchestrator
    Summary       --SummaryAgentResult-->  orchestrator (final answer)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """The sub-agents the orchestrator can route to."""
    DATA = "DataAgent"
    RAG = "RAGAgent"
    COMPARISON = "ComparisonAgent"
    SUMMARY = "SummaryAgent"


class Period(BaseModel):
    """A named, inclusive date window — e.g. label='this month' (Aug 2026)."""
    label: str                       # human name, e.g. "this month" / "July"
    start: str                       # ISO date "YYYY-MM-DD", inclusive
    end: str                         # ISO date "YYYY-MM-DD", inclusive


# ---------------------------------------------------------------------------
# TaskSpec: the structured instruction the orchestrator hands to a sub-agent.
# ---------------------------------------------------------------------------
class TaskSpec(BaseModel):
    """What the orchestrator sends DOWN to a sub-agent (never raw text alone)."""
    agent: AgentType                 # which sub-agent this task is for
    question: str                    # the original NL question, for context
    # Which metrics DataAgent should compute (e.g. ["csat", "top_themes"]).
    metrics: list[str] = Field(default_factory=list)
    # Optional filters that narrow the survey data.
    business_id: Optional[str] = None
    business_name: Optional[str] = None
    period: Optional[Period] = None            # single period (DataAgent)
    compare_periods: Optional[list[Period]] = None  # two periods (ComparisonAgent)
    # How many items to return for themes / RAG chunks.
    top_k: int = 5


# ---------------------------------------------------------------------------
# Metric primitives shared by several result types.
# ---------------------------------------------------------------------------
class ThemeCount(BaseModel):
    """One mined theme and how often it appears in free-text."""
    theme: str
    mentions: int
    share_pct: float                 # % of responses mentioning this theme


class MetricResult(BaseModel):
    """A bundle of computed metrics. Fields are optional — only what was asked."""
    csat: Optional[float] = None                 # % of ratings that are 4 or 5
    average_rating: Optional[float] = None       # mean 1-5 rating
    response_count: Optional[int] = None
    rating_distribution: Optional[dict[int, int]] = None  # {1: n, ..., 5: n}
    top_themes: Optional[list[ThemeCount]] = None


# ---------------------------------------------------------------------------
# Sub-agent result types (what each sub-agent returns UP to the orchestrator).
# ---------------------------------------------------------------------------
class DataAgentResult(BaseModel):
    agent: AgentType = AgentType.DATA
    scope: str                        # plain description of the filter applied
    metrics: MetricResult


class RetrievedChunk(BaseModel):
    """One chunk of the FAQ returned by the vector search."""
    text: str
    source: str                       # e.g. "product_faq.txt"
    score: float                      # similarity score (higher = more relevant)


class RAGAgentResult(BaseModel):
    agent: AgentType = AgentType.RAG
    query: str
    chunks: list[RetrievedChunk]


class PeriodMetrics(BaseModel):
    """Metrics for a single period inside a comparison."""
    period_label: str
    csat: float
    average_rating: float
    response_count: int
    top_themes: list[ThemeCount]


class ComparisonAgentResult(BaseModel):
    agent: AgentType = AgentType.COMPARISON
    current: PeriodMetrics
    previous: PeriodMetrics
    csat_change: float                # current.csat - previous.csat (pp)
    avg_rating_change: float
    notable_changes: list[str]        # human-readable deltas the agent flagged


class SummaryAgentResult(BaseModel):
    agent: AgentType = AgentType.SUMMARY
    answer: str                       # the final business-language narrative


# ---------------------------------------------------------------------------
# The end-to-end response returned by the /ask endpoint.
# ---------------------------------------------------------------------------
class AskResponse(BaseModel):
    question: str
    answer: str                       # SummaryAgent's narrative
    # The structured sub-agent outputs, exposed so a reviewer can inspect the
    # evidence behind the answer (great for the evaluation checkpoint).
    data: Optional[DataAgentResult] = None
    rag: Optional[RAGAgentResult] = None
    comparison: Optional[ComparisonAgentResult] = None
    plan: list[TaskSpec] = Field(default_factory=list)  # what the orchestrator decided
