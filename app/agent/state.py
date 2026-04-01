"""
Smart City AI Agent - Agent State Definition (Day 9)
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class CityAgentState(TypedDict):
    """State that flows through the agent graph."""

    messages: Annotated[list, add_messages]
    tools_to_call: list[str]
    tool_arguments: dict[str, dict]
    tool_results: dict[str, str]

    # ── Source Metadata ───────────────────────────────────────────
    source_metadata: list[dict]

    # ── Correlation ───────────────────────────────────────────────
    correlation_insights: str
    parsed_insights: list[dict]

    # ── Anomaly Detection (Day 9) ─────────────────────────────────
    anomaly_alerts: str          # Formatted for LLM
    parsed_anomalies: list[dict] # Structured for API response
    health_scores: dict          # City health scores per category

    # ── Analysis ──────────────────────────────────────────────────
    analysis: str

    # ── Metadata ──────────────────────────────────────────────────
    iteration_count: int
    error: str
