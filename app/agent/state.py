"""
Smart City AI Agent - Agent State Definition (Day 8)
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

    # ── Source Metadata (Day 8) ───────────────────────────────────
    # Per-tool timing, cache hits, and error info for structured output
    source_metadata: list[dict]

    # ── Correlation ───────────────────────────────────────────────
    correlation_insights: str
    parsed_insights: list[dict]  # Structured insights for API response

    # ── Analysis ──────────────────────────────────────────────────
    analysis: str

    # ── Metadata ──────────────────────────────────────────────────
    iteration_count: int
    error: str
