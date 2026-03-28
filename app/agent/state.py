"""
Smart City AI Agent - Agent State Definition (Day 7)
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class CityAgentState(TypedDict):
    """State that flows through the agent graph."""

    # ── Conversation ──────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Tool Selection ────────────────────────────────────────────
    tools_to_call: list[str]

    # ── Tool Arguments ────────────────────────────────────────────
    tool_arguments: dict[str, dict]

    # ── Tool Results ──────────────────────────────────────────────
    tool_results: dict[str, str]

    # ── Correlation Insights (Day 7) ──────────────────────────────
    # Pre-computed cross-source insights from the correlation engine.
    # Fed to the analyzer so the LLM can build on detected patterns.
    correlation_insights: str

    # ── Analysis ──────────────────────────────────────────────────
    analysis: str

    # ── Metadata ──────────────────────────────────────────────────
    iteration_count: int
    error: str
