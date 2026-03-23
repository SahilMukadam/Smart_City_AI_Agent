"""
Smart City AI Agent - Agent State Definition (Day 5)
The state object that flows through every node in the LangGraph.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class CityAgentState(TypedDict):
    """
    State that flows through the agent graph.

    Each node reads what it needs and returns only the keys it updates.
    LangGraph merges partial updates into the full state automatically.
    """

    # ── Conversation ──────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Tool Selection ────────────────────────────────────────────
    # Which tools the router decided to call
    # e.g., ["get_tube_status", "get_current_weather"]
    tools_to_call: list[str]

    # ── Tool Arguments ────────────────────────────────────────────
    # Arguments for each tool call, keyed by tool name
    # e.g., {"get_traffic_flow": {"latitude": 51.50, "longitude": -0.07}}
    # May include suffixed keys for duplicate tools:
    # e.g., {"get_traffic_flow__1": {...}, "get_traffic_flow__2": {...}}
    tool_arguments: dict[str, dict]

    # ── Tool Results ──────────────────────────────────────────────
    # Results from each tool call, keyed by tool name
    tool_results: dict[str, str]

    # ── Analysis ──────────────────────────────────────────────────
    # The LLM's correlated analysis of all tool results
    analysis: str

    # ── Metadata ──────────────────────────────────────────────────
    iteration_count: int
    error: str
