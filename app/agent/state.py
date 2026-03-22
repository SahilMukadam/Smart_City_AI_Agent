"""
Smart City AI Agent - Agent State Definition
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
    # Full message history. `add_messages` annotation means new messages
    # are APPENDED, not replaced. This preserves conversation context.
    messages: Annotated[list, add_messages]

    # ── Tool Selection ────────────────────────────────────────────
    # Which tools the router decided to call (e.g., ["get_tube_status", "get_current_weather"])
    tools_to_call: list[str]

    # ── Tool Results ──────────────────────────────────────────────
    # Results from each tool call, keyed by tool name
    # e.g., {"get_tube_status": "...", "get_current_weather": "..."}
    tool_results: dict[str, str]

    # ── Analysis ──────────────────────────────────────────────────
    # The LLM's correlated analysis of all tool results
    analysis: str

    # ── Metadata ──────────────────────────────────────────────────
    # Number of tool-calling iterations so far (safety limit)
    iteration_count: int

    # Error message if something went wrong
    error: str
