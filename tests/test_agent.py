"""
Smart City AI Agent - Agent Tests (Day 6)
Tests parallel execution, conditional routing, argument extraction,
and conversation memory context.

Run: pytest tests/test_agent.py -v
"""

from unittest.mock import patch, MagicMock
import pytest

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.state import CityAgentState
from app.agent.tools import ALL_TOOLS, TOOL_MAP, get_tool_descriptions
from app.agent.graph import (
    router_node,
    argument_extractor_node,
    tool_executor_node,
    analyzer_node,
    responder_node,
    direct_responder_node,
    should_use_tools,
    build_agent_graph,
    _build_conversation_context,
)


# ── Helper ────────────────────────────────────────────────────────

def _make_state(message: str, **overrides) -> dict:
    state = {
        "messages": [MagicMock(content=message, type="human")],
        "tools_to_call": [],
        "tool_arguments": {},
        "tool_results": {},
        "analysis": "",
        "iteration_count": 0,
        "error": "",
    }
    state.update(overrides)
    return state


def _make_state_with_history(
    history: list[tuple[str, str]],
    current_question: str,
    **overrides,
) -> dict:
    """Create state with conversation history.
    history: list of (user_msg, ai_msg) tuples.
    """
    messages = []
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=current_question))

    state = {
        "messages": messages,
        "tools_to_call": [],
        "tool_arguments": {},
        "tool_results": {},
        "analysis": "",
        "iteration_count": 0,
        "error": "",
    }
    state.update(overrides)
    return state


# ── Tool Definition Tests ─────────────────────────────────────────


class TestToolDefinitions:

    def test_all_tools_registered(self):
        assert len(ALL_TOOLS) == 9

    def test_tool_map_has_all_tools(self):
        expected = [
            "get_tube_status", "get_road_disruptions", "get_road_corridor_status",
            "get_current_weather", "get_weather_forecast", "get_air_quality",
            "get_traffic_flow", "get_london_traffic_overview", "get_traffic_incidents",
        ]
        for name in expected:
            assert name in TOOL_MAP

    def test_all_tools_have_descriptions(self):
        for tool in ALL_TOOLS:
            assert tool.description
            assert len(tool.description) > 20

    def test_get_tool_descriptions_format(self):
        desc = get_tool_descriptions()
        assert "get_tube_status" in desc
        assert "get_traffic_flow" in desc


# ── Conversation Context Tests ────────────────────────────────────


class TestConversationContext:
    """Test the conversation context builder."""

    def test_empty_history(self):
        messages = [HumanMessage(content="Hello")]
        context = _build_conversation_context(messages)
        assert context == ""

    def test_single_exchange(self):
        messages = [
            HumanMessage(content="How's the tube?"),
            AIMessage(content="All lines have good service."),
            HumanMessage(content="What about traffic?"),
        ]
        context = _build_conversation_context(messages)
        assert "How's the tube?" in context
        assert "All lines have good service." in context
        # Current question should NOT be in context
        assert "What about traffic?" not in context

    def test_truncates_long_ai_messages(self):
        long_response = "A" * 500
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content=long_response),
            HumanMessage(content="Q2"),
        ]
        context = _build_conversation_context(messages)
        assert "..." in context
        assert len(context) < len(long_response)

    def test_limits_to_recent_exchanges(self):
        messages = []
        for i in range(10):
            messages.append(HumanMessage(content=f"Question {i}"))
            messages.append(AIMessage(content=f"Answer {i}"))
        messages.append(HumanMessage(content="Current question"))

        context = _build_conversation_context(messages)
        # Should only include last 3 exchanges (6 messages)
        assert "Question 9" in context
        assert "Question 0" not in context

    def test_skips_non_langchain_messages(self):
        """Tuples from initial invoke should be skipped."""
        messages = [("user", "hello"), MagicMock(content="World", type="human")]
        context = _build_conversation_context(messages)
        # Only the MagicMock should be processed, but it's the current msg
        assert context == ""


# ── Router Node Tests ─────────────────────────────────────────────


class TestRouterNode:

    @patch("app.agent.graph._get_llm")
    def test_router_selects_tools(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="get_tube_status,get_current_weather")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("How's the tube and weather?"))
        assert "get_tube_status" in result["tools_to_call"]
        assert "get_current_weather" in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_router_returns_none_for_greeting(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="NONE")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("Hello!"))
        assert result["tools_to_call"] == []

    @patch("app.agent.graph._get_llm")
    def test_router_filters_invalid_tools(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="get_tube_status,fake_tool")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("Tubes"))
        assert "get_tube_status" in result["tools_to_call"]
        assert "fake_tool" not in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_router_fallback_on_garbage(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Sure, let me check weather")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("Weather?"))
        assert len(result["tools_to_call"]) > 0

    @patch("app.agent.graph._get_llm")
    def test_router_handles_llm_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("Help"))
        assert len(result["tools_to_call"]) > 0
        assert "error" in result

    @patch("app.agent.graph._get_llm")
    def test_router_with_history_context(self, mock_get_llm):
        """Router should receive conversation history in its prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="get_traffic_flow")
        mock_get_llm.return_value = mock_llm

        state = _make_state_with_history(
            history=[("How's traffic in Camden?", "Camden has moderate congestion.")],
            current_question="What about Canary Wharf?",
        )
        result = router_node(state)

        # Verify the LLM was called with context that includes history
        call_args = mock_llm.invoke.call_args[0][0][0].content
        assert "Camden" in call_args


# ── Conditional Routing Tests ─────────────────────────────────────


class TestConditionalRouting:

    def test_routes_to_tools_when_tools_selected(self):
        state = _make_state("Traffic?", tools_to_call=["get_traffic_flow"])
        assert should_use_tools(state) == "argument_extractor"

    def test_routes_to_direct_when_no_tools(self):
        state = _make_state("Hello!", tools_to_call=[])
        assert should_use_tools(state) == "direct_responder"


# ── Argument Extractor Tests ──────────────────────────────────────


class TestArgumentExtractor:

    def test_skips_llm_for_no_arg_tools(self):
        state = _make_state(
            "Tube?",
            tools_to_call=["get_tube_status", "get_road_disruptions"],
        )
        result = argument_extractor_node(state)
        assert result["tool_arguments"]["get_tube_status"] == {}

    @patch("app.agent.graph._get_llm")
    def test_extracts_location_args(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"get_traffic_flow": {"latitude": 51.5055, "longitude": -0.0754, "location_name": "Tower Bridge"}}'
        )
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            "Traffic near Tower Bridge?",
            tools_to_call=["get_traffic_flow"],
        )
        result = argument_extractor_node(state)
        assert result["tool_arguments"]["get_traffic_flow"]["latitude"] == 51.5055

    @patch("app.agent.graph._get_llm")
    def test_fallback_on_parse_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not json")
        mock_get_llm.return_value = mock_llm

        state = _make_state("Traffic?", tools_to_call=["get_traffic_flow"])
        result = argument_extractor_node(state)
        assert result["tool_arguments"]["get_traffic_flow"] == {}


# ── Tool Executor Tests ───────────────────────────────────────────


class TestToolExecutor:

    @patch("app.agent.tools._tfl")
    def test_executor_calls_tool(self, mock_tfl):
        mock_result = MagicMock()
        mock_result.to_agent_string.return_value = "[tfl] data"
        mock_tfl.get_tube_status.return_value = mock_result

        state = _make_state(
            "Tube?",
            tools_to_call=["get_tube_status"],
            tool_arguments={"get_tube_status": {}},
        )
        result = tool_executor_node(state)
        assert "get_tube_status" in result["tool_results"]

    def test_executor_handles_unknown_tool(self):
        state = _make_state(
            "Test",
            tools_to_call=["nonexistent"],
            tool_arguments={"nonexistent": {}},
        )
        result = tool_executor_node(state)
        assert "ERROR" in result["tool_results"]["nonexistent"]

    def test_executor_empty_tools(self):
        state = _make_state("Test", tools_to_call=[], tool_arguments={})
        result = tool_executor_node(state)
        assert result["tool_results"] == {}

    def test_executor_handles_suffixed_keys(self):
        state = _make_state(
            "Compare",
            tools_to_call=["get_traffic_flow"],
            tool_arguments={"get_traffic_flow__1": {}, "get_traffic_flow__2": {}},
        )
        result = tool_executor_node(state)
        assert "get_traffic_flow__1" in result["tool_results"]
        assert "get_traffic_flow__2" in result["tool_results"]

    @patch("app.agent.tools._tfl")
    @patch("app.agent.tools._weather")
    def test_executor_runs_multiple_tools(self, mock_weather, mock_tfl):
        mock_tfl_result = MagicMock()
        mock_tfl_result.to_agent_string.return_value = "[tfl] data"
        mock_tfl.get_tube_status.return_value = mock_tfl_result

        mock_weather_result = MagicMock()
        mock_weather_result.to_agent_string.return_value = "[weather] data"
        mock_weather.get_current_weather.return_value = mock_weather_result

        state = _make_state(
            "Both?",
            tools_to_call=["get_tube_status", "get_current_weather"],
            tool_arguments={"get_tube_status": {}, "get_current_weather": {}},
        )
        result = tool_executor_node(state)
        assert "get_tube_status" in result["tool_results"]
        assert "get_current_weather" in result["tool_results"]


# ── Analyzer Node Tests ───────────────────────────────────────────


class TestAnalyzerNode:

    @patch("app.agent.graph._get_llm")
    def test_analyzer_generates_analysis(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Traffic is flowing well.")
        mock_get_llm.return_value = mock_llm

        state = _make_state("Traffic?", tool_results={"get_traffic_flow": "[tomtom] 45km/h"})
        result = analyzer_node(state)
        assert result["analysis"] == "Traffic is flowing well."

    @patch("app.agent.graph._get_llm")
    def test_analyzer_fallback_on_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Rate limited")
        mock_get_llm.return_value = mock_llm

        state = _make_state("Overview", tool_results={"get_tube_status": "All good"})
        result = analyzer_node(state)
        assert "All good" in result["analysis"]


# ── Responder Node Tests ──────────────────────────────────────────


class TestResponderNode:

    def test_responder_creates_ai_message(self):
        state = _make_state(
            "Test",
            analysis="London traffic is moderate.",
            tools_to_call=["get_traffic_flow"],
            tool_results={"get_traffic_flow": "data..."},
        )
        result = responder_node(state)
        assert "London traffic is moderate." in result["messages"][0].content

    def test_responder_includes_data_sources(self):
        state = _make_state(
            "Test",
            analysis="Analysis.",
            tools_to_call=["get_tube_status"],
            tool_results={"get_tube_status": "Good service"},
        )
        result = responder_node(state)
        assert "Data sources" in result["messages"][0].content

    def test_responder_excludes_failed_tools(self):
        state = _make_state(
            "Test",
            analysis="Partial.",
            tools_to_call=["get_tube_status", "get_air_quality"],
            tool_results={
                "get_tube_status": "Good",
                "get_air_quality": "ERROR: Failed",
            },
        )
        result = responder_node(state)
        content = result["messages"][0].content
        assert "get_tube_status" in content
        assert "get_air_quality" not in content


# ── Direct Responder Tests ────────────────────────────────────────


class TestDirectResponder:

    @patch("app.agent.graph._get_llm")
    def test_direct_responder_greeting(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Hello! I'm the Smart City Agent.")
        mock_get_llm.return_value = mock_llm

        result = direct_responder_node(_make_state("Hi!"))
        assert "Hello" in result["messages"][0].content

    @patch("app.agent.graph._get_llm")
    def test_direct_responder_handles_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Down")
        mock_get_llm.return_value = mock_llm

        result = direct_responder_node(_make_state("Hey"))
        assert "Smart City AI Agent" in result["messages"][0].content


# ── Graph Structure Tests ─────────────────────────────────────────


class TestGraphStructure:

    def test_graph_compiles(self):
        graph = build_agent_graph()
        assert graph is not None

    def test_graph_has_correct_nodes(self):
        graph = build_agent_graph()
        graph_repr = graph.get_graph()
        node_ids = list(graph_repr.nodes)
        for name in ["router", "argument_extractor", "tool_executor", "analyzer", "responder", "direct_responder"]:
            assert name in node_ids

    def test_graph_has_six_nodes(self):
        graph = build_agent_graph()
        graph_repr = graph.get_graph()
        real_nodes = [n for n in graph_repr.nodes if not n.startswith("__")]
        assert len(real_nodes) == 6
