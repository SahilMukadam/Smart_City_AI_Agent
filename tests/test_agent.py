"""
Smart City AI Agent - Agent Tests (Day 7)
Tests now include correlator node and updated graph structure.

Run: pytest tests/test_agent.py -v
"""

from unittest.mock import patch, MagicMock
import pytest

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.tools import ALL_TOOLS, TOOL_MAP, get_tool_descriptions
from app.agent.graph import (
    router_node,
    argument_extractor_node,
    tool_executor_node,
    correlator_node,
    analyzer_node,
    responder_node,
    direct_responder_node,
    should_use_tools,
    build_agent_graph,
    _build_conversation_context,
)


def _make_state(message: str, **overrides) -> dict:
    state = {
        "messages": [MagicMock(content=message, type="human")],
        "tools_to_call": [],
        "tool_arguments": {},
        "tool_results": {},
        "correlation_insights": "",
        "analysis": "",
        "iteration_count": 0,
        "error": "",
    }
    state.update(overrides)
    return state


def _make_state_with_history(history, current_question, **overrides):
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
        "correlation_insights": "",
        "analysis": "",
        "iteration_count": 0,
        "error": "",
    }
    state.update(overrides)
    return state


class TestToolDefinitions:
    def test_all_tools_registered(self):
        assert len(ALL_TOOLS) == 9

    def test_tool_map_complete(self):
        for name in ["get_tube_status", "get_road_disruptions", "get_road_corridor_status",
                      "get_current_weather", "get_weather_forecast", "get_air_quality",
                      "get_traffic_flow", "get_london_traffic_overview", "get_traffic_incidents"]:
            assert name in TOOL_MAP


class TestConversationContext:
    def test_empty_history(self):
        assert _build_conversation_context([HumanMessage(content="Hi")]) == ""

    def test_single_exchange(self):
        msgs = [HumanMessage(content="Q1"), AIMessage(content="A1"), HumanMessage(content="Q2")]
        ctx = _build_conversation_context(msgs)
        assert "Q1" in ctx
        assert "Q2" not in ctx


class TestRouterNode:
    @patch("app.agent.graph._get_llm")
    def test_selects_tools(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="get_tube_status")))
        result = router_node(_make_state("Tube?"))
        assert "get_tube_status" in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_none_for_greeting(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="NONE")))
        result = router_node(_make_state("Hello"))
        assert result["tools_to_call"] == []

    @patch("app.agent.graph._get_llm")
    def test_filters_invalid(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="get_tube_status,fake")))
        result = router_node(_make_state("Tubes"))
        assert "fake" not in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_fallback_on_error(self, mock):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("fail")
        mock.return_value = llm
        result = router_node(_make_state("Help"))
        assert len(result["tools_to_call"]) > 0


class TestConditionalRouting:
    def test_to_tools(self):
        assert should_use_tools(_make_state("", tools_to_call=["get_tube_status"])) == "argument_extractor"

    def test_to_direct(self):
        assert should_use_tools(_make_state("", tools_to_call=[])) == "direct_responder"


class TestArgumentExtractor:
    def test_skips_llm_no_arg_tools(self):
        result = argument_extractor_node(_make_state("", tools_to_call=["get_tube_status"]))
        assert result["tool_arguments"]["get_tube_status"] == {}

    @patch("app.agent.graph._get_llm")
    def test_extracts_args(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(
            return_value=MagicMock(content='{"get_traffic_flow": {"latitude": 51.50}}')
        ))
        result = argument_extractor_node(_make_state("Traffic?", tools_to_call=["get_traffic_flow"]))
        assert result["tool_arguments"]["get_traffic_flow"]["latitude"] == 51.50


class TestToolExecutor:
    @patch("app.agent.tools._tfl")
    def test_calls_tool(self, mock_tfl):
        mock_tfl.get_tube_status.return_value = MagicMock(to_agent_string=MagicMock(return_value="data"))
        state = _make_state("", tools_to_call=["get_tube_status"], tool_arguments={"get_tube_status": {}})
        result = tool_executor_node(state)
        assert "get_tube_status" in result["tool_results"]

    def test_unknown_tool(self):
        state = _make_state("", tool_arguments={"fake": {}})
        result = tool_executor_node(state)
        assert "ERROR" in result["tool_results"]["fake"]


class TestCorrelatorNode:
    """Test the new correlator node."""

    def test_produces_insights_with_multiple_sources(self):
        state = _make_state("", tool_results={
            "get_current_weather": (
                "[weather:current_weather]\nSummary: Moderate rain. "
                "Temperature: 8.0°C. Humidity: 92%. Wind: 22.0 km/h. Precipitation: 3.2 mm."
            ),
            "get_traffic_flow": (
                "[tomtom:traffic_flow]\nSummary: Heavy congestion. "
                "Current speed: 12 km/h (free-flow: 48 km/h). Congestion ratio: 25%."
            ),
        })
        result = correlator_node(state)
        assert result["correlation_insights"] != ""
        assert "CORRELATION" in result["correlation_insights"]

    def test_skips_with_single_source(self):
        state = _make_state("", tool_results={"get_tube_status": "All good"})
        result = correlator_node(state)
        assert result["correlation_insights"] == ""

    def test_skips_with_all_errors(self):
        state = _make_state("", tool_results={
            "get_weather": "ERROR: timeout",
            "get_traffic": "ERROR: 500",
        })
        result = correlator_node(state)
        assert result["correlation_insights"] == ""


class TestAnalyzerNode:
    @patch("app.agent.graph._get_llm")
    def test_generates_analysis(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="Analysis done.")))
        state = _make_state("Traffic?", tool_results={"t": "data"}, correlation_insights="insights here")
        result = analyzer_node(state)
        assert result["analysis"] == "Analysis done."

    @patch("app.agent.graph._get_llm")
    def test_fallback_includes_insights(self, mock):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("fail")
        mock.return_value = llm
        state = _make_state("", tool_results={"t": "data"}, correlation_insights="Rain + congestion")
        result = analyzer_node(state)
        assert "Rain + congestion" in result["analysis"]


class TestResponderNode:
    def test_creates_message(self):
        state = _make_state("", analysis="Good.", tool_results={"t": "data"})
        result = responder_node(state)
        assert "Good." in result["messages"][0].content

    def test_excludes_failed_sources(self):
        state = _make_state("", analysis="Ok.", tool_results={"t1": "data", "t2": "ERROR: fail"})
        result = responder_node(state)
        assert "t1" in result["messages"][0].content
        assert "t2" not in result["messages"][0].content


class TestDirectResponder:
    @patch("app.agent.graph._get_llm")
    def test_responds(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="Hi there!")))
        result = direct_responder_node(_make_state("Hello"))
        assert "Hi" in result["messages"][0].content


class TestGraphStructure:
    def test_compiles(self):
        assert build_agent_graph() is not None

    def test_has_seven_nodes(self):
        graph = build_agent_graph()
        nodes = [n for n in graph.get_graph().nodes if not n.startswith("__")]
        assert len(nodes) == 7

    def test_has_correlator(self):
        graph = build_agent_graph()
        nodes = list(graph.get_graph().nodes)
        assert "correlator" in nodes
