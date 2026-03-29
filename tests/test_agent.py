"""
Smart City AI Agent - Agent Tests (Day 8)
Tests include caching, structured output, and updated state.

Run: pytest tests/test_agent.py -v
"""

from unittest.mock import patch, MagicMock
import pytest

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.tools import ALL_TOOLS, TOOL_MAP
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
        "source_metadata": [],
        "correlation_insights": "",
        "parsed_insights": [],
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
        assert "Q1" in ctx and "Q2" not in ctx


class TestRouterNode:
    @patch("app.agent.graph._get_llm")
    def test_selects_tools(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="get_tube_status")))
        result = router_node(_make_state("Tube?"))
        assert "get_tube_status" in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_none_for_greeting(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="NONE")))
        assert router_node(_make_state("Hello"))["tools_to_call"] == []

    @patch("app.agent.graph._get_llm")
    def test_filters_invalid(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="get_tube_status,fake")))
        result = router_node(_make_state("Tubes"))
        assert "fake" not in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_fallback_on_error(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(side_effect=Exception("fail")))
        result = router_node(_make_state("Help"))
        assert len(result["tools_to_call"]) > 0


class TestConditionalRouting:
    def test_to_tools(self):
        assert should_use_tools(_make_state("", tools_to_call=["t"])) == "argument_extractor"

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
    @patch("app.agent.graph.response_cache")
    @patch("app.agent.tools._tfl")
    def test_returns_source_metadata(self, mock_tfl, mock_cache):
        mock_cache.get.return_value = None
        mock_tfl.get_tube_status.return_value = MagicMock(to_agent_string=MagicMock(return_value="data"))

        state = _make_state("", tools_to_call=["get_tube_status"], tool_arguments={"get_tube_status": {}})
        result = tool_executor_node(state)

        assert "source_metadata" in result
        assert len(result["source_metadata"]) == 1
        assert result["source_metadata"][0]["tool_name"] == "get_tube_status"

    @patch("app.agent.graph.response_cache")
    def test_uses_cache_hit(self, mock_cache):
        mock_cache.get.return_value = "cached tube data"

        state = _make_state("", tools_to_call=["get_tube_status"], tool_arguments={"get_tube_status": {}})
        result = tool_executor_node(state)

        assert result["tool_results"]["get_tube_status"] == "cached tube data"
        meta = result["source_metadata"][0]
        assert meta["cached"] is True
        assert meta["success"] is True

    def test_unknown_tool_metadata(self):
        state = _make_state("", tool_arguments={"fake": {}})
        result = tool_executor_node(state)
        meta = result["source_metadata"][0]
        assert meta["success"] is False
        assert "Unknown" in meta["error"]


class TestCorrelatorNode:
    def test_produces_parsed_insights(self):
        state = _make_state("", tool_results={
            "get_current_weather": "[weather:current_weather]\nSummary: Moderate rain. Temperature: 8.0°C. Humidity: 92%. Wind: 22.0 km/h. Precipitation: 3.2 mm.",
            "get_traffic_flow": "[tomtom:traffic_flow]\nSummary: Heavy congestion. Current speed: 12 km/h (free-flow: 48 km/h). Congestion ratio: 25%.",
        })
        result = correlator_node(state)

        assert result["correlation_insights"] != ""
        assert len(result["parsed_insights"]) > 0
        assert "title" in result["parsed_insights"][0]
        assert "confidence" in result["parsed_insights"][0]

    def test_skips_single_source(self):
        state = _make_state("", tool_results={"t": "data"})
        result = correlator_node(state)
        assert result["parsed_insights"] == []


class TestAnalyzerNode:
    @patch("app.agent.graph._get_llm")
    def test_generates_analysis(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="Done.")))
        state = _make_state("Q", tool_results={"t": "d"}, correlation_insights="insight")
        assert analyzer_node(state)["analysis"] == "Done."


class TestResponderNode:
    def test_creates_message(self):
        state = _make_state("", analysis="Good.", tool_results={"t": "d"})
        result = responder_node(state)
        assert "Good." in result["messages"][0].content

    def test_excludes_failed_sources(self):
        state = _make_state("", analysis="Ok.", tool_results={"t1": "d", "t2": "ERROR: fail"})
        content = responder_node(state)["messages"][0].content
        assert "t1" in content and "t2" not in content


class TestDirectResponder:
    @patch("app.agent.graph._get_llm")
    def test_responds(self, mock):
        mock.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="Hi!")))
        assert "Hi" in direct_responder_node(_make_state("Hello"))["messages"][0].content


class TestGraphStructure:
    def test_compiles(self):
        assert build_agent_graph() is not None

    def test_has_seven_nodes(self):
        nodes = [n for n in build_agent_graph().get_graph().nodes if not n.startswith("__")]
        assert len(nodes) == 7

    def test_has_correlator(self):
        assert "correlator" in list(build_agent_graph().get_graph().nodes)
