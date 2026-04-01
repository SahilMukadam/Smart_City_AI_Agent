"""
Smart City AI Agent - Agent Tests (Day 9)

Run: pytest tests/test_agent.py -v
"""

from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.tools import ALL_TOOLS, TOOL_MAP
from app.agent.graph import (
    router_node, argument_extractor_node, tool_executor_node,
    correlator_node, analyzer_node, responder_node,
    direct_responder_node, should_use_tools, build_agent_graph,
    _build_conversation_context,
)


def _make_state(message: str, **overrides) -> dict:
    state = {
        "messages": [MagicMock(content=message, type="human")],
        "tools_to_call": [], "tool_arguments": {}, "tool_results": {},
        "source_metadata": [],
        "correlation_insights": "", "parsed_insights": [],
        "anomaly_alerts": "", "parsed_anomalies": [], "health_scores": {},
        "analysis": "", "iteration_count": 0, "error": "",
    }
    state.update(overrides)
    return state


class TestToolDefinitions:
    def test_all_registered(self):
        assert len(ALL_TOOLS) == 9
    def test_map_complete(self):
        for n in ["get_tube_status","get_current_weather","get_traffic_flow"]:
            assert n in TOOL_MAP


class TestConversationContext:
    def test_empty(self):
        assert _build_conversation_context([HumanMessage(content="Hi")]) == ""
    def test_single_exchange(self):
        msgs = [HumanMessage(content="Q"), AIMessage(content="A"), HumanMessage(content="Q2")]
        assert "Q" in _build_conversation_context(msgs)


class TestRouterNode:
    @patch("app.agent.graph._get_llm")
    def test_selects(self, m):
        m.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="get_tube_status")))
        assert "get_tube_status" in router_node(_make_state("Tube?"))["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_none(self, m):
        m.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="NONE")))
        assert router_node(_make_state("Hi"))["tools_to_call"] == []

    @patch("app.agent.graph._get_llm")
    def test_fallback(self, m):
        m.return_value = MagicMock(invoke=MagicMock(side_effect=Exception("fail")))
        assert len(router_node(_make_state("Help"))["tools_to_call"]) > 0


class TestConditionalRouting:
    def test_to_tools(self):
        assert should_use_tools(_make_state("", tools_to_call=["t"])) == "argument_extractor"
    def test_to_direct(self):
        assert should_use_tools(_make_state("", tools_to_call=[])) == "direct_responder"


class TestArgumentExtractor:
    def test_skips_no_arg(self):
        r = argument_extractor_node(_make_state("", tools_to_call=["get_tube_status"]))
        assert r["tool_arguments"]["get_tube_status"] == {}


class TestToolExecutor:
    @patch("app.agent.graph.response_cache")
    @patch("app.agent.tools._tfl")
    def test_returns_metadata(self, mock_tfl, mock_cache):
        mock_cache.get.return_value = None
        mock_tfl.get_tube_status.return_value = MagicMock(to_agent_string=MagicMock(return_value="data"))
        state = _make_state("", tools_to_call=["get_tube_status"], tool_arguments={"get_tube_status": {}})
        r = tool_executor_node(state)
        assert len(r["source_metadata"]) == 1

    @patch("app.agent.graph.response_cache")
    def test_cache_hit(self, mock_cache):
        mock_cache.get.return_value = "cached"
        state = _make_state("", tool_arguments={"get_tube_status": {}})
        r = tool_executor_node(state)
        assert r["source_metadata"][0]["cached"] is True


class TestCorrelatorNode:
    def test_produces_anomalies(self):
        state = _make_state("", tool_results={
            "get_current_weather": "[weather:current_weather]\nSummary: Moderate rain. Temperature: 8.0°C. Humidity: 92%. Wind: 55.0 km/h. Precipitation: 6.0 mm.",
            "get_traffic_flow": "[tomtom:traffic_flow]\nSummary: Heavy congestion. Current speed: 10 km/h (free-flow: 48 km/h). Congestion ratio: 21%.",
        })
        result = correlator_node(state)
        assert len(result["parsed_anomalies"]) > 0
        assert "health_scores" in result
        assert result["health_scores"].get("overall") is not None

    def test_skips_no_data(self):
        result = correlator_node(_make_state("", tool_results={}))
        assert result["parsed_anomalies"] == []
        assert result["health_scores"] == {}


class TestAnalyzerNode:
    @patch("app.agent.graph._get_llm")
    def test_generates(self, m):
        m.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="Done.")))
        state = _make_state("Q", tool_results={"t": "d"}, anomaly_alerts="alert", health_scores={"overall": 75})
        assert analyzer_node(state)["analysis"] == "Done."


class TestResponderNode:
    def test_includes_health_score(self):
        state = _make_state("", analysis="Ok.", tool_results={"t": "d"}, health_scores={"overall": 80})
        content = responder_node(state)["messages"][0].content
        assert "80/100" in content

    def test_excludes_failed(self):
        state = _make_state("", analysis="Ok.", tool_results={"t1": "d", "t2": "ERROR: x"})
        content = responder_node(state)["messages"][0].content
        assert "t1" in content and "t2" not in content


class TestGraphStructure:
    def test_compiles(self):
        assert build_agent_graph() is not None
    def test_seven_nodes(self):
        nodes = [n for n in build_agent_graph().get_graph().nodes if not n.startswith("__")]
        assert len(nodes) == 7
