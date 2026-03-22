"""
Smart City AI Agent - Agent Tests
Tests the LangGraph agent components with mocked LLM responses.
No real Gemini API calls — preserves your 250 req/day quota.

Run: pytest tests/test_agent.py -v
"""

from unittest.mock import patch, MagicMock
from app.agent import graph
import pytest

from app.agent.state import CityAgentState
from app.agent.tools import ALL_TOOLS, TOOL_MAP, get_tool_descriptions
from app.agent.graph import (
    router_node,
    tool_executor_node,
    analyzer_node,
    responder_node,
    build_agent_graph,
)


# ── Tool Definition Tests ─────────────────────────────────────────


class TestToolDefinitions:
    """Test that all tools are properly defined and registered."""

    def test_all_tools_registered(self):
        """All 9 tools should be in the registry."""
        assert len(ALL_TOOLS) == 9

    def test_tool_map_has_all_tools(self):
        """TOOL_MAP should map every tool name to its function."""
        expected_names = [
            "get_tube_status",
            "get_road_disruptions",
            "get_road_corridor_status",
            "get_current_weather",
            "get_weather_forecast",
            "get_air_quality",
            "get_traffic_flow",
            "get_london_traffic_overview",
            "get_traffic_incidents",
        ]
        for name in expected_names:
            assert name in TOOL_MAP, f"Missing tool: {name}"

    def test_all_tools_have_descriptions(self):
        """Every tool should have a non-empty description."""
        for tool in ALL_TOOLS:
            assert tool.description, f"{tool.name} has no description"
            assert len(tool.description) > 20, f"{tool.name} description too short"

    def test_get_tool_descriptions_format(self):
        """get_tool_descriptions should return a formatted string."""
        desc = get_tool_descriptions()
        assert "get_tube_status" in desc
        assert "get_current_weather" in desc
        assert "get_traffic_flow" in desc


# ── Router Node Tests ─────────────────────────────────────────────


class TestRouterNode:
    """Test the router node's tool selection logic."""

    def _make_state(self, message: str) -> CityAgentState:
        """Create a minimal state with a user message."""
        return {
            "messages": [MagicMock(content=message)],
            "tools_to_call": [],
            "tool_results": {},
            "analysis": "",
            "iteration_count": 0,
            "error": "",
        }

    @patch("app.agent.graph._get_llm")
    def test_router_selects_tools(self, mock_get_llm):
        """Router should return valid tool names."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "get_tube_status,get_current_weather"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self._make_state("How's the tube and weather?")
        result = router_node(state)

        assert "get_tube_status" in result["tools_to_call"]
        assert "get_current_weather" in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_router_filters_invalid_tools(self, mock_get_llm):
        """Router should filter out tool names that don't exist."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "get_tube_status,nonexistent_tool,get_air_quality"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self._make_state("Tell me about tubes and air")
        result = router_node(state)

        assert "get_tube_status" in result["tools_to_call"]
        assert "get_air_quality" in result["tools_to_call"]
        assert "nonexistent_tool" not in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_router_fallback_on_garbage(self, mock_get_llm):
        """If LLM returns garbage, router should use fallback tools."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I think you should use the weather tool"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self._make_state("What's the weather?")
        result = router_node(state)

        # Should fall back to defaults
        assert len(result["tools_to_call"]) > 0

    @patch("app.agent.graph._get_llm")
    def test_router_handles_llm_error(self, mock_get_llm):
        """Router should handle LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API quota exceeded")
        mock_get_llm.return_value = mock_llm

        state = self._make_state("What's happening?")
        result = router_node(state)

        # Should return fallback tools, not crash
        assert len(result["tools_to_call"]) > 0
        assert "error" in result

    @patch("app.agent.graph._get_llm")
    def test_router_increments_iteration(self, mock_get_llm):
        """Router should increment iteration count."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "get_tube_status"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self._make_state("Tube status?")
        state["iteration_count"] = 0
        result = router_node(state)

        assert result["iteration_count"] == 1


# ── Tool Executor Tests ───────────────────────────────────────────


class TestToolExecutor:
    """Test the tool executor node."""

    @patch("app.agent.tools._tfl")
    def test_executor_calls_tools(self, mock_tfl):
        """Executor should call each requested tool."""
        mock_result = MagicMock()
        mock_result.to_agent_string.return_value = "[tfl:tube_status] All good"
        mock_tfl.get_tube_status.return_value = mock_result

        state = {
            "tools_to_call": ["get_tube_status"],
            "tool_results": {},
        }
        result = tool_executor_node(state)

        assert "get_tube_status" in result["tool_results"]

    def test_executor_handles_unknown_tool(self):
        """Unknown tool name should produce an error, not crash."""
        state = {
            "tools_to_call": ["nonexistent_tool"],
            "tool_results": {},
        }
        result = tool_executor_node(state)

        assert "ERROR" in result["tool_results"]["nonexistent_tool"]

    def test_executor_empty_tools(self):
        """No tools to call should return empty results."""
        state = {
            "tools_to_call": [],
            "tool_results": {},
        }
        result = tool_executor_node(state)

        assert result["tool_results"] == {}


# ── Analyzer Node Tests ───────────────────────────────────────────


class TestAnalyzerNode:
    """Test the analyzer node."""

    @patch("app.agent.graph._get_llm")
    def test_analyzer_generates_analysis(self, mock_get_llm):
        """Analyzer should produce an analysis string."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Traffic is flowing well with clear skies."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [MagicMock(content="How's traffic?")],
            "tool_results": {
                "get_traffic_flow": "[tomtom:traffic_flow] Free flowing at 45km/h",
                "get_current_weather": "[weather:current] Clear sky, 15°C",
            },
            "analysis": "",
        }
        result = analyzer_node(state)

        assert result["analysis"] == "Traffic is flowing well with clear skies."

    @patch("app.agent.graph._get_llm")
    def test_analyzer_fallback_on_error(self, mock_get_llm):
        """If LLM fails, analyzer should return raw tool data."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Rate limited")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [MagicMock(content="Overview")],
            "tool_results": {"get_tube_status": "All lines good"},
            "analysis": "",
        }
        result = analyzer_node(state)

        assert "All lines good" in result["analysis"]
        assert "error" in result


# ── Responder Node Tests ──────────────────────────────────────────


class TestResponderNode:
    """Test the responder node."""

    def test_responder_creates_ai_message(self):
        """Responder should create an AIMessage with the analysis."""
        state = {
            "analysis": "London traffic is moderate today.",
            "tools_to_call": ["get_traffic_flow", "get_current_weather"],
            "tool_results": {
                "get_traffic_flow": "data...",
                "get_current_weather": "data...",
            },
        }
        result = responder_node(state)

        assert len(result["messages"]) == 1
        assert "London traffic is moderate today." in result["messages"][0].content

    def test_responder_includes_data_sources(self):
        """Response should list which tools provided data."""
        state = {
            "analysis": "Analysis here.",
            "tools_to_call": ["get_tube_status"],
            "tool_results": {"get_tube_status": "Good service on all lines"},
        }
        result = responder_node(state)

        assert "Data sources" in result["messages"][0].content

    def test_responder_excludes_failed_tools(self):
        """Failed tools should not appear in data sources."""
        state = {
            "analysis": "Partial analysis.",
            "tools_to_call": ["get_tube_status", "get_air_quality"],
            "tool_results": {
                "get_tube_status": "Good service",
                "get_air_quality": "ERROR: Connection failed",
            },
        }
        result = responder_node(state)
        content = result["messages"][0].content

        assert "get_tube_status" in content
        assert "get_air_quality" not in content


# ── Graph Structure Tests ─────────────────────────────────────────


class TestGraphStructure:
    """Test the graph builds correctly."""

    def test_graph_compiles(self):
        """Graph should compile without errors."""
        graph = build_agent_graph()
        assert graph is not None

    def test_graph_has_correct_nodes(self):
        """Graph should have all 4 nodes."""
        graph = build_agent_graph()
        # LangGraph compiled graph exposes nodes via .get_graph()
        graph_repr = graph.get_graph()
        node_ids = list(graph_repr.nodes)
        assert "router" in node_ids
        assert "tool_executor" in node_ids
        assert "analyzer" in node_ids
        assert "responder" in node_ids
