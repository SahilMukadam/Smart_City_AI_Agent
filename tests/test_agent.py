"""
Smart City AI Agent - Agent Tests (Day 5)
Tests parallel execution, conditional routing, argument extraction.
All LLM calls mocked — 0 Gemini API usage.

Run: pytest tests/test_agent.py -v
"""

from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import pytest

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
)


# ── Helper ────────────────────────────────────────────────────────

def _make_state(message: str, **overrides) -> dict:
    """Create a minimal state with a user message."""
    state = {
        "messages": [MagicMock(content=message)],
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
    """Test that all tools are properly defined and registered."""

    def test_all_tools_registered(self):
        assert len(ALL_TOOLS) == 9

    def test_tool_map_has_all_tools(self):
        expected = [
            "get_tube_status", "get_road_disruptions", "get_road_corridor_status",
            "get_current_weather", "get_weather_forecast", "get_air_quality",
            "get_traffic_flow", "get_london_traffic_overview", "get_traffic_incidents",
        ]
        for name in expected:
            assert name in TOOL_MAP, f"Missing tool: {name}"

    def test_all_tools_have_descriptions(self):
        for tool in ALL_TOOLS:
            assert tool.description
            assert len(tool.description) > 20

    def test_get_tool_descriptions_format(self):
        desc = get_tool_descriptions()
        assert "get_tube_status" in desc
        assert "get_traffic_flow" in desc


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
        mock_llm.invoke.return_value = MagicMock(content="get_tube_status,fake_tool,get_air_quality")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("Tubes and air"))
        assert "get_tube_status" in result["tools_to_call"]
        assert "get_air_quality" in result["tools_to_call"]
        assert "fake_tool" not in result["tools_to_call"]

    @patch("app.agent.graph._get_llm")
    def test_router_fallback_on_garbage(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="I think you should check weather")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("Weather?"))
        assert len(result["tools_to_call"]) > 0

    @patch("app.agent.graph._get_llm")
    def test_router_handles_llm_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API quota exceeded")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("What's happening?"))
        assert len(result["tools_to_call"]) > 0
        assert "error" in result

    @patch("app.agent.graph._get_llm")
    def test_router_increments_iteration(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="get_tube_status")
        mock_get_llm.return_value = mock_llm

        result = router_node(_make_state("Tube?", iteration_count=0))
        assert result["iteration_count"] == 1


# ── Conditional Routing Tests ─────────────────────────────────────


class TestConditionalRouting:
    """Test the should_use_tools conditional edge."""

    def test_routes_to_tools_when_tools_selected(self):
        state = _make_state("Traffic?", tools_to_call=["get_traffic_flow"])
        assert should_use_tools(state) == "argument_extractor"

    def test_routes_to_direct_when_no_tools(self):
        state = _make_state("Hello!", tools_to_call=[])
        assert should_use_tools(state) == "direct_responder"


# ── Argument Extractor Tests ──────────────────────────────────────


class TestArgumentExtractor:
    """Test argument extraction for tool calls."""

    def test_skips_llm_for_no_arg_tools(self):
        """Tools that need no args should skip the LLM call entirely."""
        state = _make_state(
            "Tube status?",
            tools_to_call=["get_tube_status", "get_road_disruptions"],
        )
        result = argument_extractor_node(state)
        assert "get_tube_status" in result["tool_arguments"]
        assert result["tool_arguments"]["get_tube_status"] == {}

    @patch("app.agent.graph._get_llm")
    def test_extracts_location_args(self, mock_get_llm):
        """Should extract coordinates for location-specific tools."""
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

        args = result["tool_arguments"]["get_traffic_flow"]
        assert args["latitude"] == 51.5055
        assert args["location_name"] == "Tower Bridge"

    @patch("app.agent.graph._get_llm")
    def test_fallback_on_parse_error(self, mock_get_llm):
        """Bad JSON from LLM should fall back to empty args."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not valid json at all")
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            "Traffic?",
            tools_to_call=["get_traffic_flow"],
        )
        result = argument_extractor_node(state)

        # Should have fallback empty args
        assert "get_traffic_flow" in result["tool_arguments"]
        assert result["tool_arguments"]["get_traffic_flow"] == {}


# ── Parallel Tool Executor Tests ──────────────────────────────────


class TestToolExecutor:
    """Test the parallel tool executor."""

    @patch("app.agent.tools._tfl")
    def test_executor_calls_tool(self, mock_tfl):
        mock_result = MagicMock()
        mock_result.to_agent_string.return_value = "[tfl:tube_status] All good"
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
        """Tools with __1, __2 suffixes should resolve to base tool name."""
        state = _make_state(
            "Compare",
            tools_to_call=["get_traffic_flow"],
            tool_arguments={
                "get_traffic_flow__1": {},
                "get_traffic_flow__2": {},
            },
        )
        # This will fail because the real tools aren't mocked,
        # but the keys should be attempted correctly
        result = tool_executor_node(state)
        assert "get_traffic_flow__1" in result["tool_results"]
        assert "get_traffic_flow__2" in result["tool_results"]

    @patch("app.agent.tools._tfl")
    @patch("app.agent.tools._weather")
    def test_executor_runs_multiple_tools(self, mock_weather, mock_tfl):
        """Multiple tools should all be called and results collected."""
        mock_tfl_result = MagicMock()
        mock_tfl_result.to_agent_string.return_value = "[tfl] data"
        mock_tfl.get_tube_status.return_value = mock_tfl_result

        mock_weather_result = MagicMock()
        mock_weather_result.to_agent_string.return_value = "[weather] data"
        mock_weather.get_current_weather.return_value = mock_weather_result

        state = _make_state(
            "Tube and weather?",
            tools_to_call=["get_tube_status", "get_current_weather"],
            tool_arguments={
                "get_tube_status": {},
                "get_current_weather": {},
            },
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

        state = _make_state(
            "How's traffic?",
            tool_results={"get_traffic_flow": "[tomtom] 45km/h"},
        )
        result = analyzer_node(state)
        assert result["analysis"] == "Traffic is flowing well."

    @patch("app.agent.graph._get_llm")
    def test_analyzer_fallback_on_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Rate limited")
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            "Overview",
            tool_results={"get_tube_status": "All lines good"},
        )
        result = analyzer_node(state)
        assert "All lines good" in result["analysis"]
        assert "error" in result


# ── Responder Node Tests ──────────────────────────────────────────


class TestResponderNode:

    def test_responder_creates_ai_message(self):
        state = _make_state(
            "Test",
            analysis="London traffic is moderate today.",
            tools_to_call=["get_traffic_flow"],
            tool_results={"get_traffic_flow": "data..."},
        )
        result = responder_node(state)
        assert "London traffic is moderate today." in result["messages"][0].content

    def test_responder_includes_data_sources(self):
        state = _make_state(
            "Test",
            analysis="Analysis here.",
            tools_to_call=["get_tube_status"],
            tool_results={"get_tube_status": "Good service"},
        )
        result = responder_node(state)
        assert "Data sources" in result["messages"][0].content

    def test_responder_excludes_failed_tools(self):
        state = _make_state(
            "Test",
            analysis="Partial analysis.",
            tools_to_call=["get_tube_status", "get_air_quality"],
            tool_results={
                "get_tube_status": "Good service",
                "get_air_quality": "ERROR: Connection failed",
            },
        )
        result = responder_node(state)
        content = result["messages"][0].content
        assert "get_tube_status" in content
        assert "get_air_quality" not in content


# ── Direct Responder Tests ────────────────────────────────────────


class TestDirectResponder:
    """Test the direct responder for non-tool queries."""

    @patch("app.agent.graph._get_llm")
    def test_direct_responder_greeting(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Hello! I'm the Smart City Agent. How can I help?"
        )
        mock_get_llm.return_value = mock_llm

        state = _make_state("Hi there!")
        result = direct_responder_node(state)

        assert len(result["messages"]) == 1
        assert "Hello" in result["messages"][0].content

    @patch("app.agent.graph._get_llm")
    def test_direct_responder_handles_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM down")
        mock_get_llm.return_value = mock_llm

        state = _make_state("Hey")
        result = direct_responder_node(state)

        # Should return fallback message
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
        assert "router" in node_ids
        assert "argument_extractor" in node_ids
        assert "tool_executor" in node_ids
        assert "analyzer" in node_ids
        assert "responder" in node_ids
        assert "direct_responder" in node_ids

    def test_graph_has_six_nodes(self):
        """Day 5 graph should have 6 nodes (+ __start__, __end__)."""
        graph = build_agent_graph()
        graph_repr = graph.get_graph()
        # Filter out __start__ and __end__
        real_nodes = [n for n in graph_repr.nodes if not n.startswith("__")]
        assert len(real_nodes) == 6
