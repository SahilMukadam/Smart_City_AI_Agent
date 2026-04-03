"""
Smart City AI Agent - Reasoning Tracker Tests

Run: pytest tests/test_reasoning.py -v
"""

import pytest
from app.agent.reasoning import (
    build_reasoning_steps,
    ReasoningStep,
    NODE_CONFIG,
    EXAMPLE_QUERIES,
)


class TestReasoningSteps:

    def test_builds_steps_from_node_order(self):
        result = {
            "tools_to_call": ["get_tube_status", "get_current_weather"],
            "tool_arguments": {"get_tube_status": {}, "get_current_weather": {}},
            "source_metadata": [
                {"tool_name": "get_tube_status", "success": True, "cached": False, "response_time_ms": 200},
                {"tool_name": "get_current_weather", "success": True, "cached": True, "response_time_ms": 0},
            ],
            "parsed_insights": [{"type": "weather_traffic", "title": "Test", "description": "...", "confidence": "high"}],
            "parsed_anomalies": [],
            "health_scores": {"overall": 82},
            "tool_results": {"get_tube_status": "data", "get_current_weather": "data"},
        }
        node_order = ["router", "argument_extractor", "tool_executor", "correlator", "analyzer", "responder"]
        timings = {"router": 500, "tool_executor": 1200, "analyzer": 800}

        steps = build_reasoning_steps(result, node_order, timings)

        assert len(steps) == 6
        assert steps[0]["node"] == "router"
        assert steps[0]["emoji"] == "🔀"
        assert "get_tube_status" in steps[0]["detail"]

    def test_direct_response_path(self):
        result = {"tools_to_call": [], "tool_arguments": {}, "source_metadata": [], "parsed_insights": [], "parsed_anomalies": [], "health_scores": {}, "tool_results": {}}
        node_order = ["router", "direct_responder"]
        steps = build_reasoning_steps(result, node_order, {})

        assert len(steps) == 2
        assert steps[1]["node"] == "direct_responder"

    def test_unknown_node_skipped(self):
        result = {"tools_to_call": []}
        steps = build_reasoning_steps(result, ["unknown_node"], {})
        assert len(steps) == 0

    def test_step_has_all_fields(self):
        result = {"tools_to_call": ["get_tube_status"], "tool_arguments": {}, "source_metadata": [], "parsed_insights": [], "parsed_anomalies": [], "health_scores": {}, "tool_results": {}}
        steps = build_reasoning_steps(result, ["router"], {"router": 100})

        step = steps[0]
        assert "node" in step
        assert "label" in step
        assert "emoji" in step
        assert "detail" in step
        assert "duration_ms" in step

    def test_executor_detail_shows_cached(self):
        result = {
            "source_metadata": [
                {"cached": True, "success": True},
                {"cached": False, "success": True},
                {"cached": False, "success": False},
            ],
            "tools_to_call": [], "tool_arguments": {}, "parsed_insights": [],
            "parsed_anomalies": [], "health_scores": {}, "tool_results": {},
        }
        steps = build_reasoning_steps(result, ["tool_executor"], {})
        assert "1 cached" in steps[0]["detail"]
        assert "1 API call" in steps[0]["detail"]
        assert "1 failed" in steps[0]["detail"]

    def test_correlator_detail_shows_health(self):
        result = {
            "parsed_insights": [{"x": 1}, {"x": 2}],
            "parsed_anomalies": [{"level": "critical"}, {"level": "warning"}],
            "health_scores": {"overall": 75},
            "tools_to_call": [], "tool_arguments": {}, "source_metadata": {}, "tool_results": {},
        }
        steps = build_reasoning_steps(result, ["correlator"], {})
        assert "2 correlation" in steps[0]["detail"]
        assert "1 critical" in steps[0]["detail"]
        assert "75/100" in steps[0]["detail"]


class TestReasoningStepModel:

    def test_to_dict(self):
        step = ReasoningStep(node="router", label="Selecting", emoji="🔀", detail="Picked tools", duration_ms=123.456)
        d = step.to_dict()
        assert d["node"] == "router"
        assert d["duration_ms"] == 123.5


class TestNodeConfig:

    def test_all_nodes_configured(self):
        expected = ["router", "argument_extractor", "tool_executor", "correlator", "analyzer", "responder", "direct_responder"]
        for node in expected:
            assert node in NODE_CONFIG, f"Missing config for {node}"

    def test_all_have_required_fields(self):
        for name, config in NODE_CONFIG.items():
            assert "emoji" in config, f"{name} missing emoji"
            assert "label" in config, f"{name} missing label"
            assert "detail_fn" in config, f"{name} missing detail_fn"
            assert callable(config["detail_fn"]), f"{name} detail_fn not callable"


class TestExampleQueries:

    def test_categories_exist(self):
        assert len(EXAMPLE_QUERIES) >= 5

    def test_each_category_has_queries(self):
        for cat in EXAMPLE_QUERIES:
            assert "category" in cat
            assert "queries" in cat
            assert len(cat["queries"]) >= 2

    def test_each_query_has_text_and_description(self):
        for cat in EXAMPLE_QUERIES:
            for q in cat["queries"]:
                assert "text" in q and len(q["text"]) > 5
                assert "description" in q and len(q["description"]) > 3

    def test_no_duplicate_queries(self):
        all_texts = [q["text"] for cat in EXAMPLE_QUERIES for q in cat["queries"]]
        assert len(all_texts) == len(set(all_texts)), "Duplicate example queries found"
