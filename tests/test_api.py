"""
Smart City AI Agent - API Endpoint Integration Tests
Tests the FastAPI endpoints using TestClient.
LLM calls are mocked, but everything else runs for real.

Run: pytest tests/test_api.py -v
"""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════
# Health Check
# ══════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "app" in data
        assert "version" in data
        assert "tools_available" in data
        assert data["status"] == "healthy"

    def test_health_lists_all_tools(self, client):
        data = client.get("/health").json()
        tools = data["tools_available"]
        assert "tfl" in tools
        assert "weather" in tools
        assert "air_quality" in tools
        assert "tomtom" in tools


# ══════════════════════════════════════════════════════════════════
# TfL Endpoints
# ══════════════════════════════════════════════════════════════════

class TestTfLEndpoints:

    @patch("app.tools.base.httpx.get")
    def test_tube_status_endpoint(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "Central", "lineStatuses": [{"statusSeverity": 10, "statusSeverityDescription": "Good Service", "reason": None}]}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/tfl/tube-status")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["tool_name"] == "tfl"
        assert "lines" in data["data"]

    @patch("app.tools.base.httpx.get")
    def test_disruptions_endpoint(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/tfl/disruptions")
        assert r.status_code == 200
        assert r.json()["data"]["total_count"] == 0

    @patch("app.tools.base.httpx.get")
    def test_road_status_with_ids(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"id": "A1", "displayName": "A1 Road", "statusSeverity": "Good", "statusSeverityDescription": "No delays"}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/tfl/road-status?road_ids=A1")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════
# Weather Endpoints
# ══════════════════════════════════════════════════════════════════

class TestWeatherEndpoints:

    @patch("app.tools.base.httpx.get")
    def test_current_weather_defaults(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": 15.0, "relative_humidity_2m": 60,
                "apparent_temperature": 13.0, "precipitation": 0,
                "weather_code": 1, "wind_speed_10m": 10,
                "wind_direction_10m": 180, "wind_gusts_10m": 20,
                "cloud_cover": 30, "pressure_msl": 1015, "visibility": 20000,
                "time": "2025-03-20T14:00",
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/weather/current")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["data"]["temperature_c"] == 15.0

    @patch("app.tools.base.httpx.get")
    def test_weather_custom_coords(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"current": {
            "temperature_2m": 20, "relative_humidity_2m": 50,
            "apparent_temperature": 18, "precipitation": 0,
            "weather_code": 0, "wind_speed_10m": 5,
            "wind_direction_10m": 90, "wind_gusts_10m": 10,
            "cloud_cover": 10, "pressure_msl": 1020, "visibility": 30000,
            "time": "2025-03-20T14:00",
        }}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/weather/current?lat=48.85&lon=2.35")
        assert r.status_code == 200

    @patch("app.tools.base.httpx.get")
    def test_forecast_endpoint(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"hourly": {
            "time": ["2025-03-20T14:00", "2025-03-20T15:00"],
            "temperature_2m": [15, 16], "precipitation_probability": [10, 20],
            "precipitation": [0, 0], "weather_code": [1, 2],
            "wind_speed_10m": [10, 12], "visibility": [20000, 18000],
        }}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/weather/forecast?hours=2")
        assert r.status_code == 200
        assert r.json()["success"] is True


# ══════════════════════════════════════════════════════════════════
# TomTom Endpoints
# ══════════════════════════════════════════════════════════════════

class TestTomTomEndpoints:

    def test_points_endpoint(self, client):
        """Points endpoint doesn't need mocking — it returns static data."""
        r = client.get("/api/tomtom/points")
        assert r.status_code == 200
        data = r.json()
        assert "central" in data
        assert "canary_wharf" in data

    @patch("app.tools.base.httpx.get")
    def test_flow_endpoint(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"flowSegmentData": {
            "currentSpeed": 40, "freeFlowSpeed": 50,
            "currentTravelTime": 120, "freeFlowTravelTime": 100,
            "confidence": 0.9, "roadClosure": False,
            "functionalRoadClass": "FRC3",
        }}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/tomtom/flow?name=Test")
        assert r.status_code == 200
        assert r.json()["data"]["current_speed_kmh"] == 40


# ══════════════════════════════════════════════════════════════════
# Session Endpoints
# ══════════════════════════════════════════════════════════════════

class TestSessionEndpoints:

    def test_create_session(self, client):
        r = client.post("/api/sessions")
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 8

    def test_list_sessions(self, client):
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_nonexistent_session(self, client):
        r = client.get("/api/sessions/nonexistent")
        assert r.status_code == 404

    def test_delete_nonexistent_session(self, client):
        r = client.delete("/api/sessions/nonexistent")
        assert r.status_code == 404

    def test_session_lifecycle(self, client):
        # Create
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]

        # Get
        get_r = client.get(f"/api/sessions/{sid}")
        assert get_r.status_code == 200
        assert get_r.json()["session_id"] == sid

        # History (empty)
        hist_r = client.get(f"/api/sessions/{sid}/history")
        assert hist_r.status_code == 200
        assert hist_r.json()["messages"] == []

        # Delete
        del_r = client.delete(f"/api/sessions/{sid}")
        assert del_r.status_code == 200

        # Gone
        gone_r = client.get(f"/api/sessions/{sid}")
        assert gone_r.status_code == 404


# ══════════════════════════════════════════════════════════════════
# Cache Endpoints
# ══════════════════════════════════════════════════════════════════

class TestCacheEndpoints:

    def test_cache_stats(self, client):
        r = client.get("/api/cache/stats")
        assert r.status_code == 200
        data = r.json()
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate_percent" in data

    def test_clear_cache(self, client):
        r = client.delete("/api/cache")
        assert r.status_code == 200
        assert "cleared" in r.json()["message"].lower()


# ══════════════════════════════════════════════════════════════════
# Example Queries Endpoint
# ══════════════════════════════════════════════════════════════════

class TestExamplesEndpoint:

    def test_returns_categories(self, client):
        r = client.get("/api/examples")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) >= 5

    def test_each_category_has_queries(self, client):
        data = client.get("/api/examples").json()
        for cat in data:
            assert "category" in cat
            assert "queries" in cat
            assert len(cat["queries"]) >= 2
            for q in cat["queries"]:
                assert "text" in q
                assert "description" in q


# ══════════════════════════════════════════════════════════════════
# Agent Chat Endpoint
# ══════════════════════════════════════════════════════════════════

class TestAgentChatEndpoint:

    @pytest.fixture(autouse=True)
    def _enable_agent(self):
        """Patch the global agent in app.main so chat endpoint works."""
        from app.agent.graph import create_agent
        with patch("app.agent.graph._get_llm") as mock_llm:
            # Default LLM mock for agent creation (compile doesn't call LLM)
            mock_llm.return_value = MagicMock()
            test_agent = create_agent()

        import app.main as main_module
        original = main_module.agent
        main_module.agent = test_agent
        yield
        main_module.agent = original

    def test_chat_validation_empty_message(self, client):
        r = client.post("/api/agent/chat", json={"message": ""})
        assert r.status_code == 422

    def test_chat_validation_too_long(self, client):
        r = client.post("/api/agent/chat", json={"message": "x" * 1001})
        assert r.status_code == 422

    @patch("app.agent.graph._get_llm")
    @patch("app.tools.base.httpx.get")
    def test_chat_full_flow(self, mock_http, mock_llm_factory, client):
        """Test a complete chat request through the agent."""
        mock_llm = MagicMock()
        call_count = [0]

        def mock_invoke(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(content="get_tube_status")
            elif call_count[0] == 2:
                return MagicMock(content="The tube is running well today.")
            else:
                return MagicMock(content="Additional response")

        mock_llm.invoke = mock_invoke
        mock_llm_factory.return_value = mock_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "Central", "lineStatuses": [{"statusSeverity": 10, "statusSeverityDescription": "Good Service", "reason": None}]}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_http.return_value = mock_resp

        r = client.post("/api/agent/chat", json={"message": "How's the tube?"})
        assert r.status_code == 200
        data = r.json()

        assert "response" in data
        assert "success" in data
        assert "session_id" in data
        assert "tools_used" in data
        assert "sources" in data
        assert "reasoning_steps" in data
        assert "total_time_ms" in data
        assert data["success"] is True
        assert len(data["session_id"]) == 8

    @patch("app.agent.graph._get_llm")
    def test_chat_greeting_skips_tools(self, mock_llm_factory, client):
        """Greeting should use direct responder — no tool calls."""
        mock_llm = MagicMock()
        call_count = [0]

        def mock_invoke(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(content="NONE")
            else:
                return MagicMock(content="Hello! I'm the Smart City AI Agent.")

        mock_llm.invoke = mock_invoke
        mock_llm_factory.return_value = mock_llm

        r = client.post("/api/agent/chat", json={"message": "Hello!"})
        assert r.status_code == 200
        data = r.json()

        assert data["success"] is True
        assert data["tools_used"] == []

    @patch("app.agent.graph._get_llm")
    def test_chat_with_session_continuity(self, mock_llm_factory, client):
        """Second message with same session_id should maintain context."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="NONE")
        mock_llm_factory.return_value = mock_llm

        r1 = client.post("/api/agent/chat", json={"message": "Hi"})
        assert r1.status_code == 200
        sid = r1.json()["session_id"]

        mock_llm.invoke.return_value = MagicMock(content="NONE")

        r2 = client.post("/api/agent/chat", json={"message": "Thanks", "session_id": sid})
        assert r2.json()["session_id"] == sid

    @patch("app.agent.graph._get_llm")
    def test_chat_response_has_timing(self, mock_llm_factory, client):
        """Response should always include timing info."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="NONE")
        mock_llm_factory.return_value = mock_llm

        r = client.post("/api/agent/chat", json={"message": "Hi"})
        assert r.status_code == 200
        assert r.json()["total_time_ms"] > 0


# ══════════════════════════════════════════════════════════════════
# City Overview Endpoint
# ══════════════════════════════════════════════════════════════════

class TestCityOverview:

    @patch("app.tools.base.httpx.get")
    def test_overview_returns_all_sources(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []  # Minimal response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r = client.get("/api/city/overview")
        assert r.status_code == 200
        data = r.json()
        assert "tube" in data
        assert "weather" in data
        assert "air_quality" in data
        assert "traffic_flow" in data
