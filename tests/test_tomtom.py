"""
Smart City AI Agent - TomTom Tool Tests
Tests the TomTom traffic wrapper with mocked HTTP responses.

Run: pytest tests/test_tomtom.py -v
"""

from unittest.mock import patch, MagicMock
import httpx
import pytest

from app.tools.tomtom import TomTomTool, _classify_congestion, LONDON_POINTS

# ── Mock Responses ────────────────────────────────────────────────

MOCK_FLOW_FREE = {
    "flowSegmentData": {
        "currentSpeed": 55,
        "freeFlowSpeed": 58,
        "currentTravelTime": 120,
        "freeFlowTravelTime": 115,
        "confidence": 0.95,
        "roadClosure": False,
        "functionalRoadClass": "FRC2",
    }
}

MOCK_FLOW_CONGESTED = {
    "flowSegmentData": {
        "currentSpeed": 12,
        "freeFlowSpeed": 48,
        "currentTravelTime": 480,
        "freeFlowTravelTime": 120,
        "confidence": 0.88,
        "roadClosure": False,
        "functionalRoadClass": "FRC3",
    }
}

MOCK_FLOW_CLOSED = {
    "flowSegmentData": {
        "currentSpeed": 0,
        "freeFlowSpeed": 50,
        "currentTravelTime": 0,
        "freeFlowTravelTime": 100,
        "confidence": 1.0,
        "roadClosure": True,
        "functionalRoadClass": "FRC2",
    }
}

MOCK_INCIDENTS = {
    "incidents": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[-0.1278, 51.5074], [-0.1300, 51.5100]],
            },
            "properties": {
                "id": "INC-001",
                "iconCategory": 1,
                "magnitudeOfDelay": 3,
                "delay": 180,
                "from": "Oxford Street / Regent Street",
                "to": "Oxford Circus",
                "roadNumbers": ["A40"],
                "length": 500,
                "startTime": "2025-03-18T10:00:00Z",
                "endTime": "2025-03-18T16:00:00Z",
                "events": [{"description": "Multi-vehicle accident", "code": 401}],
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[-0.0922, 51.5155], [-0.0950, 51.5160]],
            },
            "properties": {
                "id": "INC-002",
                "iconCategory": 9,
                "magnitudeOfDelay": 1,
                "delay": 45,
                "from": "Moorgate / London Wall",
                "to": "Bank Junction",
                "roadNumbers": ["A501"],
                "length": 300,
                "startTime": "2025-03-17T22:00:00Z",
                "endTime": "2025-03-19T06:00:00Z",
                "events": [{"description": "Road maintenance", "code": 701}],
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [-0.0754, 51.5055],
            },
            "properties": {
                "id": "INC-003",
                "iconCategory": 6,
                "magnitudeOfDelay": 2,
                "delay": 90,
                "from": "Tower Bridge Road",
                "to": "",
                "roadNumbers": [],
                "length": 200,
                "startTime": "2025-03-18T08:00:00Z",
                "endTime": None,
                "events": [{"description": "Traffic jam", "code": 501}],
            },
        },
    ]
}

MOCK_INCIDENTS_EMPTY = {"incidents": []}


@pytest.fixture
def tomtom():
    """Create TomTomTool with a fake API key for testing."""
    with patch("app.tools.tomtom.get_settings") as mock_settings:
        settings = MagicMock()
        settings.TOMTOM_BASE_URL = "https://api.tomtom.com"
        settings.TOMTOM_API_KEY = "test-key-12345"
        settings.HTTP_TIMEOUT_SECONDS = 15
        settings.HTTP_MAX_RETRIES = 2
        mock_settings.return_value = settings
        yield TomTomTool()


@pytest.fixture
def tomtom_no_key():
    """Create TomTomTool without an API key."""
    with patch("app.tools.tomtom.get_settings") as mock_settings:
        settings = MagicMock()
        settings.TOMTOM_BASE_URL = "https://api.tomtom.com"
        settings.TOMTOM_API_KEY = ""
        settings.HTTP_TIMEOUT_SECONDS = 15
        settings.HTTP_MAX_RETRIES = 2
        mock_settings.return_value = settings
        yield TomTomTool()


# ── Congestion Classification Tests ──────────────────────────────


class TestCongestionClassification:
    """Test congestion level classification."""

    def test_free_flowing(self):
        assert _classify_congestion(0.90) == "Free flowing"

    def test_light_congestion(self):
        assert _classify_congestion(0.70) == "Light congestion"

    def test_moderate_congestion(self):
        assert _classify_congestion(0.50) == "Moderate congestion"

    def test_heavy_congestion(self):
        assert _classify_congestion(0.25) == "Heavy congestion"

    def test_severe_congestion(self):
        assert _classify_congestion(0.10) == "Severe congestion / Standstill"


# ── Traffic Flow Tests ────────────────────────────────────────────


class TestTrafficFlow:
    """Test traffic flow data fetching."""

    @patch("app.tools.base.httpx.get")
    def test_flow_free_flowing(self, mock_get, tomtom):
        """Free-flowing traffic should be classified correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FLOW_FREE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tomtom.get_traffic_flow()

        assert result.success is True
        assert result.tool_name == "tomtom"
        assert result.data["current_speed_kmh"] == 55
        assert result.data["free_flow_speed_kmh"] == 58
        assert result.data["congestion_level"] == "Free flowing"
        assert result.data["road_closure"] is False

    @patch("app.tools.base.httpx.get")
    def test_flow_congested(self, mock_get, tomtom):
        """Congested traffic should show correct speed and level."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FLOW_CONGESTED
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tomtom.get_traffic_flow()

        assert result.success is True
        assert result.data["current_speed_kmh"] == 12
        assert result.data["congestion_level"] == "Heavy congestion"
        assert result.data["delay_seconds"] == 360

    @patch("app.tools.base.httpx.get")
    def test_flow_road_closed(self, mock_get, tomtom):
        """Road closure should be flagged."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FLOW_CLOSED
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tomtom.get_traffic_flow()

        assert result.success is True
        assert result.data["road_closure"] is True
        assert result.data["congestion_level"] == "Road closed"
        assert "ROAD CLOSED" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_flow_custom_location_name(self, mock_get, tomtom):
        """Location name should appear in summary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FLOW_FREE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tomtom.get_traffic_flow(location_name="Oxford Street")

        assert "Oxford Street" in result.summary

    def test_flow_no_api_key(self, tomtom_no_key):
        """Missing API key should return clean error."""
        result = tomtom_no_key.get_traffic_flow()

        assert result.success is False
        assert "API key" in result.error


# ── Incidents Tests ───────────────────────────────────────────────


class TestTrafficIncidents:
    """Test traffic incidents fetching."""

    @patch("app.tools.base.httpx.get")
    def test_incidents_parsed(self, mock_get, tomtom):
        """Should parse all incidents with correct categories."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_INCIDENTS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tomtom.get_traffic_incidents()

        assert result.success is True
        assert result.data["total_count"] == 3
        assert result.data["by_category"]["Accident"] == 1
        assert result.data["by_category"]["Road Works"] == 1
        assert result.data["by_category"]["Jam"] == 1

    @patch("app.tools.base.httpx.get")
    def test_incidents_sorted_by_delay(self, mock_get, tomtom):
        """Summary should show highest-delay incidents first."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_INCIDENTS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tomtom.get_traffic_incidents()

        # First incident in summary should be the one with highest delay (180s)
        assert "Accident" in result.summary.split("\n")[1]

    @patch("app.tools.base.httpx.get")
    def test_incidents_empty(self, mock_get, tomtom):
        """No incidents should return clean message."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_INCIDENTS_EMPTY
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tomtom.get_traffic_incidents()

        assert result.success is True
        assert result.data["total_count"] == 0
        assert "No traffic incidents" in result.summary


# ── London Points Tests ───────────────────────────────────────────


class TestLondonPoints:
    """Test predefined London location data."""

    def test_all_points_have_required_fields(self):
        """Each point should have lat, lon, and label."""
        for key, info in LONDON_POINTS.items():
            assert "lat" in info, f"{key} missing lat"
            assert "lon" in info, f"{key} missing lon"
            assert "label" in info, f"{key} missing label"

    def test_get_available_points(self, tomtom):
        """get_available_points should return all predefined locations."""
        points = tomtom.get_available_points()
        assert "central" in points
        assert "canary_wharf" in points
        assert len(points) == len(LONDON_POINTS)
