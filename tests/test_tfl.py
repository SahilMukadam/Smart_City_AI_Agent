"""
Smart City AI Agent - TfL Tool Tests
Tests the TfL API wrapper with mocked HTTP responses.
No real API calls made during testing.

Run: pytest tests/test_tfl.py -v
"""

from unittest.mock import patch, MagicMock
import httpx
import pytest

from app.tools.tfl import TfLTool
from app.models.schemas import ToolResponse
from tests.conftest import (
    MOCK_TUBE_STATUS_GOOD,
    MOCK_TUBE_STATUS_DISRUPTED,
    MOCK_ROAD_DISRUPTIONS,
    MOCK_ROAD_STATUS,
)


@pytest.fixture
def tfl():
    """Create a TfLTool instance for testing."""
    return TfLTool()


# ── Tube Status Tests ─────────────────────────────────────────────


class TestTubeStatus:
    """Test tube status fetching and parsing."""

    @patch("app.tools.base.httpx.get")
    def test_tube_status_all_good(self, mock_get, tfl):
        """When all lines have good service, disrupted_count should be 0."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_TUBE_STATUS_GOOD
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_tube_status()

        assert result.success is True
        assert result.tool_name == "tfl"
        assert result.query_type == "tube_status"
        assert result.data["disrupted_count"] == 0
        assert result.data["total_lines"] == 3
        assert "good service" in result.summary.lower()

    @patch("app.tools.base.httpx.get")
    def test_tube_status_with_disruptions(self, mock_get, tfl):
        """When lines are disrupted, the response should list them."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_TUBE_STATUS_DISRUPTED
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_tube_status()

        assert result.success is True
        assert result.data["disrupted_count"] == 2
        assert result.data["total_lines"] == 3
        assert "Central" in result.summary
        assert "Northern" in result.summary
        assert "Minor Delays" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_tube_status_parses_lines_correctly(self, mock_get, tfl):
        """Each line should have name, status, severity, and reason."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_TUBE_STATUS_DISRUPTED
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_tube_status()
        lines = result.data["lines"]

        central = next(l for l in lines if l["name"] == "Central")
        assert central["status"] == "Minor Delays"
        assert central["severity"] == 5
        assert "Oxford Circus" in central["reason"]

    @patch("app.tools.base.httpx.get")
    def test_tube_status_timeout(self, mock_get, tfl):
        """API timeout should return error response, not crash."""
        mock_get.side_effect = httpx.TimeoutException("Connection timed out")

        result = tfl.get_tube_status()

        assert result.success is False
        assert result.error is not None
        assert "timed out" in result.error.lower()

    @patch("app.tools.base.httpx.get")
    def test_tube_status_server_error(self, mock_get, tfl):
        """5xx errors should be retried, then return error response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_get.return_value = mock_response

        result = tfl.get_tube_status()

        assert result.success is False
        # Should have retried (2 attempts)
        assert mock_get.call_count == 2


# ── Road Disruption Tests ─────────────────────────────────────────


class TestRoadDisruptions:
    """Test road disruption fetching and parsing."""

    @patch("app.tools.base.httpx.get")
    def test_disruptions_returns_all(self, mock_get, tfl):
        """Should parse all disruptions from response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_ROAD_DISRUPTIONS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_disruptions()

        assert result.success is True
        assert result.data["total_count"] == 3

    @patch("app.tools.base.httpx.get")
    def test_disruptions_severity_count(self, mock_get, tfl):
        """Should correctly count disruptions by severity."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_ROAD_DISRUPTIONS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_disruptions()
        by_severity = result.data["by_severity"]

        assert by_severity["Serious"] == 1
        assert by_severity["Moderate"] == 1
        assert by_severity["Minimal"] == 1

    @patch("app.tools.base.httpx.get")
    def test_disruptions_road_name_extraction(self, mock_get, tfl):
        """Should extract road names from corridorIds or location."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_ROAD_DISRUPTIONS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_disruptions()
        disruptions = result.data["disruptions"]

        assert disruptions[0]["road_name"] == "A1"
        assert disruptions[1]["road_name"] == "A40"

    @patch("app.tools.base.httpx.get")
    def test_disruptions_empty_response(self, mock_get, tfl):
        """No disruptions should return clean empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_disruptions()

        assert result.success is True
        assert result.data["total_count"] == 0
        assert "no road disruptions" in result.summary.lower()

    @patch("app.tools.base.httpx.get")
    def test_disruptions_summary_truncates(self, mock_get, tfl):
        """Summary should only show first 5 disruptions to keep it concise."""
        # Create 8 mock disruptions
        many_disruptions = MOCK_ROAD_DISRUPTIONS * 3  # 9 disruptions
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = many_disruptions
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_disruptions()

        assert "more" in result.summary.lower()


# ── Road Status Tests ─────────────────────────────────────────────


class TestRoadStatus:
    """Test road corridor status fetching."""

    @patch("app.tools.base.httpx.get")
    def test_road_status_all(self, mock_get, tfl):
        """Should fetch all road corridors when no IDs specified."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_ROAD_STATUS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_status()

        assert result.success is True
        assert result.data["total_count"] == 3

        # Check the URL doesn't include specific road IDs
        call_url = mock_get.call_args[0][0]
        assert "/Road" in call_url
        assert "/Status" not in call_url

    @patch("app.tools.base.httpx.get")
    def test_road_status_specific_roads(self, mock_get, tfl):
        """Should fetch specific roads when IDs are provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [MOCK_ROAD_STATUS[0]]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_status(road_ids="A1")

        assert result.success is True
        call_url = mock_get.call_args[0][0]
        assert "A1" in call_url
        assert "/Status" in call_url

    @patch("app.tools.base.httpx.get")
    def test_road_status_parses_corridors(self, mock_get, tfl):
        """Each corridor should have id, name, status, and description."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_ROAD_STATUS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_road_status()
        corridors = result.data["corridors"]

        a2 = next(c for c in corridors if c["id"] == "A2")
        assert a2["name"] == "A2 Road"
        assert a2["status"] == "Serious"


# ── Response Format Tests ─────────────────────────────────────────


class TestResponseFormat:
    """Test that all responses conform to the ToolResponse schema."""

    @patch("app.tools.base.httpx.get")
    def test_response_has_all_fields(self, mock_get, tfl):
        """Every response should have timestamp, source_url, and response_time."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_TUBE_STATUS_GOOD
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_tube_status()

        assert result.timestamp is not None
        assert result.source_url != ""
        assert result.response_time_ms >= 0

    @patch("app.tools.base.httpx.get")
    def test_to_agent_string_success(self, mock_get, tfl):
        """to_agent_string should format nicely for the LLM."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_TUBE_STATUS_GOOD
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = tfl.get_tube_status()
        agent_str = result.to_agent_string()

        assert "[tfl:tube_status]" in agent_str
        assert "Summary:" in agent_str

    @patch("app.tools.base.httpx.get")
    def test_to_agent_string_error(self, mock_get, tfl):
        """Error responses should format clearly for the LLM."""
        mock_get.side_effect = httpx.TimeoutException("Timeout")

        result = tfl.get_tube_status()
        agent_str = result.to_agent_string()

        assert "ERROR" in agent_str
