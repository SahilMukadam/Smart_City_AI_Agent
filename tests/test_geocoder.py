"""
Smart City AI Agent - Geocoder Tests
Tests Nominatim geocoding with mocked HTTP responses.

Run: pytest tests/test_geocoder.py -v
"""

from unittest.mock import patch, MagicMock
import pytest

from app.tools.geocoder import geocode, geocode_if_needed, clear_geocode_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear geocode cache before each test."""
    clear_geocode_cache()
    yield
    clear_geocode_cache()


MOCK_NOMINATIM_RESULT = [
    {
        "lat": "51.5207763",
        "lon": "-0.1571204",
        "display_name": "Baker Street, Marylebone, City of Westminster, London, W1U 6TL, England",
        "addressdetails": {"road": "Baker Street"},
    }
]

MOCK_NOMINATIM_BRICK_LANE = [
    {
        "lat": "51.5219374",
        "lon": "-0.0717614",
        "display_name": "Brick Lane, Spitalfields, London Borough of Tower Hamlets, London, E1 6QL",
        "addressdetails": {"road": "Brick Lane"},
    }
]


class TestGeocode:

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_geocode_success(self, mock_rate, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_NOMINATIM_RESULT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = geocode("Baker Street, London")

        assert result is not None
        assert abs(result["latitude"] - 51.5208) < 0.01
        assert abs(result["longitude"] - (-0.1571)) < 0.01
        assert "Baker Street" in result["display_name"]

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_geocode_no_results(self, mock_rate, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = geocode("nonexistent_place_xyz_123")
        assert result is None

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_geocode_network_error(self, mock_rate, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        result = geocode("Baker Street")
        assert result is None

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_geocode_caches_result(self, mock_rate, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_NOMINATIM_RESULT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # First call
        result1 = geocode("Baker Street, London")
        # Second call (should hit cache)
        result2 = geocode("Baker Street, London")

        assert result1 == result2
        assert mock_get.call_count == 1  # Only one HTTP call

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_geocode_case_insensitive_cache(self, mock_rate, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_NOMINATIM_RESULT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        geocode("Baker Street")
        geocode("baker street")  # Same query, different case

        assert mock_get.call_count == 1

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_geocode_caches_none_result(self, mock_rate, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        geocode("nowhere_xyz")
        geocode("nowhere_xyz")

        assert mock_get.call_count == 1  # None result also cached

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_geocode_passes_bbox(self, mock_rate, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_NOMINATIM_RESULT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        geocode("Baker Street", constrain_to_bbox=True)

        call_params = mock_get.call_args[1].get("params", {})
        assert "viewbox" in call_params
        assert "bounded" in call_params


class TestGeocodeIfNeeded:

    @patch("app.tools.geocoder.geocode")
    def test_geocodes_unknown_location(self, mock_geocode):
        mock_geocode.return_value = {"latitude": 51.5208, "longitude": -0.1571, "display_name": "Baker Street"}

        tool_args = {
            "get_traffic_flow": {"location_name": "Baker Street"},
        }
        result = geocode_if_needed(tool_args)

        assert result["get_traffic_flow"]["latitude"] == 51.5208
        assert result["get_traffic_flow"]["longitude"] == -0.1571

    def test_skips_when_coords_already_set(self):
        tool_args = {
            "get_traffic_flow": {"latitude": 51.5055, "longitude": -0.0754, "location_name": "Tower Bridge"},
        }
        result = geocode_if_needed(tool_args)

        # Should not modify — coords already non-default
        assert result["get_traffic_flow"]["latitude"] == 51.5055

    def test_skips_tools_without_location_name(self):
        tool_args = {
            "get_tube_status": {},
            "get_current_weather": {},
        }
        result = geocode_if_needed(tool_args)

        assert result["get_tube_status"] == {}
        assert result["get_current_weather"] == {}

    @patch("app.tools.geocoder.geocode")
    def test_handles_geocode_failure(self, mock_geocode):
        mock_geocode.return_value = None

        tool_args = {
            "get_traffic_flow": {"location_name": "Unknown Place XYZ"},
        }
        result = geocode_if_needed(tool_args)

        # Should keep original args (no coordinates added)
        assert "latitude" not in result["get_traffic_flow"]

    @patch("app.tools.geocoder.geocode")
    def test_geocodes_multiple_tools(self, mock_geocode):
        mock_geocode.side_effect = [
            {"latitude": 51.52, "longitude": -0.15, "display_name": "A"},
            {"latitude": 51.50, "longitude": -0.07, "display_name": "B"},
        ]

        tool_args = {
            "get_traffic_flow__1": {"location_name": "Baker Street"},
            "get_traffic_flow__2": {"location_name": "Brick Lane"},
        }
        result = geocode_if_needed(tool_args)

        assert result["get_traffic_flow__1"]["latitude"] == 51.52
        assert result["get_traffic_flow__2"]["latitude"] == 51.50


class TestClearCache:

    @patch("app.tools.geocoder.httpx.get")
    @patch("app.tools.geocoder._enforce_rate_limit")
    def test_clear_cache_works(self, mock_rate, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_NOMINATIM_RESULT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        geocode("Baker Street")
        clear_geocode_cache()
        geocode("Baker Street")

        assert mock_get.call_count == 2  # Had to call again after cache clear
