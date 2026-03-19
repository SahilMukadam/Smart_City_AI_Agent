"""
Smart City AI Agent - Air Quality Tool Tests
Tests the OpenAQ air quality wrapper with mocked HTTP responses.

Run: pytest tests/test_air_quality.py -v
"""

from unittest.mock import patch, MagicMock, call
import httpx
import pytest

from app.tools.air_quality import AirQualityTool, PM25_AQI_LEVELS

# ── Mock Responses ────────────────────────────────────────────────

MOCK_STATIONS = {
    "results": [
        {
            "id": 12345,
            "name": "London Westminster",
            "locality": "Westminster",
            "coordinates": {"latitude": 51.4946, "longitude": -0.1315},
            "isMonitor": True,
            "parameters": [
                {"parameter": "pm25"},
                {"parameter": "pm10"},
                {"parameter": "no2"},
            ],
            "datetimeLast": {"utc": "2025-03-18T13:00:00Z"},
        },
        {
            "id": 12346,
            "name": "London Marylebone",
            "locality": "Marylebone",
            "coordinates": {"latitude": 51.5225, "longitude": -0.1546},
            "isMonitor": True,
            "parameters": [
                {"parameter": "pm25"},
                {"parameter": "no2"},
                {"parameter": "o3"},
            ],
            "datetimeLast": {"utc": "2025-03-18T12:30:00Z"},
        },
    ]
}

MOCK_STATIONS_EMPTY = {"results": []}

MOCK_STATION_READINGS = {
    "results": [
        {
            "parameter": {"name": "pm25", "displayName": "PM2.5", "units": "µg/m³"},
            "value": 8.5,
            "period": {"datetimeTo": {"utc": "2025-03-18T13:00:00Z"}},
        },
        {
            "parameter": {"name": "pm10", "displayName": "PM10", "units": "µg/m³"},
            "value": 22.3,
            "period": {"datetimeTo": {"utc": "2025-03-18T13:00:00Z"}},
        },
        {
            "parameter": {"name": "no2", "displayName": "NO₂", "units": "µg/m³"},
            "value": 35.1,
            "period": {"datetimeTo": {"utc": "2025-03-18T13:00:00Z"}},
        },
    ]
}

MOCK_STATION_READINGS_HIGH_PM25 = {
    "results": [
        {
            "parameter": {"name": "pm25", "displayName": "PM2.5", "units": "µg/m³"},
            "value": 42.0,
            "period": {"datetimeTo": {"utc": "2025-03-18T13:00:00Z"}},
        },
    ]
}


@pytest.fixture
def aq():
    return AirQualityTool()


# ── Nearby Stations Tests ─────────────────────────────────────────


class TestNearbyStations:
    """Test station discovery."""

    @patch("app.tools.base.httpx.get")
    def test_finds_stations(self, mock_get, aq):
        """Should find and parse nearby stations."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_STATIONS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = aq.get_nearby_stations()

        assert result.success is True
        assert result.data["total_found"] == 2
        assert result.data["stations"][0]["name"] == "London Westminster"
        assert "pm25" in result.data["stations"][0]["parameters"]

    @patch("app.tools.base.httpx.get")
    def test_no_stations_found(self, mock_get, aq):
        """Empty results should return clean response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_STATIONS_EMPTY
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = aq.get_nearby_stations()

        assert result.success is True
        assert result.data["total_found"] == 0
        assert "No air quality" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_stations_summary_lists_names(self, mock_get, aq):
        """Summary should list station names."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_STATIONS
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = aq.get_nearby_stations()

        assert "London Westminster" in result.summary
        assert "London Marylebone" in result.summary


# ── Latest Readings Tests ─────────────────────────────────────────


class TestLatestReadings:
    """Test fetching and aggregating latest readings."""

    @patch("app.tools.base.httpx.get")
    def test_readings_success(self, mock_get, aq):
        """Should fetch stations then readings for each."""
        # First call: get stations; subsequent calls: get readings per station
        mock_stations_resp = MagicMock()
        mock_stations_resp.status_code = 200
        mock_stations_resp.json.return_value = MOCK_STATIONS
        mock_stations_resp.raise_for_status = MagicMock()

        mock_readings_resp = MagicMock()
        mock_readings_resp.status_code = 200
        mock_readings_resp.json.return_value = MOCK_STATION_READINGS
        mock_readings_resp.raise_for_status = MagicMock()

        # Stations call first, then readings for each station
        mock_get.side_effect = [
            mock_stations_resp,
            mock_readings_resp,
            mock_readings_resp,
        ]

        result = aq.get_latest_readings()

        assert result.success is True
        assert result.data["stations_checked"] == 2
        assert "pm25" in result.data["readings"]

    @patch("app.tools.base.httpx.get")
    def test_readings_aqi_good(self, mock_get, aq):
        """PM2.5 of 8.5 should be 'Good' AQI."""
        mock_stations_resp = MagicMock()
        mock_stations_resp.status_code = 200
        mock_stations_resp.json.return_value = MOCK_STATIONS
        mock_stations_resp.raise_for_status = MagicMock()

        mock_readings_resp = MagicMock()
        mock_readings_resp.status_code = 200
        mock_readings_resp.json.return_value = MOCK_STATION_READINGS
        mock_readings_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [
            mock_stations_resp,
            mock_readings_resp,
            mock_readings_resp,
        ]

        result = aq.get_latest_readings()

        aqi = result.data["aqi_category"]
        assert aqi is not None
        assert aqi["category"] == "Good"

    @patch("app.tools.base.httpx.get")
    def test_readings_aqi_unhealthy(self, mock_get, aq):
        """PM2.5 of 42.0 should be 'Unhealthy for Sensitive Groups'."""
        mock_stations_resp = MagicMock()
        mock_stations_resp.status_code = 200
        # Return only one station
        one_station = {"results": [MOCK_STATIONS["results"][0]]}
        mock_stations_resp.json.return_value = one_station
        mock_stations_resp.raise_for_status = MagicMock()

        mock_readings_resp = MagicMock()
        mock_readings_resp.status_code = 200
        mock_readings_resp.json.return_value = MOCK_STATION_READINGS_HIGH_PM25
        mock_readings_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_stations_resp, mock_readings_resp]

        result = aq.get_latest_readings()

        aqi = result.data["aqi_category"]
        assert aqi is not None
        assert aqi["category"] == "Unhealthy for Sensitive Groups"

    @patch("app.tools.base.httpx.get")
    def test_readings_no_stations(self, mock_get, aq):
        """No stations should return clean response, not crash."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_STATIONS_EMPTY
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = aq.get_latest_readings()

        assert result.success is True
        assert result.data["stations_checked"] == 0
        assert "No monitoring stations" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_readings_station_failure_graceful(self, mock_get, aq):
        """If one station's readings fail, should still return others."""
        mock_stations_resp = MagicMock()
        mock_stations_resp.status_code = 200
        mock_stations_resp.json.return_value = MOCK_STATIONS
        mock_stations_resp.raise_for_status = MagicMock()

        mock_readings_resp = MagicMock()
        mock_readings_resp.status_code = 200
        mock_readings_resp.json.return_value = MOCK_STATION_READINGS
        mock_readings_resp.raise_for_status = MagicMock()

        # First station succeeds, second times out
        mock_get.side_effect = [
            mock_stations_resp,
            mock_readings_resp,
            httpx.TimeoutException("Timeout"),
        ]

        result = aq.get_latest_readings()

        # Should still succeed with partial data
        assert result.success is True
        assert "pm25" in result.data["readings"]


# ── AQI Category Tests ────────────────────────────────────────────


class TestAQICategories:
    """Test AQI classification logic."""

    def test_aqi_good(self, aq):
        aqi = aq._get_aqi_category(5.0)
        assert aqi["category"] == "Good"

    def test_aqi_moderate(self, aq):
        aqi = aq._get_aqi_category(20.0)
        assert aqi["category"] == "Moderate"

    def test_aqi_hazardous(self, aq):
        aqi = aq._get_aqi_category(300.0)
        assert aqi["category"] == "Hazardous"

    def test_aqi_boundary_12(self, aq):
        """12.0 should be Good (upper boundary)."""
        aqi = aq._get_aqi_category(12.0)
        assert aqi["category"] == "Good"
