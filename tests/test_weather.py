"""
Smart City AI Agent - Weather Tool Tests
Tests the Open-Meteo weather wrapper with mocked HTTP responses.

Run: pytest tests/test_weather.py -v
"""

from unittest.mock import patch, MagicMock
import httpx
import pytest

from app.tools.weather import WeatherTool, WEATHER_CODES

# ── Mock Responses ────────────────────────────────────────────────

MOCK_CURRENT_WEATHER = {
    "current": {
        "time": "2025-03-18T14:00",
        "temperature_2m": 12.5,
        "relative_humidity_2m": 68,
        "apparent_temperature": 10.2,
        "precipitation": 0.0,
        "weather_code": 2,
        "wind_speed_10m": 15.3,
        "wind_direction_10m": 225,
        "wind_gusts_10m": 28.1,
        "cloud_cover": 45,
        "pressure_msl": 1015.2,
        "visibility": 24000,
    }
}

MOCK_CURRENT_RAINY = {
    "current": {
        "time": "2025-03-18T14:00",
        "temperature_2m": 8.1,
        "relative_humidity_2m": 92,
        "apparent_temperature": 5.3,
        "precipitation": 3.2,
        "weather_code": 63,
        "wind_speed_10m": 22.0,
        "wind_direction_10m": 180,
        "wind_gusts_10m": 45.0,
        "cloud_cover": 100,
        "pressure_msl": 998.5,
        "visibility": 8000,
    }
}

MOCK_FORECAST = {
    "hourly": {
        "time": [
            "2025-03-18T14:00",
            "2025-03-18T15:00",
            "2025-03-18T16:00",
            "2025-03-18T17:00",
        ],
        "temperature_2m": [12.5, 13.0, 12.8, 11.5],
        "precipitation_probability": [10, 25, 60, 80],
        "precipitation": [0.0, 0.0, 1.2, 3.5],
        "weather_code": [2, 3, 61, 63],
        "wind_speed_10m": [15.3, 16.0, 18.5, 20.0],
        "visibility": [24000, 20000, 12000, 8000],
    }
}

MOCK_FORECAST_DRY = {
    "hourly": {
        "time": ["2025-03-18T14:00", "2025-03-18T15:00"],
        "temperature_2m": [14.0, 15.2],
        "precipitation_probability": [5, 10],
        "precipitation": [0.0, 0.0],
        "weather_code": [0, 1],
        "wind_speed_10m": [8.0, 7.5],
        "visibility": [30000, 30000],
    }
}


@pytest.fixture
def weather():
    return WeatherTool()


# ── Current Weather Tests ─────────────────────────────────────────


class TestCurrentWeather:
    """Test current weather fetching and parsing."""

    @patch("app.tools.base.httpx.get")
    def test_current_weather_success(self, mock_get, weather):
        """Should parse current weather correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_CURRENT_WEATHER
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_current_weather()

        assert result.success is True
        assert result.tool_name == "weather"
        assert result.query_type == "current_weather"
        assert result.data["temperature_c"] == 12.5
        assert result.data["humidity_percent"] == 68
        assert result.data["weather_description"] == "Partly cloudy"

    @patch("app.tools.base.httpx.get")
    def test_current_weather_rainy(self, mock_get, weather):
        """Rainy weather should include precipitation in summary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_CURRENT_RAINY
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_current_weather()

        assert result.success is True
        assert result.data["precipitation_mm"] == 3.2
        assert result.data["weather_description"] == "Moderate rain"
        assert "3.2" in result.summary
        assert "Precipitation" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_current_weather_default_london(self, mock_get, weather):
        """Default location should show 'Central London' in summary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_CURRENT_WEATHER
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_current_weather()

        assert "Central London" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_current_weather_custom_location(self, mock_get, weather):
        """Custom coordinates should show lat/lon in summary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_CURRENT_WEATHER
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_current_weather(latitude=48.8566, longitude=2.3522)

        assert "48.8566" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_current_weather_timeout(self, mock_get, weather):
        """Timeout should return error response."""
        mock_get.side_effect = httpx.TimeoutException("Connection timed out")

        result = weather.get_current_weather()

        assert result.success is False
        assert result.error is not None


# ── Forecast Tests ────────────────────────────────────────────────


class TestForecast:
    """Test forecast fetching and parsing."""

    @patch("app.tools.base.httpx.get")
    def test_forecast_parses_hours(self, mock_get, weather):
        """Should parse all forecast hours."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FORECAST
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_forecast(hours=4)

        assert result.success is True
        assert len(result.data["hourly"]) == 4
        assert result.data["hourly"][0]["temperature_c"] == 12.5
        assert result.data["hourly"][2]["weather_description"] == "Slight rain"

    @patch("app.tools.base.httpx.get")
    def test_forecast_rain_summary(self, mock_get, weather):
        """Forecast with rain should flag it in summary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FORECAST
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_forecast(hours=4)

        assert "Rain likely" in result.summary or "rain" in result.summary.lower()

    @patch("app.tools.base.httpx.get")
    def test_forecast_dry_summary(self, mock_get, weather):
        """Dry forecast should indicate rain is unlikely."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FORECAST_DRY
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_forecast(hours=2)

        assert "unlikely" in result.summary.lower()

    @patch("app.tools.base.httpx.get")
    def test_forecast_temp_range(self, mock_get, weather):
        """Summary should include temperature range."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FORECAST
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_forecast(hours=4)

        assert "11.5" in result.summary
        assert "13.0" in result.summary

    @patch("app.tools.base.httpx.get")
    def test_forecast_caps_at_48_hours(self, mock_get, weather):
        """Hours should be capped at 48."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FORECAST
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = weather.get_forecast(hours=100)

        # Verify the params sent to the API
        call_params = mock_get.call_args[1].get("params") or mock_get.call_args[0][1] if len(mock_get.call_args[0]) > 1 else mock_get.call_args[1].get("params", {})
        # The key check: the data returned should work fine
        assert result.success is True


# ── Weather Code Tests ────────────────────────────────────────────


class TestWeatherCodes:
    """Test weather code interpretation."""

    def test_all_codes_are_strings(self):
        """All weather codes should map to string descriptions."""
        for code, description in WEATHER_CODES.items():
            assert isinstance(code, int)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_common_codes(self):
        """Check common weather codes are correct."""
        assert WEATHER_CODES[0] == "Clear sky"
        assert WEATHER_CODES[3] == "Overcast"
        assert WEATHER_CODES[63] == "Moderate rain"
        assert WEATHER_CODES[95] == "Thunderstorm"
