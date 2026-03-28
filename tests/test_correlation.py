"""
Smart City AI Agent - Correlation Engine Tests
Tests the data extraction and correlation logic.
No API calls, no LLM calls — pure data analysis.

Run: pytest tests/test_correlation.py -v
"""

import pytest

from app.agent.correlation import (
    CorrelationEngine,
    correlate_data,
    format_insights_for_llm,
    Insight,
    InsightType,
    Confidence,
)


@pytest.fixture
def engine():
    return CorrelationEngine()


# ── Sample Tool Results ───────────────────────────────────────────

WEATHER_CLEAR = (
    "[weather:current_weather]\n"
    "Time: 2025-03-19 14:00 UTC\n"
    "Summary: Current weather in Central London: Clear sky. "
    "Temperature: 15.0°C (feels like 12.5°C). "
    "Humidity: 45%. Wind: 8.0 km/h."
)

WEATHER_RAINY = (
    "[weather:current_weather]\n"
    "Time: 2025-03-19 14:00 UTC\n"
    "Summary: Current weather in Central London: Moderate rain. "
    "Temperature: 8.0°C (feels like 5.0°C). "
    "Humidity: 92%. Wind: 22.0 km/h. Precipitation: 3.2 mm."
)

WEATHER_FOGGY = (
    "[weather:current_weather]\n"
    "Summary: Current weather in Central London: Foggy. "
    "Temperature: 6.0°C (feels like 3.0°C). "
    "Humidity: 98%. Wind: 3.0 km/h."
)

WEATHER_WINDY = (
    "[weather:current_weather]\n"
    "Summary: Current weather: Clear sky. "
    "Temperature: 12.0°C (feels like 7.0°C). "
    "Humidity: 50%. Wind: 55.0 km/h."
)

TRAFFIC_CONGESTED = (
    "[tomtom:traffic_flow]\n"
    "Time: 2025-03-19 14:00 UTC\n"
    "Summary: Traffic near Central London: Heavy congestion. "
    "Current speed: 12 km/h (free-flow: 48 km/h). "
    "Congestion ratio: 25%."
)

TRAFFIC_FREE = (
    "[tomtom:traffic_flow]\n"
    "Time: 2025-03-19 14:00 UTC\n"
    "Summary: Traffic near Central London: Free flowing. "
    "Current speed: 45 km/h (free-flow: 50 km/h). "
    "Congestion ratio: 90%."
)

TRAFFIC_OVERVIEW_MIXED = (
    "[tomtom:multi_point_flow]\n"
    "Summary: Traffic flow at 3 London locations (worst congestion first):\n"
    "  - Camden: Heavy congestion (12 km/h, 25% of free-flow)\n"
    "  - City of London: Light congestion (35 km/h, 70% of free-flow)\n"
    "  - Canary Wharf: Free flowing (48 km/h, 92% of free-flow)"
)

TRAFFIC_INCIDENTS = (
    "[tomtom:traffic_incidents]\n"
    "Summary: 8 traffic incidents in the area (3 Accident, 4 Road Works, 1 Jam)."
)

DISRUPTIONS_HIGH = (
    "[tfl:road_disruptions]\n"
    "Summary: 15 road disruptions currently active (5 Serious, 7 Moderate, 3 Minimal)."
)

AQ_GOOD = (
    "[air_quality:latest_readings]\n"
    "Summary: Air quality near Central London:\n"
    "Overall: Good — Air quality is satisfactory.\n"
    "  - PM2.5 (Fine Particles): 8.5 µg/m³\n"
    "  - NO₂ (Nitrogen Dioxide): 22.0 µg/m³"
)

AQ_MODERATE_HIGH_NO2 = (
    "[air_quality:latest_readings]\n"
    "Summary: Air quality near Central London:\n"
    "Overall: Moderate — Acceptable. Sensitive groups may experience minor effects.\n"
    "  - PM2.5 (Fine Particles): 18.0 µg/m³\n"
    "  - NO₂ (Nitrogen Dioxide): 45.0 µg/m³\n"
    "  - PM10 (Coarse Particles): 30.0 µg/m³"
)

AQ_UNHEALTHY = (
    "[air_quality:latest_readings]\n"
    "Summary: Air quality near Central London:\n"
    "Overall: Unhealthy for Sensitive Groups — Sensitive groups should reduce outdoor activity.\n"
    "  - PM2.5 (Fine Particles): 42.0 µg/m³\n"
    "  - NO₂ (Nitrogen Dioxide): 55.0 µg/m³"
)

TUBE_GOOD = (
    "[tfl:tube_status]\n"
    "Summary: All 11 tube lines have good service."
)

TUBE_DISRUPTED = (
    "[tfl:tube_status]\n"
    "Summary: 4 of 11 tube lines have disruptions:\n"
    "  - Central: Minor Delays\n"
    "  - Northern: Part Closure\n"
    "  - Victoria: Severe Delays\n"
    "  - Jubilee: Minor Delays"
)


# ── Data Extraction Tests ─────────────────────────────────────────


class TestWeatherExtraction:

    def test_extract_clear_weather(self, engine):
        data = engine._extract_weather_data({"get_current_weather": WEATHER_CLEAR})
        assert data["temperature_c"] == 15.0
        assert data["humidity_percent"] == 45
        assert data["wind_speed_kmh"] == 8.0
        assert data["precipitation_mm"] == 0.0
        assert data["is_raining"] is False

    def test_extract_rainy_weather(self, engine):
        data = engine._extract_weather_data({"get_current_weather": WEATHER_RAINY})
        assert data["temperature_c"] == 8.0
        assert data["precipitation_mm"] == 3.2
        assert data["is_raining"] is True

    def test_extract_foggy_weather(self, engine):
        data = engine._extract_weather_data({"get_current_weather": WEATHER_FOGGY})
        assert data["is_foggy"] is True
        assert data["has_poor_visibility"] is True

    def test_no_weather_data(self, engine):
        data = engine._extract_weather_data({"get_tube_status": "tube data"})
        assert data is None


class TestTrafficExtraction:

    def test_extract_congested_flow(self, engine):
        data = engine._extract_traffic_data({"get_traffic_flow": TRAFFIC_CONGESTED})
        assert data["has_congestion"] is True
        assert len(data["points"]) == 1
        assert data["points"][0]["current_speed"] == 12

    def test_extract_free_flow(self, engine):
        data = engine._extract_traffic_data({"get_traffic_flow": TRAFFIC_FREE})
        assert data["has_congestion"] is False

    def test_extract_overview(self, engine):
        data = engine._extract_traffic_data({"get_london_traffic_overview": TRAFFIC_OVERVIEW_MIXED})
        assert len(data["points"]) >= 2
        assert data["has_congestion"] is True

    def test_extract_incidents(self, engine):
        data = engine._extract_traffic_data({"get_traffic_incidents": TRAFFIC_INCIDENTS})
        assert data["incident_count"] == 8

    def test_extract_disruptions(self, engine):
        data = engine._extract_traffic_data({"get_road_disruptions": DISRUPTIONS_HIGH})
        assert data["disruption_count"] == 15


class TestAirQualityExtraction:

    def test_extract_good_aq(self, engine):
        data = engine._extract_air_quality_data({"get_air_quality": AQ_GOOD})
        assert data["pm25"] == 8.5
        assert data["no2"] == 22.0
        assert data["quality_level"] == "good"

    def test_extract_moderate_aq(self, engine):
        data = engine._extract_air_quality_data({"get_air_quality": AQ_MODERATE_HIGH_NO2})
        assert data["pm25"] == 18.0
        assert data["no2"] == 45.0
        assert data["quality_level"] == "moderate"

    def test_extract_unhealthy_aq(self, engine):
        data = engine._extract_air_quality_data({"get_air_quality": AQ_UNHEALTHY})
        assert data["quality_level"] == "unhealthy_sensitive"


class TestTubeExtraction:

    def test_extract_good_tube(self, engine):
        data = engine._extract_tube_data({"get_tube_status": TUBE_GOOD})
        assert data["all_clear"] is True

    def test_extract_disrupted_tube(self, engine):
        data = engine._extract_tube_data({"get_tube_status": TUBE_DISRUPTED})
        assert data["all_clear"] is False
        assert data["disrupted_count"] == 4


# ── Correlation Tests ─────────────────────────────────────────────


class TestWeatherTrafficCorrelation:

    def test_rain_plus_congestion(self, engine):
        weather = {"is_raining": True, "precipitation_mm": 3.2, "has_poor_visibility": False, "wind_speed_kmh": 15}
        traffic = {"has_congestion": True, "points": []}
        insights = engine._correlate_weather_traffic(weather, traffic)

        rain_insight = [i for i in insights if "Rain" in i.title and "contributing" in i.title]
        assert len(rain_insight) == 1
        assert rain_insight[0].confidence == Confidence.HIGH

    def test_rain_no_congestion(self, engine):
        weather = {"is_raining": True, "precipitation_mm": 1.0, "has_poor_visibility": False, "wind_speed_kmh": 10}
        traffic = {"has_congestion": False, "points": []}
        insights = engine._correlate_weather_traffic(weather, traffic)

        holding_insight = [i for i in insights if "holding up" in i.title]
        assert len(holding_insight) == 1

    def test_dry_congestion(self, engine):
        weather = {"is_raining": False, "precipitation_mm": 0, "has_poor_visibility": False, "wind_speed_kmh": 10}
        traffic = {"has_congestion": True, "points": []}
        insights = engine._correlate_weather_traffic(weather, traffic)

        not_weather = [i for i in insights if "not weather" in i.title.lower()]
        assert len(not_weather) == 1

    def test_high_wind(self, engine):
        weather = {"is_raining": False, "has_poor_visibility": False, "wind_speed_kmh": 55}
        traffic = {"has_congestion": False, "points": []}
        insights = engine._correlate_weather_traffic(weather, traffic)

        wind_insight = [i for i in insights if "wind" in i.title.lower()]
        assert len(wind_insight) == 1


class TestTrafficAirQualityCorrelation:

    def test_congestion_poor_aq(self, engine):
        traffic = {"has_congestion": True, "points": []}
        aq = {"quality_level": "unhealthy_sensitive", "no2": 55.0}
        insights = engine._correlate_traffic_air_quality(traffic, aq)

        congestion_aq = [i for i in insights if "worsening" in i.title.lower()]
        assert len(congestion_aq) == 1
        assert congestion_aq[0].confidence == Confidence.HIGH

    def test_congestion_good_aq(self, engine):
        traffic = {"has_congestion": True, "points": []}
        aq = {"quality_level": "good", "no2": 15.0}
        insights = engine._correlate_traffic_air_quality(traffic, aq)

        holding = [i for i in insights if "holding" in i.title.lower()]
        assert len(holding) == 1

    def test_high_no2(self, engine):
        traffic = {"has_congestion": False, "points": []}
        aq = {"quality_level": "moderate", "no2": 45.0}
        insights = engine._correlate_traffic_air_quality(traffic, aq)

        no2_insight = [i for i in insights if "NO₂" in i.title]
        assert len(no2_insight) == 1


class TestCongestionPatterns:

    def test_worst_congestion_identified(self, engine):
        traffic = {
            "has_congestion": True,
            "points": [
                {"location": "Camden", "congestion_ratio": 0.25, "current_speed": 12},
                {"location": "City", "congestion_ratio": 0.70, "current_speed": 35},
            ],
        }
        insights = engine._analyze_congestion_patterns(traffic)

        worst = [i for i in insights if "Worst" in i.title]
        assert len(worst) == 1
        assert "Camden" in worst[0].title

    def test_all_clear(self, engine):
        traffic = {
            "has_congestion": False,
            "points": [
                {"location": "Central", "congestion_ratio": 0.90, "current_speed": 45},
                {"location": "Canary Wharf", "congestion_ratio": 0.85, "current_speed": 42},
            ],
        }
        insights = engine._analyze_congestion_patterns(traffic)

        clear = [i for i in insights if "flowing well" in i.title.lower()]
        assert len(clear) == 1

    def test_high_disruptions(self, engine):
        traffic = {"has_congestion": True, "points": [], "disruption_count": 15}
        insights = engine._analyze_congestion_patterns(traffic)

        disruption = [i for i in insights if "disruption" in i.title.lower()]
        assert len(disruption) == 1

    def test_incidents_plus_congestion(self, engine):
        traffic = {
            "has_congestion": True,
            "points": [{"location": "X", "congestion_ratio": 0.3, "current_speed": 15}],
            "incident_count": 8,
        }
        insights = engine._analyze_congestion_patterns(traffic)

        incident = [i for i in insights if "incident" in i.title.lower()]
        assert len(incident) == 1


class TestTubePatterns:

    def test_tube_all_clear(self, engine):
        tube = {"all_clear": True, "disrupted_lines": []}
        insights = engine._analyze_tube_patterns(tube)
        assert any("normally" in i.title.lower() for i in insights)

    def test_tube_significant_disruptions(self, engine):
        tube = {"all_clear": False, "disrupted_count": 4, "total_lines": 11, "disrupted_lines": []}
        insights = engine._analyze_tube_patterns(tube)

        significant = [i for i in insights if "Significant" in i.title]
        assert len(significant) == 1
        assert "road" in significant[0].description.lower()


# ── Integration Tests ─────────────────────────────────────────────


class TestFullCorrelation:

    def test_rain_congestion_poor_aq(self):
        """Full scenario: rain + congestion + poor air quality."""
        results = {
            "get_current_weather": WEATHER_RAINY,
            "get_traffic_flow": TRAFFIC_CONGESTED,
            "get_air_quality": AQ_UNHEALTHY,
        }
        insights = correlate_data(results)

        assert len(insights) >= 3
        types = [i.insight_type for i in insights]
        assert InsightType.WEATHER_TRAFFIC in types
        assert InsightType.AIR_QUALITY_TRAFFIC in types

    def test_clear_day_no_issues(self):
        """Full scenario: clear weather, free traffic, good AQ."""
        results = {
            "get_current_weather": WEATHER_CLEAR,
            "get_traffic_flow": TRAFFIC_FREE,
            "get_air_quality": AQ_GOOD,
            "get_tube_status": TUBE_GOOD,
        }
        insights = correlate_data(results)

        # Should still produce insights (positive ones)
        assert len(insights) >= 1
        assert any(i.confidence == Confidence.HIGH for i in insights)

    def test_single_source_no_correlation(self):
        """Only one data source → minimal correlation possible."""
        results = {"get_tube_status": TUBE_GOOD}
        insights = correlate_data(results)
        # Tube-only should still give tube pattern insight
        assert any("Tube" in i.title or "tube" in i.title for i in insights)

    def test_format_for_llm(self):
        """Formatted output should be readable by the LLM."""
        results = {
            "get_current_weather": WEATHER_RAINY,
            "get_traffic_flow": TRAFFIC_CONGESTED,
        }
        insights = correlate_data(results)
        formatted = format_insights_for_llm(insights)

        assert "CORRELATION INSIGHTS" in formatted
        assert "🟢" in formatted or "🟡" in formatted or "🔴" in formatted

    def test_empty_results(self):
        insights = correlate_data({})
        assert insights == []

    def test_all_errors(self):
        results = {
            "get_current_weather": "ERROR: timeout",
            "get_traffic_flow": "ERROR: 500",
        }
        insights = correlate_data(results)
        assert insights == []


# ── Insight Model Tests ───────────────────────────────────────────


class TestInsightModel:

    def test_insight_to_string(self):
        insight = Insight(
            insight_type=InsightType.WEATHER_TRAFFIC,
            title="Rain causing delays",
            description="It's raining and traffic is slow.",
            confidence=Confidence.HIGH,
        )
        s = insight.to_string()
        assert "🟢" in s
        assert "high" in s
        assert "Rain causing delays" in s

    def test_insight_with_data_points(self):
        insight = Insight(
            insight_type=InsightType.ANOMALY,
            title="Test",
            description="Test desc",
            confidence=Confidence.LOW,
            data_points={"pm25": 42.0},
        )
        assert insight.data_points["pm25"] == 42.0
