"""
Smart City AI Agent - Anomaly Detection Tests
Tests threshold-based anomaly detection and health scoring.

Run: pytest tests/test_anomaly.py -v
"""

import pytest

from app.agent.anomaly import (
    AnomalyDetector,
    detect_anomalies,
    compute_city_health,
    format_anomalies_for_llm,
    AlertLevel,
    Anomaly,
)


@pytest.fixture
def detector():
    return AnomalyDetector()


# ── Traffic Anomaly Tests ─────────────────────────────────────────


class TestTrafficAnomalies:

    def test_critical_congestion(self, detector):
        traffic = {"points": [{"location": "Camden", "congestion_ratio": 0.15, "current_speed": 7}]}
        anomalies = detector._check_traffic(traffic)
        critical = [a for a in anomalies if a.level == AlertLevel.CRITICAL]
        assert len(critical) == 1
        assert "Camden" in critical[0].title
        assert critical[0].recommendation != ""

    def test_warning_congestion(self, detector):
        traffic = {"points": [{"location": "City", "congestion_ratio": 0.35, "current_speed": 17}]}
        anomalies = detector._check_traffic(traffic)
        warnings = [a for a in anomalies if a.level == AlertLevel.WARNING]
        assert len(warnings) == 1

    def test_no_congestion_anomaly(self, detector):
        traffic = {"points": [{"location": "Central", "congestion_ratio": 0.85, "current_speed": 42}]}
        anomalies = detector._check_traffic(traffic)
        assert len(anomalies) == 0

    def test_high_incidents(self, detector):
        traffic = {"points": [], "incident_count": 15}
        anomalies = detector._check_traffic(traffic)
        assert any("incident" in a.title.lower() for a in anomalies)

    def test_high_disruptions(self, detector):
        traffic = {"points": [], "disruption_count": 25}
        anomalies = detector._check_traffic(traffic)
        assert any("disruption" in a.title.lower() for a in anomalies)

    def test_multiple_congested_points(self, detector):
        traffic = {"points": [
            {"location": "A", "congestion_ratio": 0.18, "current_speed": 8},
            {"location": "B", "congestion_ratio": 0.30, "current_speed": 15},
            {"location": "C", "congestion_ratio": 0.80, "current_speed": 40},
        ]}
        anomalies = detector._check_traffic(traffic)
        assert len(anomalies) == 2  # A=critical, B=warning


# ── Weather Anomaly Tests ─────────────────────────────────────────


class TestWeatherAnomalies:

    def test_dangerous_wind(self, detector):
        weather = {"wind_speed_kmh": 90, "temperature_c": 15, "precipitation_mm": 0}
        anomalies = detector._check_weather(weather)
        critical = [a for a in anomalies if a.level == AlertLevel.CRITICAL]
        assert len(critical) == 1
        assert "Dangerous" in critical[0].title

    def test_high_wind(self, detector):
        weather = {"wind_speed_kmh": 55, "temperature_c": 15, "precipitation_mm": 0}
        anomalies = detector._check_weather(weather)
        warnings = [a for a in anomalies if a.level == AlertLevel.WARNING]
        assert any("wind" in a.title.lower() for a in warnings)

    def test_heavy_rain(self, detector):
        weather = {"wind_speed_kmh": 10, "temperature_c": 10, "precipitation_mm": 7.5}
        anomalies = detector._check_weather(weather)
        assert any("precipitation" in a.title.lower() for a in anomalies)

    def test_freezing(self, detector):
        weather = {"wind_speed_kmh": 10, "temperature_c": -2.0, "precipitation_mm": 0}
        anomalies = detector._check_weather(weather)
        assert any("Freezing" in a.title for a in anomalies)

    def test_heat(self, detector):
        weather = {"wind_speed_kmh": 5, "temperature_c": 33.0, "precipitation_mm": 0}
        anomalies = detector._check_weather(weather)
        assert any("temperature" in a.title.lower() for a in anomalies)

    def test_fog(self, detector):
        weather = {"wind_speed_kmh": 3, "temperature_c": 8, "precipitation_mm": 0, "is_foggy": True}
        anomalies = detector._check_weather(weather)
        assert any("Fog" in a.title for a in anomalies)

    def test_normal_weather_no_anomalies(self, detector):
        weather = {"wind_speed_kmh": 15, "temperature_c": 15, "precipitation_mm": 0}
        assert len(detector._check_weather(weather)) == 0


# ── Air Quality Anomaly Tests ─────────────────────────────────────


class TestAirQualityAnomalies:

    def test_unhealthy_pm25(self, detector):
        aq = {"pm25": 60.0, "no2": 30.0}
        anomalies = detector._check_air_quality(aq)
        critical = [a for a in anomalies if a.level == AlertLevel.CRITICAL]
        assert any("Unhealthy" in a.title for a in critical)

    def test_unhealthy_sensitive_pm25(self, detector):
        aq = {"pm25": 40.0, "no2": 25.0}
        anomalies = detector._check_air_quality(aq)
        warnings = [a for a in anomalies if a.level == AlertLevel.WARNING]
        assert any("sensitive" in a.title.lower() for a in warnings)

    def test_moderate_pm25(self, detector):
        aq = {"pm25": 15.0, "no2": 20.0}
        anomalies = detector._check_air_quality(aq)
        info = [a for a in anomalies if a.level == AlertLevel.INFO]
        assert any("Moderate" in a.title for a in info)

    def test_good_pm25_no_anomaly(self, detector):
        aq = {"pm25": 8.0, "no2": 20.0}
        anomalies = detector._check_air_quality(aq)
        pm_anomalies = [a for a in anomalies if a.metric == "pm25"]
        assert len(pm_anomalies) == 0

    def test_high_no2(self, detector):
        aq = {"pm25": 10.0, "no2": 110.0}
        anomalies = detector._check_air_quality(aq)
        critical = [a for a in anomalies if a.level == AlertLevel.CRITICAL and a.metric == "no2"]
        assert len(critical) == 1

    def test_elevated_no2(self, detector):
        aq = {"pm25": 10.0, "no2": 45.0}
        anomalies = detector._check_air_quality(aq)
        warnings = [a for a in anomalies if a.metric == "no2"]
        assert len(warnings) == 1


# ── Tube Anomaly Tests ────────────────────────────────────────────


class TestTubeAnomalies:

    def test_all_clear(self, detector):
        tube = {"all_clear": True, "disrupted_count": 0, "total_lines": 11, "disrupted_lines": []}
        assert len(detector._check_tube(tube)) == 0

    def test_major_disruptions_warning(self, detector):
        tube = {"all_clear": False, "disrupted_count": 4, "total_lines": 11, "disrupted_lines": []}
        anomalies = detector._check_tube(tube)
        assert len(anomalies) == 1
        assert anomalies[0].level == AlertLevel.WARNING

    def test_severe_disruptions_critical(self, detector):
        tube = {"all_clear": False, "disrupted_count": 7, "total_lines": 11, "disrupted_lines": []}
        anomalies = detector._check_tube(tube)
        assert anomalies[0].level == AlertLevel.CRITICAL

    def test_minor_disruptions_info(self, detector):
        tube = {"all_clear": False, "disrupted_count": 2, "total_lines": 11, "disrupted_lines": []}
        anomalies = detector._check_tube(tube)
        assert len(anomalies) == 1
        assert anomalies[0].level == AlertLevel.INFO


# ── Health Score Tests ────────────────────────────────────────────


class TestHealthScore:

    def test_perfect_conditions(self, detector):
        data = {
            "weather": {"precipitation_mm": 0, "wind_speed_kmh": 8, "temperature_c": 18, "is_foggy": False, "is_stormy": False},
            "traffic": {"points": [{"congestion_ratio": 0.90}]},
            "air_quality": {"pm25": 5.0},
            "tube": {"all_clear": True, "disrupted_count": 0, "total_lines": 11},
        }
        scores = detector.compute_health_score(data)
        assert scores["overall"] >= 85
        assert scores["traffic"] >= 85
        assert scores["tube"] == 100

    def test_terrible_conditions(self, detector):
        data = {
            "weather": {"precipitation_mm": 10, "wind_speed_kmh": 60, "temperature_c": -3, "is_foggy": True, "is_stormy": True},
            "traffic": {"points": [{"congestion_ratio": 0.15}]},
            "air_quality": {"pm25": 80.0},
            "tube": {"all_clear": False, "disrupted_count": 8, "total_lines": 11},
        }
        scores = detector.compute_health_score(data)
        assert scores["overall"] <= 30
        assert scores["traffic"] <= 20
        assert scores["tube"] <= 30

    def test_partial_data(self, detector):
        data = {"weather": {"precipitation_mm": 0, "wind_speed_kmh": 10, "temperature_c": 15, "is_foggy": False, "is_stormy": False}}
        scores = detector.compute_health_score(data)
        assert scores["overall"] is not None
        assert scores["weather"] is not None
        assert scores["traffic"] is None

    def test_no_data(self, detector):
        scores = detector.compute_health_score({})
        assert scores["overall"] is None

    def test_scores_bounded_0_100(self, detector):
        data = {
            "weather": {"precipitation_mm": 50, "wind_speed_kmh": 100, "temperature_c": -20, "is_foggy": True, "is_stormy": True},
            "traffic": {"points": [{"congestion_ratio": 0.01}]},
            "air_quality": {"pm25": 500},
            "tube": {"all_clear": False, "disrupted_count": 11, "total_lines": 11},
        }
        scores = detector.compute_health_score(data)
        for key, val in scores.items():
            if val is not None:
                assert 0 <= val <= 100, f"{key} score {val} out of bounds"


# ── Integration Tests ─────────────────────────────────────────────


class TestFullAnomalyDetection:

    def test_detect_with_all_sources(self):
        data = {
            "weather": {"wind_speed_kmh": 55, "temperature_c": 15, "precipitation_mm": 6},
            "traffic": {"points": [{"location": "Camden", "congestion_ratio": 0.18, "current_speed": 8}]},
            "air_quality": {"pm25": 42.0, "no2": 50.0},
            "tube": {"all_clear": False, "disrupted_count": 5, "total_lines": 11, "disrupted_lines": []},
        }
        anomalies = detect_anomalies(data)
        assert len(anomalies) >= 4
        categories = {a.category for a in anomalies}
        assert "traffic" in categories
        assert "weather" in categories
        assert "air_quality" in categories
        assert "tube" in categories

    def test_sorted_by_severity(self):
        data = {
            "weather": {"wind_speed_kmh": 90, "temperature_c": 15, "precipitation_mm": 0},
            "tube": {"all_clear": False, "disrupted_count": 2, "total_lines": 11, "disrupted_lines": []},
        }
        anomalies = detect_anomalies(data)
        # Critical should come first
        if len(anomalies) >= 2:
            assert anomalies[0].level.value <= anomalies[-1].level.value or anomalies[0].level == AlertLevel.CRITICAL


class TestFormatAnomalies:

    def test_format_empty(self):
        result = format_anomalies_for_llm([])
        assert "No anomalies" in result

    def test_format_with_anomalies(self):
        anomalies = [Anomaly(
            level=AlertLevel.WARNING, category="traffic",
            title="Heavy congestion", description="Traffic is slow",
            metric="ratio", current_value=0.3, threshold=0.4,
            recommendation="Avoid the area",
        )]
        result = format_anomalies_for_llm(anomalies)
        assert "⚠️" in result
        assert "Heavy congestion" in result
        assert "Avoid the area" in result


class TestAnomalyModel:

    def test_to_string(self):
        a = Anomaly(
            level=AlertLevel.CRITICAL, category="weather",
            title="Dangerous wind", description="Very windy",
            metric="wind", current_value=90, threshold=80,
        )
        s = a.to_string()
        assert "🚨" in s
        assert "CRITICAL" in s
