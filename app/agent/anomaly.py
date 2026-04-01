"""
Smart City AI Agent - Anomaly Detection
Threshold-based anomaly detection across all data sources.
Flags unusual conditions and computes a city health score.

This module extends the correlation engine with proactive alerting.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Severity of an anomaly alert."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """A detected anomaly / unusual condition."""
    level: AlertLevel
    category: str  # "traffic", "weather", "air_quality", "tube"
    title: str
    description: str
    metric: str
    current_value: float | str
    threshold: float | str
    recommendation: str = ""

    def to_string(self) -> str:
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
        }
        emoji = level_emoji.get(self.level, "❓")
        return f"{emoji} [{self.level.value.upper()}] {self.title}: {self.description}"


# ══════════════════════════════════════════════════════════════════
# Thresholds
# ══════════════════════════════════════════════════════════════════

# Traffic thresholds
CONGESTION_WARNING = 0.40    # Below 40% of free-flow = warning
CONGESTION_CRITICAL = 0.20   # Below 20% = critical

# Air quality thresholds (PM2.5 µg/m³, WHO guidelines)
PM25_MODERATE = 12.0
PM25_UNHEALTHY_SENSITIVE = 35.4
PM25_UNHEALTHY = 55.4

# NO2 thresholds (µg/m³)
NO2_ELEVATED = 40.0
NO2_HIGH = 100.0

# Weather thresholds
WIND_HIGH = 50.0          # km/h
WIND_DANGEROUS = 80.0     # km/h
TEMP_FREEZING = 0.0       # °C
TEMP_HEAT = 30.0          # °C
PRECIP_HEAVY = 5.0        # mm

# Tube thresholds
TUBE_MINOR_DISRUPTIONS = 2
TUBE_MAJOR_DISRUPTIONS = 4


class AnomalyDetector:
    """
    Detects anomalies across all data sources using predefined thresholds.
    Also computes a city health score (0-100).
    """

    def detect(self, extracted_data: dict) -> list[Anomaly]:
        """
        Run all anomaly checks on extracted data.
        Args:
            extracted_data: Dict with keys "weather", "traffic", "air_quality", "tube"
                          containing extracted data from the correlation engine.
        """
        anomalies = []

        weather = extracted_data.get("weather")
        traffic = extracted_data.get("traffic")
        air_quality = extracted_data.get("air_quality")
        tube = extracted_data.get("tube")

        if traffic:
            anomalies.extend(self._check_traffic(traffic))
        if weather:
            anomalies.extend(self._check_weather(weather))
        if air_quality:
            anomalies.extend(self._check_air_quality(air_quality))
        if tube:
            anomalies.extend(self._check_tube(tube))

        # Sort by severity (critical first)
        level_order = {AlertLevel.CRITICAL: 0, AlertLevel.WARNING: 1, AlertLevel.INFO: 2}
        anomalies.sort(key=lambda a: level_order.get(a.level, 3))

        logger.info(f"🔍 Anomaly detector: {len(anomalies)} anomalie(s) found")
        return anomalies

    def compute_health_score(self, extracted_data: dict) -> dict:
        """
        Compute a city health score from 0-100.
        100 = perfect conditions, 0 = everything is terrible.

        Returns dict with overall score and per-category scores.
        """
        scores = {}

        # Traffic score (0-100)
        traffic = extracted_data.get("traffic")
        if traffic:
            points = traffic.get("points", [])
            if points:
                ratios = [p.get("congestion_ratio", 1.0) for p in points]
                avg_ratio = sum(ratios) / len(ratios)
                scores["traffic"] = round(min(100, avg_ratio * 100))
            else:
                scores["traffic"] = 50  # No data = neutral
        else:
            scores["traffic"] = None

        # Weather score (0-100)
        weather = extracted_data.get("weather")
        if weather:
            score = 100
            precip = weather.get("precipitation_mm", 0)
            wind = weather.get("wind_speed_kmh", 0)
            temp = weather.get("temperature_c", 15)

            if precip > 0:
                score -= min(30, precip * 6)
            if wind > 30:
                score -= min(20, (wind - 30) * 0.5)
            if weather.get("is_foggy"):
                score -= 15
            if weather.get("is_stormy"):
                score -= 25
            if temp < TEMP_FREEZING or temp > TEMP_HEAT:
                score -= 10

            scores["weather"] = round(max(0, min(100, score)))
        else:
            scores["weather"] = None

        # Air quality score (0-100)
        air_quality = extracted_data.get("air_quality")
        if air_quality:
            pm25 = air_quality.get("pm25", 0)
            if pm25 <= PM25_MODERATE:
                scores["air_quality"] = round(100 - (pm25 / PM25_MODERATE * 15))
            elif pm25 <= PM25_UNHEALTHY_SENSITIVE:
                scores["air_quality"] = round(85 - ((pm25 - PM25_MODERATE) / (PM25_UNHEALTHY_SENSITIVE - PM25_MODERATE) * 35))
            elif pm25 <= PM25_UNHEALTHY:
                scores["air_quality"] = round(50 - ((pm25 - PM25_UNHEALTHY_SENSITIVE) / (PM25_UNHEALTHY - PM25_UNHEALTHY_SENSITIVE) * 30))
            else:
                scores["air_quality"] = max(0, round(20 - (pm25 - PM25_UNHEALTHY) * 0.1))
        else:
            scores["air_quality"] = None

        # Tube score (0-100)
        tube = extracted_data.get("tube")
        if tube:
            if tube.get("all_clear"):
                scores["tube"] = 100
            else:
                disrupted = tube.get("disrupted_count", 0)
                total = tube.get("total_lines", 11)
                scores["tube"] = round(max(0, 100 - (disrupted / total * 100)))
        else:
            scores["tube"] = None

        # Overall score (weighted average of available scores)
        available = {k: v for k, v in scores.items() if v is not None}
        if available:
            weights = {"traffic": 0.35, "weather": 0.2, "air_quality": 0.2, "tube": 0.25}
            total_weight = sum(weights.get(k, 0.25) for k in available)
            weighted_sum = sum(v * weights.get(k, 0.25) for k, v in available.items())
            scores["overall"] = round(weighted_sum / total_weight)
        else:
            scores["overall"] = None

        return scores

    # ── Traffic Checks ────────────────────────────────────────────

    def _check_traffic(self, traffic: dict) -> list[Anomaly]:
        anomalies = []
        points = traffic.get("points", [])

        for point in points:
            ratio = point.get("congestion_ratio", 1.0)
            speed = point.get("current_speed", 0)
            location = point.get("location", "Unknown")

            if ratio < CONGESTION_CRITICAL:
                anomalies.append(Anomaly(
                    level=AlertLevel.CRITICAL,
                    category="traffic",
                    title=f"Severe congestion at {location}",
                    description=(
                        f"Traffic at {location} is at {ratio:.0%} of free-flow speed "
                        f"({speed:.0f} km/h). This indicates near-standstill conditions."
                    ),
                    metric="congestion_ratio",
                    current_value=ratio,
                    threshold=CONGESTION_CRITICAL,
                    recommendation=f"Avoid {location} if possible. Consider tube or alternative routes.",
                ))
            elif ratio < CONGESTION_WARNING:
                anomalies.append(Anomaly(
                    level=AlertLevel.WARNING,
                    category="traffic",
                    title=f"Heavy congestion at {location}",
                    description=(
                        f"Traffic at {location} is at {ratio:.0%} of free-flow speed "
                        f"({speed:.0f} km/h). Expect significant delays."
                    ),
                    metric="congestion_ratio",
                    current_value=ratio,
                    threshold=CONGESTION_WARNING,
                    recommendation=f"Allow extra travel time through {location}.",
                ))

        # High incident count
        incidents = traffic.get("incident_count", 0)
        if incidents > 10:
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="traffic",
                title="High number of traffic incidents",
                description=f"{incidents} active incidents across London — above typical levels.",
                metric="incident_count",
                current_value=incidents,
                threshold=10,
                recommendation="Check specific routes before travelling.",
            ))

        # High disruption count
        disruptions = traffic.get("disruption_count", 0)
        if disruptions > 20:
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="traffic",
                title="Unusually high road disruptions",
                description=f"{disruptions} road disruptions active — significantly above normal.",
                metric="disruption_count",
                current_value=disruptions,
                threshold=20,
                recommendation="Major roadworks may be causing network-wide delays.",
            ))

        return anomalies

    # ── Weather Checks ────────────────────────────────────────────

    def _check_weather(self, weather: dict) -> list[Anomaly]:
        anomalies = []

        wind = weather.get("wind_speed_kmh", 0)
        temp = weather.get("temperature_c", 15)
        precip = weather.get("precipitation_mm", 0)

        if wind >= WIND_DANGEROUS:
            anomalies.append(Anomaly(
                level=AlertLevel.CRITICAL,
                category="weather",
                title="Dangerous wind speeds",
                description=f"Wind at {wind:.0f} km/h — may cause structural damage and transport disruptions.",
                metric="wind_speed_kmh",
                current_value=wind,
                threshold=WIND_DANGEROUS,
                recommendation="Avoid exposed areas and bridges. Check for transport cancellations.",
            ))
        elif wind >= WIND_HIGH:
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="weather",
                title="High wind speeds",
                description=f"Wind at {wind:.0f} km/h — may affect high-sided vehicles and cycling.",
                metric="wind_speed_kmh",
                current_value=wind,
                threshold=WIND_HIGH,
                recommendation="Take care on exposed routes. Bridges may have restrictions.",
            ))

        if precip >= PRECIP_HEAVY:
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="weather",
                title="Heavy precipitation",
                description=f"Precipitation at {precip:.1f} mm — expect reduced visibility and slower traffic.",
                metric="precipitation_mm",
                current_value=precip,
                threshold=PRECIP_HEAVY,
                recommendation="Allow extra travel time. Drive carefully.",
            ))

        if temp <= TEMP_FREEZING:
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="weather",
                title="Freezing temperatures",
                description=f"Temperature at {temp:.1f}°C — risk of ice on roads and pavements.",
                metric="temperature_c",
                current_value=temp,
                threshold=TEMP_FREEZING,
                recommendation="Watch for black ice, especially on bridges and overpasses.",
            ))

        if temp >= TEMP_HEAT:
            anomalies.append(Anomaly(
                level=AlertLevel.INFO,
                category="weather",
                title="High temperatures",
                description=f"Temperature at {temp:.1f}°C — stay hydrated and avoid prolonged sun exposure.",
                metric="temperature_c",
                current_value=temp,
                threshold=TEMP_HEAT,
                recommendation="Tube carriages may be uncomfortably warm. Carry water.",
            ))

        if weather.get("is_foggy"):
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="weather",
                title="Fog affecting visibility",
                description="Foggy conditions reducing visibility significantly.",
                metric="visibility",
                current_value="fog",
                threshold="clear",
                recommendation="Drive with fog lights. Expect slower traffic.",
            ))

        return anomalies

    # ── Air Quality Checks ────────────────────────────────────────

    def _check_air_quality(self, air_quality: dict) -> list[Anomaly]:
        anomalies = []

        pm25 = air_quality.get("pm25", 0)
        no2 = air_quality.get("no2", 0)

        if pm25 > PM25_UNHEALTHY:
            anomalies.append(Anomaly(
                level=AlertLevel.CRITICAL,
                category="air_quality",
                title="Unhealthy air quality",
                description=(
                    f"PM2.5 at {pm25:.1f} µg/m³ — significantly above healthy levels. "
                    f"Everyone may experience health effects."
                ),
                metric="pm25",
                current_value=pm25,
                threshold=PM25_UNHEALTHY,
                recommendation="Reduce outdoor activity. Sensitive groups should stay indoors.",
            ))
        elif pm25 > PM25_UNHEALTHY_SENSITIVE:
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="air_quality",
                title="Air quality unhealthy for sensitive groups",
                description=(
                    f"PM2.5 at {pm25:.1f} µg/m³ — above safe levels for children, "
                    f"elderly, and those with respiratory conditions."
                ),
                metric="pm25",
                current_value=pm25,
                threshold=PM25_UNHEALTHY_SENSITIVE,
                recommendation="Sensitive groups should limit prolonged outdoor exertion.",
            ))
        elif pm25 > PM25_MODERATE:
            anomalies.append(Anomaly(
                level=AlertLevel.INFO,
                category="air_quality",
                title="Moderate air quality",
                description=f"PM2.5 at {pm25:.1f} µg/m³ — acceptable for most people.",
                metric="pm25",
                current_value=pm25,
                threshold=PM25_MODERATE,
                recommendation="Unusually sensitive individuals may want to reduce prolonged outdoor exertion.",
            ))

        if no2 > NO2_HIGH:
            anomalies.append(Anomaly(
                level=AlertLevel.CRITICAL,
                category="air_quality",
                title="Very high NO₂ levels",
                description=f"NO₂ at {no2:.1f} µg/m³ — well above safe levels.",
                metric="no2",
                current_value=no2,
                threshold=NO2_HIGH,
                recommendation="Avoid busy roadsides. Consider face coverings near heavy traffic.",
            ))
        elif no2 > NO2_ELEVATED:
            anomalies.append(Anomaly(
                level=AlertLevel.WARNING,
                category="air_quality",
                title="Elevated NO₂ levels",
                description=(
                    f"NO₂ at {no2:.1f} µg/m³ — exceeds WHO annual guideline. "
                    f"Common in London due to diesel traffic."
                ),
                metric="no2",
                current_value=no2,
                threshold=NO2_ELEVATED,
                recommendation="Avoid prolonged exposure near busy roads.",
            ))

        return anomalies

    # ── Tube Checks ───────────────────────────────────────────────

    def _check_tube(self, tube: dict) -> list[Anomaly]:
        anomalies = []

        if tube.get("all_clear"):
            return anomalies

        disrupted = tube.get("disrupted_count", 0)
        total = tube.get("total_lines", 11)

        if disrupted >= TUBE_MAJOR_DISRUPTIONS:
            anomalies.append(Anomaly(
                level=AlertLevel.CRITICAL if disrupted >= 6 else AlertLevel.WARNING,
                category="tube",
                title=f"Major tube disruptions ({disrupted}/{total} lines)",
                description=(
                    f"{disrupted} of {total} tube lines are disrupted. "
                    f"This is significantly above normal and will push commuters onto roads."
                ),
                metric="disrupted_lines",
                current_value=disrupted,
                threshold=TUBE_MAJOR_DISRUPTIONS,
                recommendation="Check TfL status before travelling. Consider buses or cycling.",
            ))
        elif disrupted >= TUBE_MINOR_DISRUPTIONS:
            anomalies.append(Anomaly(
                level=AlertLevel.INFO,
                category="tube",
                title=f"Minor tube disruptions ({disrupted}/{total} lines)",
                description=f"{disrupted} tube lines have disruptions — mostly minor delays.",
                metric="disrupted_lines",
                current_value=disrupted,
                threshold=TUBE_MINOR_DISRUPTIONS,
                recommendation="Allow extra time if using affected lines.",
            ))

        return anomalies


# ── Module-level convenience ──────────────────────────────────────
_detector = AnomalyDetector()


def detect_anomalies(extracted_data: dict) -> list[Anomaly]:
    """Run anomaly detection on extracted data."""
    return _detector.detect(extracted_data)


def compute_city_health(extracted_data: dict) -> dict:
    """Compute city health score."""
    return _detector.compute_health_score(extracted_data)


def format_anomalies_for_llm(anomalies: list[Anomaly]) -> str:
    """Format anomalies for the LLM analyzer."""
    if not anomalies:
        return "ANOMALY CHECK: No anomalies detected. All metrics within normal ranges."

    lines = [f"ANOMALY ALERTS ({len(anomalies)} detected):"]
    for a in anomalies:
        lines.append(a.to_string())
        if a.recommendation:
            lines.append(f"  → Recommendation: {a.recommendation}")
    return "\n".join(lines)
