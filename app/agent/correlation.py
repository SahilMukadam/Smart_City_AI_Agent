"""
Smart City AI Agent - Data Correlation Engine
Cross-analyzes traffic, weather, and air quality data to find
patterns, correlations, and anomalies.

This runs BEFORE the LLM analyzer so the LLM gets pre-computed
insights instead of raw data — leading to much better analysis.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """Categories of insight the correlation engine can produce."""
    WEATHER_TRAFFIC = "weather_traffic"
    AIR_QUALITY_TRAFFIC = "air_quality_traffic"
    WEATHER_AIR_QUALITY = "weather_air_quality"
    CONGESTION_PATTERN = "congestion_pattern"
    ANOMALY = "anomaly"
    SUMMARY = "summary"


class Confidence(str, Enum):
    """Confidence levels for insights."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """A single correlation insight."""
    insight_type: InsightType
    title: str
    description: str
    confidence: Confidence
    data_points: dict = field(default_factory=dict)

    def to_string(self) -> str:
        """Format for the LLM analyzer."""
        confidence_emoji = {
            Confidence.HIGH: "🟢",
            Confidence.MEDIUM: "🟡",
            Confidence.LOW: "🔴",
        }
        emoji = confidence_emoji.get(self.confidence, "⚪")
        return f"{emoji} [{self.confidence.value}] {self.title}: {self.description}"


class CorrelationEngine:
    """
    Analyzes tool results to find cross-source correlations.

    Extracts structured data from tool result strings,
    then runs correlation checks across data sources.
    """

    def analyze(self, tool_results: dict[str, str]) -> list[Insight]:
        """
        Main entry: analyze all tool results and produce insights.
        Returns a list of Insight objects sorted by confidence.
        """
        insights: list[Insight] = []

        # Extract structured data from tool result strings
        weather = self._extract_weather_data(tool_results)
        traffic = self._extract_traffic_data(tool_results)
        air_quality = self._extract_air_quality_data(tool_results)
        tube = self._extract_tube_data(tool_results)

        # Run correlation checks
        if weather and traffic:
            insights.extend(self._correlate_weather_traffic(weather, traffic))

        if traffic and air_quality:
            insights.extend(self._correlate_traffic_air_quality(traffic, air_quality))

        if weather and air_quality:
            insights.extend(self._correlate_weather_air_quality(weather, air_quality))

        if traffic:
            insights.extend(self._analyze_congestion_patterns(traffic))

        if tube:
            insights.extend(self._analyze_tube_patterns(tube))

        # Sort by confidence (high first)
        confidence_order = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
        insights.sort(key=lambda i: confidence_order.get(i.confidence, 3))

        logger.info(f"📊 Correlation engine produced {len(insights)} insight(s)")
        return insights

    # ══════════════════════════════════════════════════════════════
    # Data Extraction (from tool result strings)
    # ══════════════════════════════════════════════════════════════

    def _extract_weather_data(self, results: dict) -> dict | None:
        """Extract weather metrics from tool result strings."""
        weather_str = None
        for key, val in results.items():
            if "weather" in key.lower() and "forecast" not in key.lower():
                weather_str = val
                break

        # Also check forecast
        forecast_str = None
        for key, val in results.items():
            if "forecast" in key.lower():
                forecast_str = val
                break

        if not weather_str:
            return None

        data = {}

        # Extract temperature
        temp_match = re.search(r"Temperature:\s*([-\d.]+)\s*°C", weather_str)
        if temp_match:
            data["temperature_c"] = float(temp_match.group(1))

        # Extract humidity
        humidity_match = re.search(r"Humidity:\s*(\d+)%", weather_str)
        if humidity_match:
            data["humidity_percent"] = int(humidity_match.group(1))

        # Extract wind speed
        wind_match = re.search(r"Wind:\s*([-\d.]+)\s*km/h", weather_str)
        if wind_match:
            data["wind_speed_kmh"] = float(wind_match.group(1))

        # Extract precipitation
        precip_match = re.search(r"Precipitation:\s*([-\d.]+)\s*mm", weather_str)
        if precip_match:
            data["precipitation_mm"] = float(precip_match.group(1))
        else:
            data["precipitation_mm"] = 0.0

        # Detect rain/adverse conditions from description
        weather_lower = weather_str.lower()
        data["is_raining"] = any(
            w in weather_lower for w in ["rain", "drizzle", "shower", "precipitation"]
        )
        data["is_foggy"] = "fog" in weather_lower
        data["is_stormy"] = any(
            w in weather_lower for w in ["thunder", "storm"]
        )
        data["has_poor_visibility"] = data["is_foggy"] or data["is_stormy"]

        # Forecast data
        if forecast_str:
            data["rain_likely"] = any(
                w in forecast_str.lower() for w in ["rain likely", "rain expected"]
            )
            rain_prob_match = re.search(r"up to (\d+)%\s*(?:chance|probability)", forecast_str)
            if rain_prob_match:
                data["max_rain_probability"] = int(rain_prob_match.group(1))

        return data if data else None

    def _extract_traffic_data(self, results: dict) -> dict | None:
        """Extract traffic metrics from tool result strings."""
        data = {"points": [], "has_congestion": False}

        for key, val in results.items():
            if "traffic" not in key.lower() and "tomtom" not in key.lower():
                continue
            if "incident" in key.lower():
                data["incidents_text"] = val
                # Count incidents
                incident_count = re.search(r"(\d+)\s*traffic incidents", val)
                if incident_count:
                    data["incident_count"] = int(incident_count.group(1))
                continue

            # Single point traffic flow
            speed_match = re.search(r"Current speed:\s*([-\d.]+)\s*km/h", val)
            free_flow_match = re.search(r"free-flow:\s*([-\d.]+)\s*km/h", val)
            ratio_match = re.search(r"Congestion ratio:\s*([\d.]+)%", val)

            if speed_match and free_flow_match:
                point = {
                    "current_speed": float(speed_match.group(1)),
                    "free_flow_speed": float(free_flow_match.group(1)),
                    "source_key": key,
                }
                if ratio_match:
                    point["congestion_ratio"] = float(ratio_match.group(1)) / 100
                elif point["free_flow_speed"] > 0:
                    point["congestion_ratio"] = point["current_speed"] / point["free_flow_speed"]
                else:
                    point["congestion_ratio"] = 1.0

                data["points"].append(point)

                if point["congestion_ratio"] < 0.65:
                    data["has_congestion"] = True

            # Multi-point overview
            if "worst congestion first" in val.lower() or "locations" in val.lower():
                data["is_overview"] = True
                # Extract individual location data
                location_matches = re.findall(
                    r"-\s*(.+?):\s*(.+?)\s*\(([\d.]+)\s*km/h,\s*([\d.]+)%",
                    val,
                )
                for loc_name, level, speed, ratio in location_matches:
                    point = {
                        "location": loc_name.strip(),
                        "congestion_level": level.strip(),
                        "current_speed": float(speed),
                        "congestion_ratio": float(ratio) / 100,
                        "source_key": key,
                    }
                    data["points"].append(point)
                    if point["congestion_ratio"] < 0.65:
                        data["has_congestion"] = True

        # Check TfL disruptions
        for key, val in results.items():
            if "disruption" in key.lower():
                data["disruptions_text"] = val
                disruption_count = re.search(r"(\d+)\s*road disruptions", val)
                if disruption_count:
                    data["disruption_count"] = int(disruption_count.group(1))

        return data if (data["points"] or "incident_count" in data or "disruption_count" in data) else None

    def _extract_air_quality_data(self, results: dict) -> dict | None:
        """Extract air quality metrics from tool result strings."""
        for key, val in results.items():
            if "air_quality" not in key.lower() and "air quality" not in val.lower():
                continue

            data = {}

            # Extract AQI category
            aqi_match = re.search(r"Overall:\s*(\w[\w\s]*?)(?:\s*[—–-])", val)
            if aqi_match:
                data["aqi_category"] = aqi_match.group(1).strip()

            # Extract PM2.5
            pm25_match = re.search(r"PM2\.5.*?:\s*([\d.]+)", val)
            if pm25_match:
                data["pm25"] = float(pm25_match.group(1))

            # Extract NO2
            no2_match = re.search(r"NO[₂2].*?:\s*([\d.]+)", val)
            if no2_match:
                data["no2"] = float(no2_match.group(1))

            # Extract PM10
            pm10_match = re.search(r"PM10.*?:\s*([\d.]+)", val)
            if pm10_match:
                data["pm10"] = float(pm10_match.group(1))

            if data:
                # Classify overall quality
                pm25_val = data.get("pm25", 0)
                if pm25_val <= 12:
                    data["quality_level"] = "good"
                elif pm25_val <= 35.4:
                    data["quality_level"] = "moderate"
                elif pm25_val <= 55.4:
                    data["quality_level"] = "unhealthy_sensitive"
                else:
                    data["quality_level"] = "unhealthy"

                return data

        return None

    def _extract_tube_data(self, results: dict) -> dict | None:
        """Extract tube status from tool result strings."""
        for key, val in results.items():
            if "tube" not in key.lower():
                continue

            data = {"disrupted_lines": []}

            # Check for "good service" on all lines
            if "all" in val.lower() and "good service" in val.lower():
                data["all_clear"] = True
                return data

            data["all_clear"] = False

            # Extract disrupted line count
            disrupted_match = re.search(r"(\d+)\s*of\s*(\d+)\s*tube lines have disruptions", val)
            if disrupted_match:
                data["disrupted_count"] = int(disrupted_match.group(1))
                data["total_lines"] = int(disrupted_match.group(2))

            # Extract individual line disruptions
            line_matches = re.findall(r"-\s*(\w[\w\s]*?):\s*([\w\s]+?)(?:\s*\(|$)", val)
            for line_name, status in line_matches:
                data["disrupted_lines"].append({
                    "name": line_name.strip(),
                    "status": status.strip(),
                })

            return data

        return None

    # ══════════════════════════════════════════════════════════════
    # Correlation Analysis
    # ══════════════════════════════════════════════════════════════

    def _correlate_weather_traffic(
        self, weather: dict, traffic: dict
    ) -> list[Insight]:
        """Find correlations between weather conditions and traffic."""
        insights = []

        # Rain + congestion correlation
        if weather.get("is_raining") and traffic.get("has_congestion"):
            insights.append(Insight(
                insight_type=InsightType.WEATHER_TRAFFIC,
                title="Rain likely contributing to congestion",
                description=(
                    f"Current rainfall ({weather.get('precipitation_mm', 0):.1f}mm) "
                    f"is coinciding with traffic congestion. Rain typically reduces "
                    f"average speeds by 10-20% due to reduced visibility and "
                    f"cautious driving."
                ),
                confidence=Confidence.HIGH,
                data_points={
                    "precipitation_mm": weather.get("precipitation_mm"),
                    "has_congestion": True,
                },
            ))

        # Rain but no congestion — noteworthy
        elif weather.get("is_raining") and not traffic.get("has_congestion"):
            insights.append(Insight(
                insight_type=InsightType.WEATHER_TRAFFIC,
                title="Traffic holding up despite rain",
                description=(
                    "Despite current rainfall, traffic is flowing reasonably well. "
                    "This may be due to off-peak hours or lighter-than-usual volume."
                ),
                confidence=Confidence.MEDIUM,
                data_points={"is_raining": True, "has_congestion": False},
            ))

        # Dry weather + congestion — other causes
        elif not weather.get("is_raining") and traffic.get("has_congestion"):
            insights.append(Insight(
                insight_type=InsightType.WEATHER_TRAFFIC,
                title="Congestion not weather-related",
                description=(
                    "Weather conditions are dry, so current congestion is likely "
                    "caused by other factors (incidents, roadworks, peak hours, events)."
                ),
                confidence=Confidence.HIGH,
                data_points={"is_raining": False, "has_congestion": True},
            ))

        # High wind effects
        wind = weather.get("wind_speed_kmh", 0)
        if wind > 50:
            insights.append(Insight(
                insight_type=InsightType.WEATHER_TRAFFIC,
                title="High winds may affect traffic",
                description=(
                    f"Wind speed of {wind:.0f} km/h could cause issues for "
                    f"high-sided vehicles and may lead to bridge restrictions."
                ),
                confidence=Confidence.MEDIUM,
                data_points={"wind_speed_kmh": wind},
            ))

        # Fog/poor visibility
        if weather.get("has_poor_visibility"):
            insights.append(Insight(
                insight_type=InsightType.WEATHER_TRAFFIC,
                title="Poor visibility may slow traffic",
                description=(
                    "Fog or storm conditions reduce visibility, which typically "
                    "causes traffic to slow significantly, especially on motorways."
                ),
                confidence=Confidence.MEDIUM,
                data_points={"has_poor_visibility": True},
            ))

        return insights

    def _correlate_traffic_air_quality(
        self, traffic: dict, air_quality: dict
    ) -> list[Insight]:
        """Find correlations between traffic conditions and air quality."""
        insights = []

        quality = air_quality.get("quality_level", "unknown")
        has_congestion = traffic.get("has_congestion", False)
        no2 = air_quality.get("no2", 0)

        # Heavy traffic + poor air quality
        if has_congestion and quality in ("unhealthy_sensitive", "unhealthy"):
            insights.append(Insight(
                insight_type=InsightType.AIR_QUALITY_TRAFFIC,
                title="Traffic congestion worsening air quality",
                description=(
                    f"Heavy traffic congestion correlates with elevated pollution levels. "
                    f"NO₂ at {no2:.1f} µg/m³ is typical of high-traffic conditions. "
                    f"Idling vehicles in congested areas are a major source of roadside pollution."
                ),
                confidence=Confidence.HIGH,
                data_points={
                    "has_congestion": True,
                    "quality_level": quality,
                    "no2": no2,
                },
            ))

        # Congestion but OK air quality
        elif has_congestion and quality == "good":
            insights.append(Insight(
                insight_type=InsightType.AIR_QUALITY_TRAFFIC,
                title="Air quality holding despite congestion",
                description=(
                    "Despite traffic congestion, air quality remains good. "
                    "This may be due to favorable wind dispersing pollutants "
                    "or the congestion being localized."
                ),
                confidence=Confidence.MEDIUM,
                data_points={"has_congestion": True, "quality_level": "good"},
            ))

        # High NO2 specifically
        if no2 > 40:
            insights.append(Insight(
                insight_type=InsightType.AIR_QUALITY_TRAFFIC,
                title="Elevated NO₂ levels",
                description=(
                    f"NO₂ at {no2:.1f} µg/m³ exceeds the WHO annual guideline of 10 µg/m³. "
                    f"This is common in London due to diesel vehicle emissions and "
                    f"is often worse during congested conditions."
                ),
                confidence=Confidence.HIGH,
                data_points={"no2": no2},
            ))

        return insights

    def _correlate_weather_air_quality(
        self, weather: dict, air_quality: dict
    ) -> list[Insight]:
        """Find correlations between weather and air quality."""
        insights = []

        quality = air_quality.get("quality_level", "unknown")
        is_raining = weather.get("is_raining", False)
        humidity = weather.get("humidity_percent", 0)
        wind = weather.get("wind_speed_kmh", 0)

        # Rain washes pollutants
        if is_raining and quality == "good":
            insights.append(Insight(
                insight_type=InsightType.WEATHER_AIR_QUALITY,
                title="Rainfall helping air quality",
                description=(
                    "Current rainfall is helping keep air quality good by "
                    "washing particulate matter out of the atmosphere."
                ),
                confidence=Confidence.MEDIUM,
                data_points={"is_raining": True, "quality_level": "good"},
            ))

        # Low wind + poor air quality = stagnation
        if wind < 10 and quality in ("moderate", "unhealthy_sensitive", "unhealthy"):
            insights.append(Insight(
                insight_type=InsightType.WEATHER_AIR_QUALITY,
                title="Low wind trapping pollutants",
                description=(
                    f"Light winds ({wind:.0f} km/h) are allowing pollutants to accumulate. "
                    f"Higher wind speeds would help disperse pollution more effectively."
                ),
                confidence=Confidence.MEDIUM,
                data_points={"wind_speed_kmh": wind, "quality_level": quality},
            ))

        # High humidity can worsen perceived air quality
        if humidity > 80 and quality != "good":
            insights.append(Insight(
                insight_type=InsightType.WEATHER_AIR_QUALITY,
                title="High humidity amplifying pollution effects",
                description=(
                    f"Humidity at {humidity}% can make pollution feel worse — "
                    f"moisture helps particulates stay suspended and can "
                    f"exacerbate respiratory symptoms."
                ),
                confidence=Confidence.LOW,
                data_points={"humidity_percent": humidity, "quality_level": quality},
            ))

        return insights

    def _analyze_congestion_patterns(self, traffic: dict) -> list[Insight]:
        """Analyze traffic data for notable patterns."""
        insights = []
        points = traffic.get("points", [])

        # Find worst congestion point
        congested_points = [
            p for p in points if p.get("congestion_ratio", 1) < 0.65
        ]

        if congested_points:
            worst = min(congested_points, key=lambda p: p.get("congestion_ratio", 1))
            location = worst.get("location", "an area")
            ratio = worst.get("congestion_ratio", 0)
            speed = worst.get("current_speed", 0)

            insights.append(Insight(
                insight_type=InsightType.CONGESTION_PATTERN,
                title=f"Worst congestion at {location}",
                description=(
                    f"Traffic at {location} is at {ratio:.0%} of free-flow speed "
                    f"({speed:.0f} km/h), indicating {'heavy' if ratio < 0.4 else 'moderate'} congestion. "
                    f"{len(congested_points)} location(s) currently experiencing notable congestion."
                ),
                confidence=Confidence.HIGH,
                data_points={
                    "location": location,
                    "congestion_ratio": ratio,
                    "current_speed": speed,
                    "congested_locations": len(congested_points),
                },
            ))

        # All clear pattern
        if points and not congested_points:
            avg_ratio = sum(p.get("congestion_ratio", 1) for p in points) / len(points)
            insights.append(Insight(
                insight_type=InsightType.CONGESTION_PATTERN,
                title="Traffic flowing well across monitored areas",
                description=(
                    f"All {len(points)} monitored location(s) show acceptable traffic flow "
                    f"(average {avg_ratio:.0%} of free-flow speed)."
                ),
                confidence=Confidence.HIGH,
                data_points={"avg_congestion_ratio": avg_ratio, "total_points": len(points)},
            ))

        # Incidents correlating with congestion
        incident_count = traffic.get("incident_count", 0)
        disruption_count = traffic.get("disruption_count", 0)

        if incident_count > 5 and congested_points:
            insights.append(Insight(
                insight_type=InsightType.CONGESTION_PATTERN,
                title="Multiple incidents contributing to congestion",
                description=(
                    f"{incident_count} active traffic incidents are likely contributing "
                    f"to the congestion observed at {len(congested_points)} location(s)."
                ),
                confidence=Confidence.HIGH,
                data_points={
                    "incident_count": incident_count,
                    "congested_locations": len(congested_points),
                },
            ))

        if disruption_count > 10:
            insights.append(Insight(
                insight_type=InsightType.CONGESTION_PATTERN,
                title="High number of road disruptions",
                description=(
                    f"{disruption_count} road disruptions are active across London. "
                    f"This above-average level of roadworks and closures may be "
                    f"contributing to traffic redistribution and delays."
                ),
                confidence=Confidence.MEDIUM,
                data_points={"disruption_count": disruption_count},
            ))

        return insights

    def _analyze_tube_patterns(self, tube: dict) -> list[Insight]:
        """Analyze tube status for notable patterns."""
        insights = []

        if tube.get("all_clear"):
            insights.append(Insight(
                insight_type=InsightType.SUMMARY,
                title="Tube running normally",
                description="All tube lines have good service — no disruptions.",
                confidence=Confidence.HIGH,
                data_points={"all_clear": True},
            ))
        else:
            disrupted = tube.get("disrupted_count", len(tube.get("disrupted_lines", [])))
            total = tube.get("total_lines", 11)
            if disrupted > 3:
                insights.append(Insight(
                    insight_type=InsightType.ANOMALY,
                    title="Significant tube disruptions",
                    description=(
                        f"{disrupted} of {total} tube lines are disrupted — "
                        f"this is above normal and may push more commuters onto roads, "
                        f"worsening traffic congestion."
                    ),
                    confidence=Confidence.HIGH,
                    data_points={"disrupted_count": disrupted, "total_lines": total},
                ))

        return insights


# ── Module-level convenience ──────────────────────────────────────
_engine = CorrelationEngine()


def correlate_data(tool_results: dict[str, str]) -> list[Insight]:
    """Run correlation analysis on tool results."""
    return _engine.analyze(tool_results)


def format_insights_for_llm(insights: list[Insight]) -> str:
    """Format insights into a string the LLM analyzer can use."""
    if not insights:
        return "No cross-source correlations detected."

    lines = ["CORRELATION INSIGHTS (pre-computed by analysis engine):"]
    for insight in insights:
        lines.append(insight.to_string())
    return "\n".join(lines)
