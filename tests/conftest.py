"""
Smart City AI Agent - Test Configuration & Fixtures
Shared fixtures for all test files.
Mock API responses to avoid hitting real APIs during testing.
"""

import pytest

# ── Mock TfL API Responses ────────────────────────────────────────

MOCK_TUBE_STATUS_GOOD = [
    {
        "id": "bakerloo",
        "name": "Bakerloo",
        "lineStatuses": [
            {
                "statusSeverity": 10,
                "statusSeverityDescription": "Good Service",
                "reason": None,
            }
        ],
    },
    {
        "id": "central",
        "name": "Central",
        "lineStatuses": [
            {
                "statusSeverity": 10,
                "statusSeverityDescription": "Good Service",
                "reason": None,
            }
        ],
    },
    {
        "id": "victoria",
        "name": "Victoria",
        "lineStatuses": [
            {
                "statusSeverity": 10,
                "statusSeverityDescription": "Good Service",
                "reason": None,
            }
        ],
    },
]

MOCK_TUBE_STATUS_DISRUPTED = [
    {
        "id": "bakerloo",
        "name": "Bakerloo",
        "lineStatuses": [
            {
                "statusSeverity": 10,
                "statusSeverityDescription": "Good Service",
                "reason": None,
            }
        ],
    },
    {
        "id": "central",
        "name": "Central",
        "lineStatuses": [
            {
                "statusSeverity": 5,
                "statusSeverityDescription": "Minor Delays",
                "reason": "Signal failure at Oxford Circus",
            }
        ],
    },
    {
        "id": "northern",
        "name": "Northern",
        "lineStatuses": [
            {
                "statusSeverity": 4,
                "statusSeverityDescription": "Part Closure",
                "reason": "Planned engineering works between Camden Town and High Barnet",
            }
        ],
    },
]

MOCK_ROAD_DISRUPTIONS = [
    {
        "id": "TIMS-12345",
        "severity": "Serious",
        "category": "RoadWorks",
        "location": "A1 / Great North Road near Archway",
        "comments": "Scheduled roadworks with lane closures",
        "corridorIds": ["A1"],
        "startDate": "2025-03-01T00:00:00Z",
        "endDate": "2025-04-15T23:59:59Z",
    },
    {
        "id": "TIMS-12346",
        "severity": "Moderate",
        "category": "PlannedWork",
        "location": "A40 / Western Avenue near Acton",
        "comments": "Utility works causing delays",
        "corridorIds": ["A40"],
        "startDate": "2025-03-10T06:00:00Z",
        "endDate": "2025-03-20T18:00:00Z",
    },
    {
        "id": "TIMS-12347",
        "severity": "Minimal",
        "category": "Incident",
        "location": "A205 / South Circular near Lewisham",
        "comments": "Minor incident cleared, residual delays",
        "corridorIds": ["A205"],
        "startDate": None,
        "endDate": None,
    },
]

MOCK_ROAD_STATUS = [
    {
        "id": "A1",
        "displayName": "A1 Road",
        "statusSeverity": "Good",
        "statusSeverityDescription": "No exceptional delays",
    },
    {
        "id": "A2",
        "displayName": "A2 Road",
        "statusSeverity": "Serious",
        "statusSeverityDescription": "Serious delays reported on this road",
    },
    {
        "id": "A40",
        "displayName": "A40 Road",
        "statusSeverity": "Good",
        "statusSeverityDescription": "No exceptional delays",
    },
]


@pytest.fixture
def mock_tube_good():
    return MOCK_TUBE_STATUS_GOOD


@pytest.fixture
def mock_tube_disrupted():
    return MOCK_TUBE_STATUS_DISRUPTED


@pytest.fixture
def mock_disruptions():
    return MOCK_ROAD_DISRUPTIONS


@pytest.fixture
def mock_road_status():
    return MOCK_ROAD_STATUS
