"""
Smart City AI Agent - Response Cache Tests

Run: pytest tests/test_cache.py -v
"""

import time
import pytest

from app.agent.cache import ResponseCache


@pytest.fixture
def cache():
    return ResponseCache(default_ttl=10, max_entries=50)


class TestCacheBasics:

    def test_set_and_get(self, cache):
        cache.set("get_tube_status", {}, "tube data here")
        result = cache.get("get_tube_status", {})
        assert result == "tube data here"

    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent", {}) is None

    def test_different_args_different_keys(self, cache):
        cache.set("get_traffic_flow", {"lat": 51.50}, "data A")
        cache.set("get_traffic_flow", {"lat": 51.53}, "data B")

        assert cache.get("get_traffic_flow", {"lat": 51.50}) == "data A"
        assert cache.get("get_traffic_flow", {"lat": 51.53}) == "data B"

    def test_same_args_same_key(self, cache):
        cache.set("get_weather", {"lat": 51.50, "lon": -0.12}, "data")
        result = cache.get("get_weather", {"lat": 51.50, "lon": -0.12})
        assert result == "data"

    def test_overwrite_existing(self, cache):
        cache.set("tool", {}, "old data")
        cache.set("tool", {}, "new data")
        assert cache.get("tool", {}) == "new data"


class TestCacheTTL:

    def test_not_expired(self, cache):
        cache.set("tool", {}, "data")
        assert cache.get("tool", {}) == "data"

    def test_expired_returns_none(self):
        cache = ResponseCache(default_ttl=1)
        cache.set("tool", {}, "data")

        # Manually expire
        key = cache._make_key("tool", {})
        cache._cache[key].created_at = time.monotonic() - 2

        assert cache.get("tool", {}) is None

    def test_custom_ttl(self, cache):
        cache.set("tool", {}, "data", ttl=1)
        key = cache._make_key("tool", {})
        assert cache._cache[key].ttl_seconds == 1


class TestCacheErrors:

    def test_does_not_cache_errors(self, cache):
        cache.set("tool", {}, "ERROR: timeout")
        assert cache.get("tool", {}) is None

    def test_does_not_cache_error_prefix(self, cache):
        cache.set("tool", {}, "ERROR: 500 server error")
        assert cache.get("tool", {}) is None


class TestCacheInvalidation:

    def test_invalidate_all(self, cache):
        cache.set("tool1", {}, "data1")
        cache.set("tool2", {}, "data2")
        cache.invalidate()
        assert cache.get("tool1", {}) is None
        assert cache.get("tool2", {}) is None

    def test_invalidate_specific(self, cache):
        cache.set("get_tube_status", {}, "tube data")
        cache.set("get_weather", {}, "weather data")
        cache.invalidate("get_tube_status")

        # Tube invalidated but weather still there
        assert cache.get("get_weather", {}) == "weather data"


class TestCacheStats:

    def test_stats_initial(self, cache):
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["current_entries"] == 0

    def test_stats_after_operations(self, cache):
        cache.set("tool", {}, "data")
        cache.get("tool", {})  # hit
        cache.get("tool", {})  # hit
        cache.get("missing", {})  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["hit_rate_percent"] == pytest.approx(66.7, abs=0.1)

    def test_stats_current_entries(self, cache):
        cache.set("a", {}, "1")
        cache.set("b", {}, "2")
        assert cache.get_stats()["current_entries"] == 2


class TestCacheCapacity:

    def test_eviction_at_capacity(self):
        cache = ResponseCache(default_ttl=300, max_entries=3)
        cache.set("a", {}, "1")
        cache.set("b", {}, "2")
        cache.set("c", {}, "3")
        cache.set("d", {}, "4")  # Should evict oldest

        assert cache.get_stats()["current_entries"] == 3
        assert cache.get("d", {}) == "4"

    def test_expired_evicted_before_capacity(self):
        cache = ResponseCache(default_ttl=1, max_entries=3)
        cache.set("old1", {}, "data1")
        cache.set("old2", {}, "data2")

        # Expire them
        for entry in cache._cache.values():
            entry.created_at = time.monotonic() - 2

        cache.set("new1", {}, "data3")
        cache.set("new2", {}, "data4")
        cache.set("new3", {}, "data5")

        # Expired entries should have been cleaned up
        assert cache.get_stats()["current_entries"] == 3
