"""
Smart City AI Agent - Response Cache
TTL-based cache for API tool responses.
Prevents redundant API calls when the same data is requested
within a short time window.

Thread-safe for use with parallel tool execution.
"""

import time
import logging
import hashlib
from threading import Lock
from dataclasses import dataclass, field

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached response."""
    key: str
    value: str
    created_at: float  # monotonic time
    ttl_seconds: int
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


class ResponseCache:
    """
    Thread-safe TTL cache for tool responses.

    Usage:
        cache = ResponseCache(default_ttl=300)
        cached = cache.get("get_tube_status", {})
        if cached:
            return cached
        result = call_api()
        cache.set("get_tube_status", {}, result)
    """

    def __init__(self, default_ttl: int | None = None, max_entries: int = 200):
        settings = get_settings()
        self._default_ttl = default_ttl or settings.CACHE_TTL_SECONDS
        self._max_entries = max_entries
        self._cache: dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    def _make_key(self, tool_name: str, args: dict) -> str:
        """Generate a cache key from tool name and arguments."""
        # Sort args for consistent hashing
        args_str = str(sorted(args.items())) if args else ""
        raw = f"{tool_name}:{args_str}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def get(self, tool_name: str, args: dict) -> str | None:
        """
        Get cached response if available and not expired.
        Returns None on miss.
        """
        key = self._make_key(tool_name, args)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                logger.debug(f"Cache expired: {tool_name}")
                return None

            entry.hit_count += 1
            self._stats["hits"] += 1
            logger.info(
                f"📦 Cache hit: {tool_name} (age: {entry.age_seconds:.0f}s, "
                f"hits: {entry.hit_count})"
            )
            return entry.value

    def set(
        self,
        tool_name: str,
        args: dict,
        value: str,
        ttl: int | None = None,
    ):
        """Cache a tool response."""
        # Don't cache errors
        if value.startswith("ERROR"):
            return

        key = self._make_key(tool_name, args)
        ttl = ttl or self._default_ttl

        with self._lock:
            # Evict expired entries if at capacity
            if len(self._cache) >= self._max_entries:
                self._evict_expired()

            # If still at capacity, remove least recently used
            if len(self._cache) >= self._max_entries:
                oldest_key = min(
                    self._cache,
                    key=lambda k: self._cache[k].created_at,
                )
                del self._cache[oldest_key]
                self._stats["evictions"] += 1

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.monotonic(),
                ttl_seconds=ttl,
            )
            self._stats["sets"] += 1
            logger.debug(f"Cache set: {tool_name} (TTL: {ttl}s)")

    def invalidate(self, tool_name: str | None = None):
        """
        Invalidate cache entries.
        If tool_name is None, clears entire cache.
        """
        with self._lock:
            if tool_name is None:
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cache cleared: {count} entries")
            else:
                # Remove all entries for this tool
                to_remove = [
                    k for k, v in self._cache.items()
                    if tool_name in v.key
                ]
                for k in to_remove:
                    del self._cache[k]
                logger.info(f"Cache invalidated for {tool_name}: {len(to_remove)} entries")

    def _evict_expired(self):
        """Remove expired entries. Must be called with lock held."""
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
        if expired:
            self._stats["evictions"] += len(expired)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
            return {
                **self._stats,
                "current_entries": len(self._cache),
                "hit_rate_percent": round(hit_rate, 1),
            }
