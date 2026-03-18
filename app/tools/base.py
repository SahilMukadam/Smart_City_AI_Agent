"""
Smart City AI Agent - Base Tool Class
Abstract base class for all data source wrappers.
Provides shared HTTP client, timeout handling, and response timing.
"""

import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

from app.config import get_settings
from app.models.schemas import ToolResponse

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base for all data source tools.
    Subclasses implement specific API calls; this class
    provides shared HTTP logic and error handling.
    """

    def __init__(self):
        settings = get_settings()
        self._timeout = httpx.Timeout(settings.HTTP_TIMEOUT_SECONDS)
        self._max_retries = settings.HTTP_MAX_RETRIES

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier (e.g., 'tfl', 'weather')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description for the LLM agent explaining what this tool does.
        This becomes the tool description in LangGraph.
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """List of query types this tool supports."""
        ...

    def _make_request(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> httpx.Response:
        """
        Make an HTTP GET request with timeout and retry logic.
        Raises httpx.HTTPError on failure after retries.
        """
        last_error = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = httpx.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"[{self.name}] Timeout on attempt {attempt}/{self._max_retries}: {url}"
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    raise
                logger.warning(
                    f"[{self.name}] HTTP {e.response.status_code} on attempt "
                    f"{attempt}/{self._max_retries}: {url}"
                )
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"[{self.name}] HTTP error on attempt {attempt}/{self._max_retries}: {e}"
                )

            # Brief pause before retry
            if attempt < self._max_retries:
                time.sleep(0.5 * attempt)

        raise last_error  # type: ignore[misc]

    def _build_error_response(
        self,
        query_type: str,
        error: Exception,
        source_url: str = "",
    ) -> ToolResponse:
        """Build a standardized error response."""
        return ToolResponse(
            tool_name=self.name,
            query_type=query_type,
            success=False,
            data=None,
            summary=f"Failed to fetch {query_type} data: {str(error)}",
            error=str(error),
            timestamp=datetime.now(tz=timezone.utc),
            source_url=source_url,
            response_time_ms=0.0,
        )

    def _timed_request(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> tuple[dict | list, float]:
        """
        Make a request and return (parsed_json, elapsed_ms).
        Used by subclasses to time their API calls.
        """
        start = time.perf_counter()
        response = self._make_request(url, params=params, headers=headers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return response.json(), elapsed_ms
