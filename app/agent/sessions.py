"""
Smart City AI Agent - Session Manager
In-memory session store for conversation history.
Each session tracks message history so the agent can handle follow-ups.

Sessions auto-expire after a configurable TTL (default: 30 min).
"""

import logging
import uuid
import time
from datetime import datetime, timezone
from threading import Lock

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


class Session:
    """A single conversation session with message history."""

    def __init__(self, session_id: str, ttl_seconds: int = 1800):
        self.session_id = session_id
        self.created_at = datetime.now(tz=timezone.utc)
        self.last_active = time.monotonic()
        self.ttl_seconds = ttl_seconds
        self.messages: list[BaseMessage] = []
        self.metadata: dict = {
            "total_queries": 0,
            "tools_used": [],
        }

    @property
    def is_expired(self) -> bool:
        """Check if session has exceeded its TTL."""
        return (time.monotonic() - self.last_active) > self.ttl_seconds

    def touch(self):
        """Update last active timestamp."""
        self.last_active = time.monotonic()

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(HumanMessage(content=content))
        self.metadata["total_queries"] += 1
        self.touch()

    def add_ai_message(self, content: str):
        """Add an AI response to history."""
        self.messages.append(AIMessage(content=content))
        self.touch()

    def add_tools_used(self, tools: list[str]):
        """Track which tools were used in this session."""
        for tool in tools:
            if tool not in self.metadata["tools_used"]:
                self.metadata["tools_used"].append(tool)

    def get_messages(self) -> list[BaseMessage]:
        """Get full message history for this session."""
        return self.messages.copy()

    def get_recent_messages(self, max_messages: int = 10) -> list[BaseMessage]:
        """
        Get recent messages, capped to avoid context overflow.
        Always includes the latest user message.
        Keeps pairs (user + AI) together.
        """
        if len(self.messages) <= max_messages:
            return self.messages.copy()

        # Take the last N messages, ensuring we start with a user message
        recent = self.messages[-max_messages:]
        if recent and isinstance(recent[0], AIMessage):
            # Drop the orphaned AI message so we start with user context
            recent = recent[1:]

        return recent

    def get_summary(self) -> dict:
        """Get session summary for API responses."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "total_messages": len(self.messages),
            "total_queries": self.metadata["total_queries"],
            "tools_used": self.metadata["tools_used"],
            "is_expired": self.is_expired,
        }


class SessionManager:
    """
    Thread-safe in-memory session store.
    Handles creation, retrieval, cleanup, and expiration of sessions.
    """

    def __init__(self, session_ttl_seconds: int = 1800, max_sessions: int = 100):
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        self._session_ttl = session_ttl_seconds
        self._max_sessions = max_sessions

    def create_session(self) -> Session:
        """Create a new session with a unique ID."""
        session_id = str(uuid.uuid4())[:8]  # Short IDs for convenience

        with self._lock:
            # Cleanup expired sessions if we're at capacity
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_expired()

            # If still at capacity after cleanup, remove oldest
            if len(self._sessions) >= self._max_sessions:
                oldest_id = min(
                    self._sessions,
                    key=lambda k: self._sessions[k].last_active,
                )
                del self._sessions[oldest_id]
                logger.info(f"Evicted oldest session: {oldest_id}")

            session = Session(session_id, ttl_seconds=self._session_ttl)
            self._sessions[session_id] = session
            logger.info(f"Created session: {session_id}")

        return session

    def get_session(self, session_id: str) -> Session | None:
        """
        Get a session by ID.
        Returns None if not found or expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                logger.info(f"Session expired: {session_id}")
                return None
            return session

    def get_or_create_session(self, session_id: str | None = None) -> Session:
        """
        Get existing session or create new one.
        If session_id is None or not found, creates a new session.
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        return self.create_session()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False

    def list_sessions(self) -> list[dict]:
        """List all active (non-expired) sessions."""
        with self._lock:
            self._cleanup_expired()
            return [
                session.get_summary()
                for session in self._sessions.values()
            ]

    def _cleanup_expired(self):
        """Remove all expired sessions. Must be called with lock held."""
        expired = [
            sid for sid, s in self._sessions.items()
            if s.is_expired
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired session(s)")

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        with self._lock:
            return sum(
                1 for s in self._sessions.values()
                if not s.is_expired
            )
