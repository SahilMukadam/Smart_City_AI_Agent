"""
Smart City AI Agent - Session Manager Tests
Tests session creation, retrieval, expiration, and cleanup.

Run: pytest tests/test_sessions.py -v
"""

import time
from unittest.mock import patch
import pytest

from app.agent.sessions import Session, SessionManager


# ── Session Tests ─────────────────────────────────────────────────


class TestSession:
    """Test individual session behavior."""

    def test_session_creation(self):
        session = Session("test-001", ttl_seconds=300)
        assert session.session_id == "test-001"
        assert session.ttl_seconds == 300
        assert len(session.messages) == 0
        assert session.metadata["total_queries"] == 0

    def test_add_user_message(self):
        session = Session("test-001")
        session.add_user_message("How's the tube?")
        assert len(session.messages) == 1
        assert session.messages[0].content == "How's the tube?"
        assert session.messages[0].type == "human"
        assert session.metadata["total_queries"] == 1

    def test_add_ai_message(self):
        session = Session("test-001")
        session.add_ai_message("All lines have good service.")
        assert len(session.messages) == 1
        assert session.messages[0].type == "ai"

    def test_multi_turn_conversation(self):
        session = Session("test-001")
        session.add_user_message("How's traffic?")
        session.add_ai_message("Traffic is moderate in Central London.")
        session.add_user_message("What about Canary Wharf?")
        session.add_ai_message("Canary Wharf has light congestion.")

        assert len(session.messages) == 4
        assert session.metadata["total_queries"] == 2

    def test_tools_tracking(self):
        session = Session("test-001")
        session.add_tools_used(["get_tube_status", "get_current_weather"])
        session.add_tools_used(["get_traffic_flow", "get_tube_status"])  # duplicate

        assert "get_tube_status" in session.metadata["tools_used"]
        assert "get_current_weather" in session.metadata["tools_used"]
        assert "get_traffic_flow" in session.metadata["tools_used"]
        # No duplicates
        assert len(session.metadata["tools_used"]) == 3

    def test_get_recent_messages_under_limit(self):
        session = Session("test-001")
        session.add_user_message("Q1")
        session.add_ai_message("A1")
        session.add_user_message("Q2")

        recent = session.get_recent_messages(max_messages=10)
        assert len(recent) == 3

    def test_get_recent_messages_over_limit(self):
        session = Session("test-001")
        for i in range(10):
            session.add_user_message(f"Question {i}")
            session.add_ai_message(f"Answer {i}")

        # 20 messages total, limit to 6
        recent = session.get_recent_messages(max_messages=6)
        assert len(recent) <= 6

    def test_get_recent_messages_starts_with_user(self):
        session = Session("test-001")
        session.add_user_message("Q1")
        session.add_ai_message("A1")
        session.add_user_message("Q2")
        session.add_ai_message("A2")
        session.add_user_message("Q3")
        session.add_ai_message("A3")
        session.add_user_message("Q4")

        recent = session.get_recent_messages(max_messages=4)
        # Should start with a user (human) message
        assert recent[0].type == "human"

    def test_session_not_expired(self):
        session = Session("test-001", ttl_seconds=300)
        assert session.is_expired is False

    def test_session_expired(self):
        session = Session("test-001", ttl_seconds=1)
        # Manually set last_active to the past
        session.last_active = time.monotonic() - 2
        assert session.is_expired is True

    def test_touch_resets_expiry(self):
        session = Session("test-001", ttl_seconds=1)
        session.last_active = time.monotonic() - 0.5
        session.touch()
        assert session.is_expired is False

    def test_get_summary(self):
        session = Session("test-001")
        session.add_user_message("Hi")
        session.add_ai_message("Hello!")
        session.add_tools_used(["get_tube_status"])

        summary = session.get_summary()
        assert summary["session_id"] == "test-001"
        assert summary["total_messages"] == 2
        assert summary["total_queries"] == 1
        assert "get_tube_status" in summary["tools_used"]


# ── SessionManager Tests ──────────────────────────────────────────


class TestSessionManager:
    """Test session manager operations."""

    def test_create_session(self):
        manager = SessionManager()
        session = manager.create_session()
        assert session.session_id is not None
        assert len(session.session_id) == 8

    def test_get_session(self):
        manager = SessionManager()
        session = manager.create_session()
        retrieved = manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_nonexistent_session(self):
        manager = SessionManager()
        assert manager.get_session("nonexistent") is None

    def test_get_expired_session(self):
        manager = SessionManager(session_ttl_seconds=1)
        session = manager.create_session()
        session.last_active = time.monotonic() - 2
        assert manager.get_session(session.session_id) is None

    def test_get_or_create_existing(self):
        manager = SessionManager()
        session1 = manager.create_session()
        session2 = manager.get_or_create_session(session1.session_id)
        assert session1.session_id == session2.session_id

    def test_get_or_create_new(self):
        manager = SessionManager()
        session = manager.get_or_create_session(None)
        assert session is not None
        assert len(session.session_id) == 8

    def test_get_or_create_expired(self):
        manager = SessionManager(session_ttl_seconds=1)
        session1 = manager.create_session()
        old_id = session1.session_id
        session1.last_active = time.monotonic() - 2

        session2 = manager.get_or_create_session(old_id)
        assert session2.session_id != old_id  # New session created

    def test_delete_session(self):
        manager = SessionManager()
        session = manager.create_session()
        assert manager.delete_session(session.session_id) is True
        assert manager.get_session(session.session_id) is None

    def test_delete_nonexistent(self):
        manager = SessionManager()
        assert manager.delete_session("nonexistent") is False

    def test_list_sessions(self):
        manager = SessionManager()
        manager.create_session()
        manager.create_session()
        sessions = manager.list_sessions()
        assert len(sessions) == 2

    def test_list_excludes_expired(self):
        manager = SessionManager(session_ttl_seconds=1)
        s1 = manager.create_session()
        manager.create_session()  # s2 is fresh
        s1.last_active = time.monotonic() - 2

        sessions = manager.list_sessions()
        assert len(sessions) == 1

    def test_active_count(self):
        manager = SessionManager()
        manager.create_session()
        manager.create_session()
        assert manager.active_count == 2

    def test_max_sessions_eviction(self):
        manager = SessionManager(max_sessions=3)
        s1 = manager.create_session()
        s2 = manager.create_session()
        s3 = manager.create_session()

        # At capacity — creating a 4th should evict the oldest
        s1.last_active = time.monotonic() - 100  # Make s1 the oldest
        s4 = manager.create_session()

        assert manager.get_session(s1.session_id) is None  # Evicted
        assert manager.get_session(s4.session_id) is not None
