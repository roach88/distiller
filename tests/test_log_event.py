"""Tests for the log_event module."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.log_event import (
    log_session_start,
    log_session_end,
    log_turn,
    log_tool_call,
    process_event,
    get_db_path,
)


class TestSessionLogging:
    """Tests for session logging."""

    def test_log_session_start(self, initialized_db, monkeypatch):
        """Should create a new session record."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        session_id = log_session_start({
            "session_id": "test-session-123",
            "project_path": "/path/to/project",
            "project_name": "test-project",
            "git_branch": "main",
            "git_repo": "https://github.com/test/repo",
        })

        assert session_id == "test-session-123"

        # Verify in database
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test-session-123"
        assert row[3] == "/path/to/project"
        assert row[4] == "test-project"

    def test_log_session_start_generates_id(self, initialized_db, monkeypatch):
        """Should generate session ID if not provided."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        session_id = log_session_start({
            "project_name": "test-project",
        })

        assert session_id is not None
        assert len(session_id) > 0

    def test_log_session_end(self, initialized_db, monkeypatch):
        """Should update session with end time and reason."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Create session first
        session_id = log_session_start({"session_id": "test-end-session"})

        # End the session
        log_session_end({
            "session_id": session_id,
            "exit_reason": "user_exit",
        })

        # Verify
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("SELECT ended_at, exit_reason FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        assert row[0] is not None  # ended_at should be set
        assert row[1] == "user_exit"


class TestTurnLogging:
    """Tests for turn logging."""

    def test_log_user_turn(self, initialized_db, monkeypatch):
        """Should log a user turn."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Create session first
        session_id = log_session_start({"session_id": "turn-test-session"})

        # Log turn
        turn_id = log_turn({
            "session_id": session_id,
            "role": "user",
            "content": "Hello, Claude!",
            "token_count": 3,
        })

        assert turn_id is not None

        # Verify
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[1] == session_id
        assert row[2] == 1  # turn_number
        assert row[4] == "user"
        assert row[5] == "Hello, Claude!"

    def test_log_assistant_turn(self, initialized_db, monkeypatch):
        """Should log an assistant turn."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        session_id = log_session_start({"session_id": "assist-turn-session"})

        turn_id = log_turn({
            "session_id": session_id,
            "role": "assistant",
            "content": "Hello! How can I help?",
        })

        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("SELECT role, content FROM turns WHERE id = ?", (turn_id,))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "assistant"
        assert row[1] == "Hello! How can I help?"

    def test_turn_numbers_increment(self, initialized_db, monkeypatch):
        """Turn numbers should increment within a session."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        session_id = log_session_start({"session_id": "multi-turn-session"})

        log_turn({"session_id": session_id, "role": "user", "content": "First"})
        log_turn({"session_id": session_id, "role": "assistant", "content": "Second"})
        log_turn({"session_id": session_id, "role": "user", "content": "Third"})

        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute(
            "SELECT turn_number FROM turns WHERE session_id = ? ORDER BY turn_number",
            (session_id,)
        )
        turn_numbers = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert turn_numbers == [1, 2, 3]

    def test_log_turn_requires_session(self, initialized_db, monkeypatch):
        """Should raise error if session_id not provided."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        with pytest.raises(ValueError, match="session_id is required"):
            log_turn({"role": "user", "content": "No session"})


class TestToolCallLogging:
    """Tests for tool call logging."""

    def test_log_tool_call(self, initialized_db, monkeypatch):
        """Should log a tool call."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        session_id = log_session_start({"session_id": "tool-session"})
        turn_id = log_turn({"session_id": session_id, "role": "assistant", "content": "Using tool..."})

        tool_call_id = log_tool_call({
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_name": "Read",
            "tool_input": {"file_path": "/test.txt"},
            "tool_output": {"content": "file contents"},
            "duration_ms": 50,
            "success": True,
        })

        assert tool_call_id is not None

        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[3] == "Read"
        assert '"/test.txt"' in row[4]  # JSON input
        assert row[6] == 50  # duration_ms
        assert row[7] == 1  # success (True)

    def test_log_tool_call_finds_turn(self, initialized_db, monkeypatch):
        """Should find the last assistant turn if turn_id not provided."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        session_id = log_session_start({"session_id": "tool-find-turn"})
        log_turn({"session_id": session_id, "role": "assistant", "content": "I'll read that file"})

        # Don't provide turn_id
        tool_call_id = log_tool_call({
            "session_id": session_id,
            "tool_name": "Read",
        })

        assert tool_call_id is not None


class TestProcessEvent:
    """Tests for the main event processing function."""

    def test_process_session_start(self, initialized_db, monkeypatch):
        """Should process session_start event."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        result = process_event({
            "event": "session_start",
            "session_id": "process-test-1",
            "project_name": "test",
        })

        assert result["status"] == "ok"
        assert result["session_id"] == "process-test-1"

    def test_process_session_end(self, initialized_db, monkeypatch):
        """Should process session_end event."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Create session first
        process_event({
            "event": "session_start",
            "session_id": "process-test-2",
        })

        result = process_event({
            "event": "session_end",
            "session_id": "process-test-2",
            "exit_reason": "completed",
        })

        assert result["status"] == "ok"

    def test_process_user_message(self, initialized_db, monkeypatch):
        """Should process user_message event."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        process_event({
            "event": "session_start",
            "session_id": "process-test-3",
        })

        result = process_event({
            "event": "user_message",
            "session_id": "process-test-3",
            "content": "Test message",
        })

        assert result["status"] == "ok"
        assert "turn_id" in result

    def test_process_assistant_message(self, initialized_db, monkeypatch):
        """Should process assistant_message event."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        process_event({
            "event": "session_start",
            "session_id": "process-test-4",
        })

        result = process_event({
            "event": "assistant_message",
            "session_id": "process-test-4",
            "content": "Response",
        })

        assert result["status"] == "ok"

    def test_process_tool_call(self, initialized_db, monkeypatch):
        """Should process tool_call event."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        process_event({
            "event": "session_start",
            "session_id": "process-test-5",
        })
        process_event({
            "event": "assistant_message",
            "session_id": "process-test-5",
            "content": "Using tool",
        })

        result = process_event({
            "event": "tool_call",
            "session_id": "process-test-5",
            "tool_name": "Bash",
        })

        assert result["status"] == "ok"
        assert "tool_call_id" in result

    def test_process_unknown_event(self, initialized_db, monkeypatch):
        """Should handle unknown event types gracefully."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        result = process_event({
            "event": "unknown_event",
        })

        assert result["status"] == "ignored"

    def test_process_missing_event(self, initialized_db, monkeypatch):
        """Should handle missing event type."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        result = process_event({})

        assert result["status"] == "error"

    def test_process_handles_errors(self, initialized_db, monkeypatch):
        """Should catch and return errors gracefully."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Try to log turn without session
        result = process_event({
            "event": "user_message",
            "content": "No session",
        })

        assert result["status"] == "error"


class TestErrorHandling:
    """Tests for error handling (hook must not break Claude Code)."""

    def test_missing_database(self, temp_db_path, monkeypatch):
        """Should handle missing database gracefully."""
        # Point to non-existent db
        monkeypatch.setenv("DISTILLER_DB_PATH", str(temp_db_path / "nonexistent.db"))

        result = process_event({
            "event": "session_start",
            "project_name": "test",
        })

        assert result["status"] == "error"
        assert "not found" in result["message"].lower() or "no such" in result["message"].lower()
