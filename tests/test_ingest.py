"""Tests for the ingest module."""

import json
import sqlite3
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest import (
    decode_project_path,
    extract_project_name,
    find_session_files,
    parse_session_file,
    session_exists,
    insert_session,
    insert_turns_and_tools,
    ingest_session_file,
    ingest_backfill,
    ingest_single_session,
    IngestStats,
)


class TestPathDecoding:
    """Tests for path encoding/decoding."""

    def test_decode_project_path(self):
        """Should decode Claude's path encoding."""
        encoded = "-Users-tyler-dev-myproject"
        decoded = decode_project_path(encoded)
        assert decoded == "/Users/tyler/dev/myproject"

    def test_decode_simple_path(self):
        """Should handle simple paths."""
        encoded = "-Users-tyler"
        decoded = decode_project_path(encoded)
        assert decoded == "/Users/tyler"

    def test_extract_project_name(self):
        """Should extract last path component."""
        path = "/Users/tyler/dev/myproject"
        name = extract_project_name(path)
        assert name == "myproject"


class TestSessionFileFinding:
    """Tests for finding session files."""

    def test_find_session_files(self, temp_claude_dir):
        """Should find session JSONL files."""
        files = list(find_session_files(temp_claude_dir))
        assert len(files) == 2  # main-session.jsonl and another-session.jsonl

    def test_find_session_files_with_filter(self, temp_claude_dir):
        """Should filter by project name."""
        files = list(find_session_files(temp_claude_dir, project_filter="myproject"))
        assert len(files) == 2

        files = list(find_session_files(temp_claude_dir, project_filter="nonexistent"))
        assert len(files) == 0

    def test_skip_agent_files(self, temp_claude_dir):
        """Should skip agent-*.jsonl files."""
        files = list(find_session_files(temp_claude_dir))
        filenames = [f.name for f in files]
        assert not any(f.startswith("agent-") for f in filenames)


class TestSessionParsing:
    """Tests for parsing session files."""

    def test_parse_session_file(self, temp_session_file):
        """Should parse session JSONL correctly."""
        data = parse_session_file(temp_session_file)

        assert data["session_id"] == "test-session-123"
        assert data["project_path"] == "/test/project"
        assert data["project_name"] == "project"
        assert data["git_branch"] == "main"
        assert len(data["messages"]) >= 2

    def test_parse_extracts_messages(self, temp_session_file):
        """Should extract user and assistant messages."""
        data = parse_session_file(temp_session_file)

        roles = [m["role"] for m in data["messages"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_parse_extracts_tool_uses(self, temp_session_file):
        """Should extract tool uses from assistant messages."""
        data = parse_session_file(temp_session_file)

        # Find assistant message with tool use
        assistant_msgs = [m for m in data["messages"] if m["role"] == "assistant"]
        tool_uses = []
        for msg in assistant_msgs:
            tool_uses.extend(msg.get("tool_uses", []))

        assert len(tool_uses) >= 1
        assert tool_uses[0]["name"] == "Read"

    def test_parse_handles_string_content(self, tmp_path):
        """Should handle messages with string content."""
        session_file = tmp_path / "string-content.jsonl"
        session_file.write_text(json.dumps({
            "type": "user",
            "sessionId": "string-session",
            "cwd": "/test",
            "timestamp": "2025-01-01T00:00:00Z",
            "message": {
                "role": "user",
                "content": "Simple string content"
            }
        }))

        data = parse_session_file(session_file)
        assert data["messages"][0]["content"] == "Simple string content"


class TestDatabaseOperations:
    """Tests for database operations."""

    def test_session_exists(self, initialized_db, monkeypatch):
        """Should detect existing sessions."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        conn = sqlite3.connect(initialized_db)
        conn.execute("INSERT INTO sessions (id, started_at) VALUES ('existing', '2025-01-01')")
        conn.commit()

        assert session_exists(conn, "existing") is True
        assert session_exists(conn, "nonexistent") is False
        conn.close()

    def test_insert_session(self, initialized_db, monkeypatch):
        """Should insert a new session."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        conn = sqlite3.connect(initialized_db)
        session_data = {
            "session_id": "new-session",
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": "2025-01-01T01:00:00Z",
            "project_path": "/test/project",
            "project_name": "project",
            "git_branch": "main",
        }

        result = insert_session(conn, session_data)
        assert result is True

        # Verify in database
        cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", ("new-session",))
        row = cursor.fetchone()
        assert row is not None
        assert row[3] == "/test/project"
        conn.close()

    def test_insert_session_idempotent(self, initialized_db, monkeypatch):
        """Should skip if session already exists."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        conn = sqlite3.connect(initialized_db)
        session_data = {
            "session_id": "dup-session",
            "started_at": "2025-01-01T00:00:00Z",
        }

        # First insert
        result1 = insert_session(conn, session_data)
        assert result1 is True

        # Second insert (should skip)
        result2 = insert_session(conn, session_data)
        assert result2 is False

        # Only one row
        cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", ("dup-session",))
        assert cursor.fetchone()[0] == 1
        conn.close()

    def test_insert_turns_and_tools(self, initialized_db, monkeypatch):
        """Should insert turns and tool calls."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        conn = sqlite3.connect(initialized_db)
        conn.execute("PRAGMA foreign_keys = ON")

        # Create session first
        conn.execute("INSERT INTO sessions (id, started_at) VALUES ('turns-session', '2025-01-01')")

        messages = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": "2025-01-01T00:00:01Z",
                "tool_uses": [],
                "usage": {},
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "timestamp": "2025-01-01T00:00:02Z",
                "tool_uses": [
                    {"id": "tool-1", "name": "Read", "input": {"file": "test.txt"}}
                ],
                "usage": {"output_tokens": 10},
            },
        ]

        turns, tools = insert_turns_and_tools(conn, "turns-session", messages)
        conn.commit()

        assert turns == 2
        assert tools == 1

        # Verify turns
        cursor = conn.execute("SELECT COUNT(*) FROM turns WHERE session_id = ?", ("turns-session",))
        assert cursor.fetchone()[0] == 2

        # Verify tool calls
        cursor = conn.execute("SELECT COUNT(*) FROM tool_calls WHERE session_id = ?", ("turns-session",))
        assert cursor.fetchone()[0] == 1

        conn.close()


class TestIngestOperations:
    """Tests for full ingest operations."""

    def test_ingest_session_file(self, initialized_db, temp_session_file, monkeypatch):
        """Should ingest a single session file."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        conn = sqlite3.connect(initialized_db)
        conn.execute("PRAGMA foreign_keys = ON")
        stats = IngestStats()

        ingest_session_file(conn, temp_session_file, stats)
        conn.commit()

        assert stats.sessions_processed == 1
        assert stats.sessions_skipped == 0
        assert stats.turns_extracted >= 2
        assert len(stats.errors) == 0

        conn.close()

    def test_ingest_session_file_idempotent(self, initialized_db, temp_session_file, monkeypatch):
        """Should skip already ingested sessions."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        conn = sqlite3.connect(initialized_db)
        conn.execute("PRAGMA foreign_keys = ON")

        # First ingest
        stats1 = IngestStats()
        ingest_session_file(conn, temp_session_file, stats1)
        conn.commit()

        # Second ingest
        stats2 = IngestStats()
        ingest_session_file(conn, temp_session_file, stats2)
        conn.commit()

        assert stats1.sessions_processed == 1
        assert stats2.sessions_processed == 0
        assert stats2.sessions_skipped == 1

        conn.close()

    def test_ingest_single_session(self, initialized_db, temp_session_file, monkeypatch):
        """Should ingest a single session via API."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        stats = ingest_single_session(temp_session_file)

        assert stats.sessions_processed == 1
        assert stats.turns_extracted >= 2

    def test_ingest_backfill(self, initialized_db, temp_claude_dir, monkeypatch):
        """Should backfill multiple sessions."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        stats = ingest_backfill(temp_claude_dir)

        assert stats.sessions_processed == 2  # main-session.jsonl and another-session.jsonl
        assert stats.turns_extracted >= 4

    def test_ingest_backfill_with_filter(self, initialized_db, temp_claude_dir, monkeypatch):
        """Should filter by project name."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Filter to myproject
        stats = ingest_backfill(temp_claude_dir, project_filter="myproject")
        assert stats.sessions_processed == 2

        # Filter to nonexistent project
        conn = sqlite3.connect(initialized_db)
        conn.execute("DELETE FROM tool_calls")
        conn.execute("DELETE FROM turns")
        conn.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()

        stats = ingest_backfill(temp_claude_dir, project_filter="nonexistent")
        assert stats.sessions_processed == 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_ingest_handles_malformed_json(self, initialized_db, tmp_path, monkeypatch):
        """Should handle malformed JSON gracefully."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Create malformed session file
        session_file = tmp_path / "malformed.jsonl"
        session_file.write_text("not valid json\n{also bad")

        stats = ingest_single_session(session_file)

        # Should report error (no valid messages)
        assert stats.sessions_processed == 0
        assert len(stats.errors) > 0

    def test_ingest_handles_empty_file(self, initialized_db, tmp_path, monkeypatch):
        """Should handle empty files gracefully."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        session_file = tmp_path / "empty.jsonl"
        session_file.write_text("")

        stats = ingest_single_session(session_file)

        assert stats.sessions_processed == 0
        assert len(stats.errors) > 0

    def test_ingest_stats_to_dict(self):
        """IngestStats should convert to dict."""
        stats = IngestStats(
            sessions_processed=5,
            sessions_skipped=2,
            turns_extracted=100,
            tool_calls_extracted=50,
            errors=["error1", "error2"],
        )

        d = stats.to_dict()
        assert d["sessions_processed"] == 5
        assert d["sessions_skipped"] == 2
        assert d["turns_extracted"] == 100
        assert d["tool_calls_extracted"] == 50
        assert len(d["errors"]) == 2


class TestDataIntegrity:
    """Tests for data capture integrity (STORY-1.4)."""

    def test_no_orphaned_turns(self, initialized_db, temp_claude_dir, monkeypatch):
        """Turns should always have valid session references."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Ingest sessions
        ingest_backfill(temp_claude_dir)

        # Check for orphaned turns
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("""
            SELECT COUNT(*) FROM turns t
            LEFT JOIN sessions s ON t.session_id = s.id
            WHERE s.id IS NULL
        """)
        orphaned = cursor.fetchone()[0]
        conn.close()

        assert orphaned == 0, "Found orphaned turns without sessions"

    def test_no_orphaned_tool_calls(self, initialized_db, temp_claude_dir, monkeypatch):
        """Tool calls should always have valid turn and session references."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        ingest_backfill(temp_claude_dir)

        conn = sqlite3.connect(initialized_db)

        # Check for tool calls without turns
        cursor = conn.execute("""
            SELECT COUNT(*) FROM tool_calls tc
            LEFT JOIN turns t ON tc.turn_id = t.id
            WHERE t.id IS NULL
        """)
        orphaned_turns = cursor.fetchone()[0]

        # Check for tool calls without sessions
        cursor = conn.execute("""
            SELECT COUNT(*) FROM tool_calls tc
            LEFT JOIN sessions s ON tc.session_id = s.id
            WHERE s.id IS NULL
        """)
        orphaned_sessions = cursor.fetchone()[0]

        conn.close()

        assert orphaned_turns == 0, "Found tool calls without turns"
        assert orphaned_sessions == 0, "Found tool calls without sessions"

    def test_turn_numbers_sequential(self, initialized_db, temp_claude_dir, monkeypatch):
        """Turn numbers should be sequential within each session."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        ingest_backfill(temp_claude_dir)

        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("SELECT DISTINCT id FROM sessions")
        sessions = [row[0] for row in cursor.fetchall()]

        for session_id in sessions:
            cursor = conn.execute(
                "SELECT turn_number FROM turns WHERE session_id = ? ORDER BY turn_number",
                (session_id,)
            )
            turn_numbers = [row[0] for row in cursor.fetchall()]

            # Check sequential (1, 2, 3, ...)
            expected = list(range(1, len(turn_numbers) + 1))
            assert turn_numbers == expected, f"Non-sequential turns in session {session_id}"

        conn.close()

    def test_all_turns_captured(self, initialized_db, temp_session_file, monkeypatch):
        """Should capture all user and assistant messages from session file."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Count expected messages in file
        expected_count = 0
        with open(temp_session_file) as f:
            for line in f:
                import json
                try:
                    obj = json.loads(line)
                    if obj.get("type") in ("user", "assistant"):
                        msg = obj.get("message", {})
                        if msg.get("role") and msg.get("content"):
                            expected_count += 1
                except:
                    pass

        # Ingest
        stats = ingest_single_session(temp_session_file)

        assert stats.turns_extracted == expected_count, \
            f"Expected {expected_count} turns, got {stats.turns_extracted}"

    def test_tool_calls_captured(self, initialized_db, temp_session_file, monkeypatch):
        """Should capture all tool_use blocks from assistant messages."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Count expected tool uses
        expected_tools = 0
        with open(temp_session_file) as f:
            for line in f:
                import json
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "assistant":
                        content = obj.get("message", {}).get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    expected_tools += 1
                except:
                    pass

        stats = ingest_single_session(temp_session_file)

        assert stats.tool_calls_extracted == expected_tools, \
            f"Expected {expected_tools} tool calls, got {stats.tool_calls_extracted}"

    def test_unicode_content_preserved(self, initialized_db, tmp_path, monkeypatch):
        """Should preserve unicode characters in content."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        unicode_content = "Hello 你好 مرحبا 🎉 émoji"
        session_file = tmp_path / "unicode.jsonl"
        session_file.write_text(json.dumps({
            "type": "user",
            "sessionId": "unicode-test",
            "cwd": "/test",
            "timestamp": "2025-01-01T00:00:00Z",
            "message": {"role": "user", "content": unicode_content}
        }))

        ingest_single_session(session_file)

        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("SELECT content FROM turns WHERE session_id = 'unicode-test'")
        stored_content = cursor.fetchone()[0]
        conn.close()

        assert stored_content == unicode_content

    def test_long_session_handling(self, initialized_db, tmp_path, monkeypatch):
        """Should handle sessions with many messages."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        # Create session with 100 messages
        session_file = tmp_path / "long-session.jsonl"
        lines = []
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            lines.append(json.dumps({
                "type": role,
                "sessionId": "long-session-test",
                "cwd": "/test",
                "timestamp": f"2025-01-01T00:00:{i:02d}Z",
                "message": {"role": role, "content": f"Message {i}"}
            }))
        session_file.write_text("\n".join(lines))

        stats = ingest_single_session(session_file)

        assert stats.sessions_processed == 1
        assert stats.turns_extracted == 100

    def test_session_metadata_captured(self, initialized_db, temp_session_file, monkeypatch):
        """Should capture session metadata correctly."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        ingest_single_session(temp_session_file)

        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("""
            SELECT project_path, project_name, git_branch
            FROM sessions WHERE id = 'test-session-123'
        """)
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "/test/project"  # project_path
        assert row[1] == "project"  # project_name
        assert row[2] == "main"  # git_branch

    def test_timestamps_preserved(self, initialized_db, temp_session_file, monkeypatch):
        """Should preserve message timestamps."""
        monkeypatch.setenv("DISTILLER_DB_PATH", str(initialized_db))

        ingest_single_session(temp_session_file)

        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("""
            SELECT timestamp FROM turns
            WHERE session_id = 'test-session-123'
            ORDER BY turn_number
        """)
        timestamps = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Timestamps should be in order
        assert timestamps == sorted(timestamps)
        # First timestamp should match fixture
        assert timestamps[0] == "2025-01-01T00:00:01Z"


# Fixtures

@pytest.fixture
def temp_claude_dir(tmp_path):
    """Create a mock Claude Code directory structure."""
    projects_dir = tmp_path / "projects"
    project_dir = projects_dir / "-Users-tyler-dev-myproject"
    project_dir.mkdir(parents=True)

    # Create main session file
    session_file = project_dir / "main-session.jsonl"
    session_file.write_text("\n".join([
        json.dumps({
            "type": "user",
            "sessionId": "test-session-123",
            "cwd": "/Users/tyler/dev/myproject",
            "gitBranch": "main",
            "timestamp": "2025-01-01T00:00:01Z",
            "message": {"role": "user", "content": "Hello"}
        }),
        json.dumps({
            "type": "assistant",
            "sessionId": "test-session-123",
            "cwd": "/Users/tyler/dev/myproject",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi!"},
                    {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"file": "test.txt"}}
                ],
                "usage": {"output_tokens": 5}
            }
        }),
    ]))

    # Create another session file
    another_session = project_dir / "another-session.jsonl"
    another_session.write_text("\n".join([
        json.dumps({
            "type": "user",
            "sessionId": "another-session-456",
            "cwd": "/Users/tyler/dev/myproject",
            "timestamp": "2025-01-02T00:00:01Z",
            "message": {"role": "user", "content": "Test"}
        }),
        json.dumps({
            "type": "assistant",
            "sessionId": "another-session-456",
            "cwd": "/Users/tyler/dev/myproject",
            "timestamp": "2025-01-02T00:00:02Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Response"}]}
        }),
    ]))

    # Create agent file (should be skipped)
    agent_file = project_dir / "agent-abc123.jsonl"
    agent_file.write_text(json.dumps({
        "type": "user",
        "sessionId": "agent-session",
        "message": {"role": "user", "content": "Agent"}
    }))

    return tmp_path


@pytest.fixture
def temp_session_file(tmp_path):
    """Create a mock session JSONL file."""
    session_file = tmp_path / "test-session.jsonl"
    session_file.write_text("\n".join([
        json.dumps({
            "type": "user",
            "sessionId": "test-session-123",
            "cwd": "/test/project",
            "gitBranch": "main",
            "timestamp": "2025-01-01T00:00:01Z",
            "message": {"role": "user", "content": "Hello"}
        }),
        json.dumps({
            "type": "assistant",
            "sessionId": "test-session-123",
            "cwd": "/test/project",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi!"},
                    {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"file": "test.txt"}}
                ],
                "usage": {"output_tokens": 5}
            }
        }),
        json.dumps({
            "type": "user",
            "sessionId": "test-session-123",
            "cwd": "/test/project",
            "timestamp": "2025-01-01T00:00:03Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "file contents"}
                ]
            }
        }),
    ]))
    return session_file
