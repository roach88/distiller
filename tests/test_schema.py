"""Tests for database schema creation and validation."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.init_db import init_database, verify_database, get_schema_path


class TestSchemaFile:
    """Tests for schema file existence and validity."""

    def test_schema_file_exists(self):
        """Schema file should exist at expected location."""
        schema_path = get_schema_path()
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_schema_file_not_empty(self):
        """Schema file should contain SQL."""
        schema_path = get_schema_path()
        content = schema_path.read_text()
        assert len(content) > 100, "Schema file appears to be empty or too short"
        assert "CREATE TABLE" in content, "Schema should contain CREATE TABLE statements"


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_init_creates_database(self, temp_db_path):
        """init_database should create database file."""
        assert not temp_db_path.exists()
        init_database(temp_db_path)
        assert temp_db_path.exists()

    def test_init_creates_all_tables(self, initialized_db):
        """All expected tables should be created."""
        result = verify_database(initialized_db)
        assert result["valid"]

        expected_tables = [
            'feedback_signals',
            'sessions',
            'task_completions',
            'tool_calls',
            'turns',
        ]
        for table in expected_tables:
            assert table in result["tables"], f"Missing table: {table}"

    def test_init_creates_indexes(self, initialized_db):
        """Indexes should be created for performance."""
        result = verify_database(initialized_db)
        assert result["valid"]
        assert len(result["indexes"]) >= 10, "Expected at least 10 indexes"

    def test_foreign_keys_enabled(self, initialized_db):
        """Foreign key constraints should be enabled."""
        result = verify_database(initialized_db)
        assert result["foreign_keys_enabled"]

    def test_init_is_idempotent(self, temp_db_path):
        """Running init twice should not fail."""
        init_database(temp_db_path)
        init_database(temp_db_path)  # Should not raise
        result = verify_database(temp_db_path)
        assert result["valid"]


class TestSessionsTable:
    """Tests for sessions table schema."""

    def test_sessions_columns(self, db_connection):
        """Sessions table should have all required columns."""
        cursor = db_connection.execute("PRAGMA table_info(sessions)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        expected = {
            'id': 'TEXT',
            'started_at': 'TIMESTAMP',
            'ended_at': 'TIMESTAMP',
            'project_path': 'TEXT',
            'project_name': 'TEXT',
            'git_branch': 'TEXT',
            'git_repo': 'TEXT',
            'exit_reason': 'TEXT',
        }

        for col, dtype in expected.items():
            assert col in columns, f"Missing column: {col}"
            assert columns[col] == dtype, f"Wrong type for {col}: {columns[col]}"

    def test_sessions_insert(self, db_connection):
        """Should be able to insert a session."""
        db_connection.execute("""
            INSERT INTO sessions (id, started_at, project_name)
            VALUES ('test-123', '2025-01-01 00:00:00', 'test-project')
        """)
        db_connection.commit()

        cursor = db_connection.execute("SELECT * FROM sessions WHERE id = 'test-123'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'test-123'


class TestTurnsTable:
    """Tests for turns table schema."""

    def test_turns_columns(self, db_connection):
        """Turns table should have all required columns."""
        cursor = db_connection.execute("PRAGMA table_info(turns)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        expected = {
            'id': 'INTEGER',
            'session_id': 'TEXT',
            'turn_number': 'INTEGER',
            'timestamp': 'TIMESTAMP',
            'role': 'TEXT',
            'content': 'TEXT',
            'token_count': 'INTEGER',
        }

        for col, dtype in expected.items():
            assert col in columns, f"Missing column: {col}"

    def test_turns_foreign_key(self, db_connection):
        """Turns should require valid session_id."""
        with pytest.raises(sqlite3.IntegrityError):
            db_connection.execute("""
                INSERT INTO turns (session_id, turn_number, timestamp, role, content)
                VALUES ('nonexistent', 1, '2025-01-01 00:00:00', 'user', 'test')
            """)

    def test_turns_role_constraint(self, db_connection):
        """Role should only allow 'user' or 'assistant'."""
        # First create a session
        db_connection.execute("""
            INSERT INTO sessions (id, started_at)
            VALUES ('test-session', '2025-01-01 00:00:00')
        """)

        # Valid role should work
        db_connection.execute("""
            INSERT INTO turns (session_id, turn_number, timestamp, role, content)
            VALUES ('test-session', 1, '2025-01-01 00:00:01', 'user', 'test')
        """)

        # Invalid role should fail
        with pytest.raises(sqlite3.IntegrityError):
            db_connection.execute("""
                INSERT INTO turns (session_id, turn_number, timestamp, role, content)
                VALUES ('test-session', 2, '2025-01-01 00:00:02', 'invalid', 'test')
            """)


class TestToolCallsTable:
    """Tests for tool_calls table schema."""

    def test_tool_calls_columns(self, db_connection):
        """Tool calls table should have all required columns."""
        cursor = db_connection.execute("PRAGMA table_info(tool_calls)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        expected = ['id', 'turn_id', 'session_id', 'tool_name', 'tool_input', 'tool_output', 'duration_ms', 'success']
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_tool_calls_json_columns(self, db_connection):
        """JSON columns should store JSON data."""
        # Setup
        db_connection.execute("""
            INSERT INTO sessions (id, started_at) VALUES ('s1', '2025-01-01 00:00:00')
        """)
        db_connection.execute("""
            INSERT INTO turns (session_id, turn_number, timestamp, role, content)
            VALUES ('s1', 1, '2025-01-01 00:00:01', 'assistant', 'test')
        """)

        # Insert with JSON
        db_connection.execute("""
            INSERT INTO tool_calls (turn_id, session_id, tool_name, tool_input, tool_output, success)
            VALUES (1, 's1', 'Read', '{"path": "/test.txt"}', '{"content": "hello"}', 1)
        """)
        db_connection.commit()

        cursor = db_connection.execute("SELECT tool_input, tool_output FROM tool_calls WHERE id = 1")
        row = cursor.fetchone()
        assert '{"path"' in row[0]
        assert '{"content"' in row[1]


class TestFeedbackSignalsTable:
    """Tests for feedback_signals table schema."""

    def test_feedback_signals_columns(self, db_connection):
        """Feedback signals table should have all required columns."""
        cursor = db_connection.execute("PRAGMA table_info(feedback_signals)")
        columns = {row[1] for row in cursor.fetchall()}

        expected = ['id', 'turn_id', 'signal_type', 'confidence', 'evidence']
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_signal_type_constraint(self, db_connection):
        """Signal type should only allow valid values."""
        # Setup
        db_connection.execute("""
            INSERT INTO sessions (id, started_at) VALUES ('s1', '2025-01-01 00:00:00')
        """)
        db_connection.execute("""
            INSERT INTO turns (session_id, turn_number, timestamp, role, content)
            VALUES ('s1', 1, '2025-01-01 00:00:01', 'assistant', 'test')
        """)

        # Valid signal type
        db_connection.execute("""
            INSERT INTO feedback_signals (turn_id, signal_type, confidence)
            VALUES (1, 'accepted', 0.9)
        """)

        # Invalid signal type
        with pytest.raises(sqlite3.IntegrityError):
            db_connection.execute("""
                INSERT INTO feedback_signals (turn_id, signal_type, confidence)
                VALUES (1, 'invalid', 0.5)
            """)

    def test_confidence_constraint(self, db_connection):
        """Confidence should be between 0 and 1."""
        # Setup
        db_connection.execute("""
            INSERT INTO sessions (id, started_at) VALUES ('s1', '2025-01-01 00:00:00')
        """)
        db_connection.execute("""
            INSERT INTO turns (session_id, turn_number, timestamp, role, content)
            VALUES ('s1', 1, '2025-01-01 00:00:01', 'assistant', 'test')
        """)

        # Invalid confidence > 1
        with pytest.raises(sqlite3.IntegrityError):
            db_connection.execute("""
                INSERT INTO feedback_signals (turn_id, signal_type, confidence)
                VALUES (1, 'accepted', 1.5)
            """)


class TestTaskCompletionsTable:
    """Tests for task_completions table schema."""

    def test_task_completions_columns(self, db_connection):
        """Task completions table should have all required columns."""
        cursor = db_connection.execute("PRAGMA table_info(task_completions)")
        columns = {row[1] for row in cursor.fetchall()}

        expected = ['id', 'session_id', 'task_description', 'outcome', 'turns_involved', 'extracted_at']
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_outcome_constraint(self, db_connection):
        """Outcome should only allow valid values."""
        # Setup
        db_connection.execute("""
            INSERT INTO sessions (id, started_at) VALUES ('s1', '2025-01-01 00:00:00')
        """)

        # Valid outcome
        db_connection.execute("""
            INSERT INTO task_completions (session_id, task_description, outcome)
            VALUES ('s1', 'Fix bug', 'success')
        """)

        # Invalid outcome
        with pytest.raises(sqlite3.IntegrityError):
            db_connection.execute("""
                INSERT INTO task_completions (session_id, task_description, outcome)
                VALUES ('s1', 'Fix bug', 'invalid')
            """)


class TestCascadeDeletes:
    """Tests for foreign key cascade behavior."""

    def test_session_delete_cascades_to_turns(self, db_connection):
        """Deleting a session should delete its turns."""
        # Setup
        db_connection.execute("""
            INSERT INTO sessions (id, started_at) VALUES ('s1', '2025-01-01 00:00:00')
        """)
        db_connection.execute("""
            INSERT INTO turns (session_id, turn_number, timestamp, role, content)
            VALUES ('s1', 1, '2025-01-01 00:00:01', 'user', 'test')
        """)
        db_connection.commit()

        # Verify turn exists
        cursor = db_connection.execute("SELECT COUNT(*) FROM turns WHERE session_id = 's1'")
        assert cursor.fetchone()[0] == 1

        # Delete session
        db_connection.execute("DELETE FROM sessions WHERE id = 's1'")
        db_connection.commit()

        # Verify turn was deleted
        cursor = db_connection.execute("SELECT COUNT(*) FROM turns WHERE session_id = 's1'")
        assert cursor.fetchone()[0] == 0
