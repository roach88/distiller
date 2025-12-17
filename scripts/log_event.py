#!/usr/bin/env python3
"""
Log Claude Code events to the Distiller database.

This module handles database writes for Claude Code hook events.
It's designed to be called from the hook shell script.

Usage:
    echo '{"event": "session_start", ...}' | python scripts/log_event.py
"""

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_db_path() -> Path:
    """Get the database path, with environment override."""
    env_path = os.environ.get("DISTILLER_DB_PATH")
    if env_path:
        return Path(env_path)
    return get_project_root() / "data" / "raw_logs.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with foreign keys enabled."""
    db_path = get_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run init_db.py first.")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def log_session_start(data: dict) -> str:
    """
    Log a session start event.

    Args:
        data: Event data containing session metadata

    Returns:
        Session ID
    """
    session_id = data.get("session_id") or str(uuid.uuid4())

    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO sessions (id, started_at, project_path, project_name, git_branch, git_repo)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().isoformat(),
            data.get("project_path"),
            data.get("project_name"),
            data.get("git_branch"),
            data.get("git_repo"),
        ))
        conn.commit()
        return session_id
    finally:
        conn.close()


def log_session_end(data: dict) -> None:
    """
    Log a session end event.

    Args:
        data: Event data containing session_id and exit_reason
    """
    session_id = data.get("session_id")
    if not session_id:
        return

    conn = get_connection()
    try:
        conn.execute("""
            UPDATE sessions
            SET ended_at = ?, exit_reason = ?
            WHERE id = ?
        """, (
            datetime.now().isoformat(),
            data.get("exit_reason"),
            session_id,
        ))
        conn.commit()
    finally:
        conn.close()


def log_turn(data: dict) -> int:
    """
    Log a conversation turn (user or assistant message).

    Args:
        data: Event data containing session_id, role, content

    Returns:
        Turn ID
    """
    session_id = data.get("session_id")
    if not session_id:
        raise ValueError("session_id is required for turn logging")

    conn = get_connection()
    try:
        # Get the next turn number for this session
        cursor = conn.execute(
            "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM turns WHERE session_id = ?",
            (session_id,)
        )
        turn_number = cursor.fetchone()[0]

        cursor = conn.execute("""
            INSERT INTO turns (session_id, turn_number, timestamp, role, content, token_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            turn_number,
            datetime.now().isoformat(),
            data.get("role", "user"),
            data.get("content", ""),
            data.get("token_count"),
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def log_tool_call(data: dict) -> int:
    """
    Log a tool call.

    Args:
        data: Event data containing turn_id, session_id, tool details

    Returns:
        Tool call ID
    """
    session_id = data.get("session_id")
    turn_id = data.get("turn_id")

    if not session_id:
        raise ValueError("session_id is required for tool call logging")

    # If no turn_id provided, get the last assistant turn
    if not turn_id:
        conn = get_connection()
        try:
            cursor = conn.execute(
                "SELECT id FROM turns WHERE session_id = ? AND role = 'assistant' ORDER BY turn_number DESC LIMIT 1",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                turn_id = row[0]
        finally:
            conn.close()

    if not turn_id:
        raise ValueError("Could not determine turn_id for tool call")

    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO tool_calls (turn_id, session_id, tool_name, tool_input, tool_output, duration_ms, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            turn_id,
            session_id,
            data.get("tool_name"),
            json.dumps(data.get("tool_input")) if data.get("tool_input") else None,
            json.dumps(data.get("tool_output")) if data.get("tool_output") else None,
            data.get("duration_ms"),
            data.get("success", True),
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def process_event(event_data: dict) -> dict:
    """
    Process a Claude Code hook event.

    Args:
        event_data: The event data from stdin

    Returns:
        Result dict with status and any IDs created
    """
    event_type = event_data.get("event")

    if not event_type:
        return {"status": "error", "message": "No event type specified"}

    try:
        if event_type == "session_start":
            session_id = log_session_start(event_data)
            return {"status": "ok", "session_id": session_id}

        elif event_type == "session_end":
            log_session_end(event_data)
            return {"status": "ok"}

        elif event_type in ("user_message", "assistant_message"):
            # Map event type to role
            role = "user" if event_type == "user_message" else "assistant"
            event_data["role"] = role
            turn_id = log_turn(event_data)
            return {"status": "ok", "turn_id": turn_id}

        elif event_type == "tool_call":
            tool_call_id = log_tool_call(event_data)
            return {"status": "ok", "tool_call_id": tool_call_id}

        else:
            return {"status": "ignored", "message": f"Unknown event type: {event_type}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    """Main entry point - read JSON from stdin and process."""
    try:
        # Read JSON from stdin
        input_data = sys.stdin.read().strip()
        if not input_data:
            print(json.dumps({"status": "error", "message": "No input data"}))
            return 1

        event_data = json.loads(input_data)
        result = process_event(event_data)
        print(json.dumps(result))
        return 0 if result.get("status") != "error" else 1

    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}))
        return 1
    except Exception as e:
        # Catch all exceptions to avoid breaking Claude Code
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
