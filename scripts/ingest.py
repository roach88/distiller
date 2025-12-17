#!/usr/bin/env python3
"""
Backfill existing Claude Code logs into the Distiller database.

This script reads from ~/.claude/projects/ and parses session JSONL files
to populate the database with historical data.

Usage:
    python scripts/ingest.py --backfill ~/.claude
    python scripts/ingest.py --backfill ~/.claude --project myproject
    python scripts/ingest.py --session ~/.claude/projects/-Users-.../session-id.jsonl
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator


@dataclass
class IngestStats:
    """Statistics from an ingest run."""
    sessions_processed: int = 0
    sessions_skipped: int = 0  # Already existed (idempotent)
    turns_extracted: int = 0
    tool_calls_extracted: int = 0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sessions_processed": self.sessions_processed,
            "sessions_skipped": self.sessions_skipped,
            "turns_extracted": self.turns_extracted,
            "tool_calls_extracted": self.tool_calls_extracted,
            "errors": self.errors,
        }


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_db_path() -> Path:
    """Get the database path, with environment override."""
    env_path = os.environ.get("DISTILLER_DB_PATH")
    if env_path:
        return Path(env_path)
    return get_project_root() / "data" / "raw_logs.db"


def decode_project_path(encoded: str) -> str:
    """
    Decode a Claude Code project directory name to a path.

    Claude encodes paths by replacing / with -
    e.g., -Users-tyler-dev-myproject -> /Users/tyler/dev/myproject
    """
    if encoded.startswith("-"):
        # Replace leading - with /, then subsequent - with /
        return "/" + encoded[1:].replace("-", "/")
    return encoded.replace("-", "/")


def extract_project_name(project_path: str) -> str:
    """Extract project name from path (last component)."""
    return Path(project_path).name


def find_session_files(
    claude_dir: Path,
    project_filter: Optional[str] = None,
    since_timestamp: Optional[float] = None,
) -> Iterator[Path]:
    """
    Find all session JSONL files in Claude Code directory.

    Args:
        claude_dir: Path to ~/.claude
        project_filter: Optional project name to filter by
        since_timestamp: Only include files modified after this Unix timestamp

    Yields:
        Paths to session JSONL files
    """
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        # Decode project path for filtering
        project_path = decode_project_path(project_dir.name)
        project_name = extract_project_name(project_path)

        if project_filter and project_filter.lower() not in project_name.lower():
            continue

        # Find all JSONL files (session files)
        for jsonl_file in project_dir.glob("*.jsonl"):
            # Skip agent files (they're subprocesses, not main sessions)
            if jsonl_file.name.startswith("agent-"):
                continue

            # Skip files older than since_timestamp (incremental mode)
            if since_timestamp is not None:
                file_mtime = jsonl_file.stat().st_mtime
                if file_mtime <= since_timestamp:
                    continue

            yield jsonl_file


def parse_session_file(session_file: Path) -> dict:
    """
    Parse a session JSONL file and extract session data.

    Returns:
        Dict with session metadata and messages
    """
    session_data = {
        "session_id": None,
        "project_path": None,
        "project_name": None,
        "git_branch": None,
        "started_at": None,
        "ended_at": None,
        "messages": [],  # List of (type, role, content, timestamp, tool_info)
    }

    messages = []
    first_timestamp = None
    last_timestamp = None

    with open(session_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            timestamp = entry.get("timestamp")

            # Track timestamps for session boundaries
            if timestamp:
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp

            # Extract session metadata from first user/assistant message
            if entry_type in ("user", "assistant"):
                if session_data["session_id"] is None:
                    session_data["session_id"] = entry.get("sessionId")
                    session_data["project_path"] = entry.get("cwd")
                    session_data["git_branch"] = entry.get("gitBranch") or None
                    if session_data["project_path"]:
                        session_data["project_name"] = extract_project_name(session_data["project_path"])

                # Parse message content
                message = entry.get("message", {})
                role = message.get("role")
                content = message.get("content")

                if role and content:
                    # Handle content that can be string or array
                    if isinstance(content, str):
                        text_content = content
                        tool_uses = []
                        tool_results = []
                    else:
                        # Array of content blocks
                        text_parts = []
                        tool_uses = []
                        tool_results = []

                        for block in content:
                            if isinstance(block, dict):
                                block_type = block.get("type")
                                if block_type == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block_type == "tool_use":
                                    tool_uses.append({
                                        "id": block.get("id"),
                                        "name": block.get("name"),
                                        "input": block.get("input"),
                                    })
                                elif block_type == "tool_result":
                                    tool_results.append({
                                        "tool_use_id": block.get("tool_use_id"),
                                        "content": block.get("content"),
                                    })
                            elif isinstance(block, str):
                                text_parts.append(block)

                        text_content = "\n".join(text_parts) if text_parts else ""

                    messages.append({
                        "type": entry_type,
                        "role": role,
                        "content": text_content,
                        "timestamp": timestamp,
                        "tool_uses": tool_uses if role == "assistant" else [],
                        "tool_results": tool_results if role == "user" else [],
                        "usage": message.get("usage"),
                    })

    # If no session ID found, derive from filename
    if session_data["session_id"] is None:
        session_data["session_id"] = session_file.stem

    session_data["started_at"] = first_timestamp
    session_data["ended_at"] = last_timestamp
    session_data["messages"] = messages

    return session_data


def session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    """Check if a session already exists in the database."""
    cursor = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
    return cursor.fetchone() is not None


def insert_session(conn: sqlite3.Connection, session_data: dict) -> bool:
    """
    Insert a session and its messages into the database.

    Returns:
        True if inserted, False if skipped (already exists)
    """
    session_id = session_data["session_id"]

    # Check if session already exists (idempotent)
    if session_exists(conn, session_id):
        return False

    # Insert session
    conn.execute("""
        INSERT INTO sessions (id, started_at, ended_at, project_path, project_name, git_branch, git_repo, exit_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        session_data.get("started_at"),
        session_data.get("ended_at"),
        session_data.get("project_path"),
        session_data.get("project_name"),
        session_data.get("git_branch"),
        None,  # git_repo - not available in logs
        None,  # exit_reason - not available in logs
    ))

    return True


def insert_turns_and_tools(conn: sqlite3.Connection, session_id: str, messages: list) -> tuple[int, int]:
    """
    Insert turns and tool calls for a session.

    Returns:
        Tuple of (turns_count, tool_calls_count)
    """
    turns_count = 0
    tool_calls_count = 0
    turn_number = 0

    # First pass: collect all tool_results by tool_use_id for matching
    tool_results_map = {}  # tool_use_id -> result content
    for msg in messages:
        for tr in msg.get("tool_results", []):
            tool_use_id = tr.get("tool_use_id")
            if tool_use_id:
                # Extract content from tool_result
                content = tr.get("content")
                if isinstance(content, list):
                    # Content can be array of blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    tool_results_map[tool_use_id] = "\n".join(text_parts)
                elif isinstance(content, str):
                    tool_results_map[tool_use_id] = content
                else:
                    tool_results_map[tool_use_id] = json.dumps(content) if content else None

    # Second pass: insert turns and tool calls with matched results
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg["timestamp"]
        tool_uses = msg.get("tool_uses", [])
        usage = msg.get("usage", {})

        # Get token count if available
        token_count = None
        if usage:
            token_count = usage.get("output_tokens") if role == "assistant" else usage.get("input_tokens")

        turn_number += 1

        cursor = conn.execute("""
            INSERT INTO turns (session_id, turn_number, timestamp, role, content, token_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            turn_number,
            timestamp,
            role,
            content,
            token_count,
        ))
        turn_id = cursor.lastrowid
        turns_count += 1

        # Insert tool calls from assistant messages with matched results
        for tool_use in tool_uses:
            tool_use_id = tool_use.get("id")
            # Look up the result for this tool call
            tool_output = tool_results_map.get(tool_use_id) if tool_use_id else None

            conn.execute("""
                INSERT INTO tool_calls (turn_id, session_id, tool_name, tool_input, tool_output, duration_ms, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                turn_id,
                session_id,
                tool_use.get("name"),
                json.dumps(tool_use.get("input")) if tool_use.get("input") else None,
                tool_output,  # Now populated from matched tool_result
                None,  # duration_ms not available
                None,  # success not determined
            ))
            tool_calls_count += 1

    return turns_count, tool_calls_count


def ingest_session_file(conn: sqlite3.Connection, session_file: Path, stats: IngestStats) -> None:
    """
    Ingest a single session file into the database.

    Updates stats in place.
    """
    try:
        session_data = parse_session_file(session_file)

        if not session_data["session_id"]:
            stats.errors.append(f"{session_file}: No session ID found")
            return

        if not session_data["messages"]:
            stats.errors.append(f"{session_file}: No messages found")
            return

        # Insert session
        inserted = insert_session(conn, session_data)

        if not inserted:
            stats.sessions_skipped += 1
            return

        # Insert turns and tool calls
        turns, tools = insert_turns_and_tools(
            conn,
            session_data["session_id"],
            session_data["messages"]
        )

        stats.sessions_processed += 1
        stats.turns_extracted += turns
        stats.tool_calls_extracted += tools

    except Exception as e:
        stats.errors.append(f"{session_file}: {str(e)}")


def ingest_backfill(
    claude_dir: Path,
    project_filter: Optional[str] = None,
    verbose: bool = False,
    since_timestamp: Optional[float] = None,
) -> IngestStats:
    """
    Backfill all sessions from Claude Code directory.

    Args:
        claude_dir: Path to ~/.claude
        project_filter: Optional project name filter
        verbose: Print progress
        since_timestamp: Only process files modified after this Unix timestamp

    Returns:
        IngestStats with results
    """
    stats = IngestStats()

    db_path = get_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run init_db.py first.")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        session_files = list(find_session_files(claude_dir, project_filter, since_timestamp))
        total = len(session_files)

        if verbose:
            print(f"Found {total} session files to process")

        for i, session_file in enumerate(session_files, 1):
            if verbose and i % 10 == 0:
                print(f"Processing {i}/{total}...")

            ingest_session_file(conn, session_file, stats)

        conn.commit()

    finally:
        conn.close()

    return stats


def ingest_single_session(session_file: Path) -> IngestStats:
    """
    Ingest a single session file.

    Args:
        session_file: Path to session JSONL file

    Returns:
        IngestStats with results
    """
    stats = IngestStats()

    db_path = get_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run init_db.py first.")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        ingest_session_file(conn, session_file, stats)
        conn.commit()
    finally:
        conn.close()

    return stats


def export_sessions_from_db(db_path: Path, output_path: Path, project_filter: Optional[str] = None) -> dict:
    """
    Export sessions from database to JSON file for pipeline use.

    Args:
        db_path: Path to SQLite database
        output_path: Path to write JSON output
        project_filter: Optional project name filter

    Returns:
        Stats dict with counts
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Query sessions
        if project_filter:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE project_name LIKE ?",
                (f"%{project_filter}%",)
            )
        else:
            cursor = conn.execute("SELECT * FROM sessions")

        sessions = []
        for row in cursor:
            session = dict(row)
            session_id = session["id"]

            # Get turns for this session
            turn_cursor = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number",
                (session_id,)
            )
            session["turns"] = [dict(t) for t in turn_cursor]

            # Get tool calls for this session
            tool_cursor = conn.execute(
                "SELECT * FROM tool_calls WHERE session_id = ?",
                (session_id,)
            )
            session["tool_calls"] = [dict(tc) for tc in tool_cursor]

            sessions.append(session)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({"sessions": sessions}, f, indent=2, default=str)

        return {
            "sessions_exported": len(sessions),
            "total_turns": sum(len(s["turns"]) for s in sessions),
            "total_tool_calls": sum(len(s["tool_calls"]) for s in sessions),
        }

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Claude Code logs into Distiller database or export for pipeline"
    )
    parser.add_argument(
        "--backfill",
        type=Path,
        metavar="CLAUDE_DIR",
        help="Path to Claude Code directory (e.g., ~/.claude)",
    )
    parser.add_argument(
        "--session",
        type=Path,
        metavar="SESSION_FILE",
        help="Path to single session JSONL file",
    )
    parser.add_argument(
        "--db",
        type=Path,
        metavar="DB_PATH",
        help="Path to SQLite database (for --output mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="OUTPUT_FILE",
        help="Export sessions from database to JSON file (pipeline mode)",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Filter to specific project name (partial match)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Pipeline mode: export from DB to JSON file
    if args.output:
        db_path = args.db or get_db_path()
        try:
            stats = export_sessions_from_db(db_path, args.output, args.project)
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(f"Exported {stats['sessions_exported']} sessions to {args.output}")
            return 0
        except Exception as e:
            if args.json:
                print(json.dumps({"error": str(e)}))
            else:
                print(f"Error: {e}", file=sys.stderr)
            return 1

    if not args.backfill and not args.session:
        parser.error("Either --backfill, --session, or --output is required")

    try:
        if args.session:
            if not args.session.exists():
                raise FileNotFoundError(f"Session file not found: {args.session}")
            stats = ingest_single_session(args.session)
        else:
            if not args.backfill.exists():
                raise FileNotFoundError(f"Claude directory not found: {args.backfill}")
            stats = ingest_backfill(args.backfill, args.project, args.verbose)

        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print(f"\nIngest Complete:")
            print(f"  Sessions processed: {stats.sessions_processed}")
            print(f"  Sessions skipped:   {stats.sessions_skipped}")
            print(f"  Turns extracted:    {stats.turns_extracted}")
            print(f"  Tool calls:         {stats.tool_calls_extracted}")
            if stats.errors:
                print(f"  Errors:             {len(stats.errors)}")
                for err in stats.errors[:5]:
                    print(f"    - {err}")
                if len(stats.errors) > 5:
                    print(f"    ... and {len(stats.errors) - 5} more")

        return 0 if not stats.errors else 1

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
