#!/usr/bin/env python3
"""
Initialize the Distiller SQLite database with the raw_logs schema.

Usage:
    python scripts/init_db.py [--db-path PATH]

Options:
    --db-path PATH    Path to database file (default: data/raw_logs.db)
"""

import argparse
import sqlite3
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_schema_path() -> Path:
    """Get the path to the schema SQL file."""
    return get_project_root() / "schema" / "raw_logs.sql"


def get_default_db_path() -> Path:
    """Get the default database path."""
    return get_project_root() / "data" / "raw_logs.db"


def init_database(db_path: Path) -> None:
    """
    Initialize the database with the raw_logs schema.

    Args:
        db_path: Path to the SQLite database file
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Read schema SQL
    schema_path = get_schema_path()
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_sql = schema_path.read_text()

    # Connect and execute schema
    conn = sqlite3.connect(db_path)
    try:
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Execute schema
        conn.executescript(schema_sql)
        conn.commit()

        # Verify tables created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'feedback_signals',
            'sessions',
            'task_completions',
            'tool_calls',
            'turns',
        ]

        for table in expected_tables:
            if table not in tables:
                raise RuntimeError(f"Table '{table}' was not created")

        print(f"Database initialized: {db_path}")
        print(f"Tables created: {', '.join(expected_tables)}")

    finally:
        conn.close()


def verify_database(db_path: Path) -> dict:
    """
    Verify the database schema is correct.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        dict with verification results
    """
    if not db_path.exists():
        return {"valid": False, "error": "Database file does not exist"}

    conn = sqlite3.connect(db_path)
    try:
        # Enable and check foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.execute("PRAGMA foreign_keys")
        fk_enabled = cursor.fetchone()[0]

        # Get tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Get indexes
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        indexes = [row[0] for row in cursor.fetchall()]

        return {
            "valid": True,
            "tables": tables,
            "indexes": indexes,
            "foreign_keys_enabled": bool(fk_enabled),
        }

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Initialize Distiller database")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to database file (default: data/raw_logs.db)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing database instead of initializing",
    )

    args = parser.parse_args()
    db_path = args.db_path or get_default_db_path()

    if args.verify:
        result = verify_database(db_path)
        if result["valid"]:
            print(f"Database valid: {db_path}")
            print(f"Tables: {', '.join(result['tables'])}")
            print(f"Indexes: {len(result['indexes'])}")
            print(f"Foreign keys enabled: {result['foreign_keys_enabled']}")
        else:
            print(f"Database invalid: {result['error']}")
            return 1
    else:
        init_database(db_path)

    return 0


if __name__ == "__main__":
    exit(main())
