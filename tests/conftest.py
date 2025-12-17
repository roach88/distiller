"""Shared pytest fixtures for Distiller tests."""

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db_path():
    """Create a temporary database path (file does not exist yet)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.db"
        yield path


@pytest.fixture
def initialized_db(temp_db_path):
    """Create an initialized database with schema."""
    from scripts.init_db import init_database

    init_database(temp_db_path)
    return temp_db_path


@pytest.fixture
def db_connection(initialized_db):
    """Get a connection to an initialized database."""
    conn = sqlite3.connect(initialized_db)
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()
