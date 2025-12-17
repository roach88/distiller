-- Distiller: Claude Code Training Data Pipeline
-- Database Schema for Raw Logs
-- Version: 1.0

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- SESSIONS TABLE
-- Stores metadata about each Claude Code session
-- ============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    project_path TEXT,
    project_name TEXT,
    git_branch TEXT,
    git_repo TEXT,
    exit_reason TEXT
);

-- ============================================================================
-- TURNS TABLE
-- Stores individual conversation turns within a session
-- ============================================================================
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    token_count INTEGER,
    UNIQUE(session_id, turn_number)
);

-- ============================================================================
-- TOOL_CALLS TABLE
-- Stores tool invocations made by the assistant
-- ============================================================================
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    tool_input JSON,
    tool_output JSON,
    duration_ms INTEGER,
    success BOOLEAN
);

-- ============================================================================
-- FEEDBACK_SIGNALS TABLE
-- Stores extracted feedback signals (accepted, rejected, modified, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS feedback_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    signal_type TEXT NOT NULL CHECK (signal_type IN ('accepted', 'rejected', 'modified', 'corrected')),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    evidence JSON
);

-- ============================================================================
-- TASK_COMPLETIONS TABLE
-- Stores extracted task-outcome pairs
-- ============================================================================
CREATE TABLE IF NOT EXISTS task_completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_description TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'partial', 'failed', 'abandoned')),
    turns_involved JSON,
    extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- Optimize common query patterns
-- ============================================================================

-- Sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_project_name ON sessions(project_name);

-- Turns indexes
CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp);
CREATE INDEX IF NOT EXISTS idx_turns_role ON turns(role);

-- Tool calls indexes
CREATE INDEX IF NOT EXISTS idx_tool_calls_turn_id ON tool_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);

-- Feedback signals indexes
CREATE INDEX IF NOT EXISTS idx_feedback_signals_turn_id ON feedback_signals(turn_id);
CREATE INDEX IF NOT EXISTS idx_feedback_signals_signal_type ON feedback_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_feedback_signals_confidence ON feedback_signals(confidence);

-- Task completions indexes
CREATE INDEX IF NOT EXISTS idx_task_completions_session_id ON task_completions(session_id);
CREATE INDEX IF NOT EXISTS idx_task_completions_outcome ON task_completions(outcome);
