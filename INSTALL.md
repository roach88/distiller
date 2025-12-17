# Distiller Installation Guide

## Prerequisites

- Python 3.10+
- Claude Code CLI installed
- SQLite 3.x (included with Python)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd distiller

# 2. Create virtual environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Initialize the database
python scripts/init_db.py

# 4. Install the Claude Code hook (see below)
```

## Installing the Claude Code Hook

The hook captures your Claude Code interactions and logs them to the Distiller database.

### Step 1: Make the hook executable

```bash
chmod +x hooks/log_interaction.sh
```

### Step 2: Configure Claude Code

Add the hook to your Claude Code settings. Edit `~/.claude/settings.json`:

```json
{
  "hooks": {
    "post_tool_use": "/path/to/distiller/hooks/log_interaction.sh",
    "post_message": "/path/to/distiller/hooks/log_interaction.sh"
  }
}
```

Replace `/path/to/distiller` with the actual path to your Distiller installation.

### Step 3: Verify Installation

Run a quick Claude Code session and check the database:

```bash
# Start a Claude Code session
claude

# After the session, verify data was captured
sqlite3 data/raw_logs.db "SELECT COUNT(*) FROM sessions;"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DISTILLER_DB_PATH` | Override database path | `data/raw_logs.db` |
| `DISTILLER_S3_BUCKET` | S3 bucket for output uploads | None |

## Backfilling Existing Logs

To import your existing Claude Code history:

```bash
# Backfill all projects
python scripts/ingest.py --backfill ~/.claude -v

# Backfill a specific project
python scripts/ingest.py --backfill ~/.claude --project myproject -v

# Backfill a single session file
python scripts/ingest.py --session ~/.claude/projects/-Users-.../session-id.jsonl

# Get JSON output for scripting
python scripts/ingest.py --backfill ~/.claude --json
```

The script is idempotent - running it multiple times won't create duplicates.

## Running the Pipeline

Once data is captured, run the full Distiller pipeline:

```bash
# Quick setup with shell aliases
./setup.sh
source ~/.zshrc  # or ~/.bashrc

# Run the pipeline
distiller-run                           # Incremental with LLM scoring
distiller --help                        # See all options
distiller --s3-bucket my-bucket         # Upload to S3
```

## Troubleshooting

### Hook not capturing data

1. Check hook is executable: `ls -la hooks/log_interaction.sh`
2. Check database exists: `ls -la data/raw_logs.db`
3. Test hook manually:
   ```bash
   echo '{"event": "session_start", "project_name": "test"}' | ./hooks/log_interaction.sh
   ```

### Database errors

1. Reinitialize the database:
   ```bash
   rm data/raw_logs.db
   python scripts/init_db.py
   ```

2. Verify database:
   ```bash
   python scripts/init_db.py --verify
   ```

### Hook breaking Claude Code

The hook is designed to fail silently. Check logs at:
- Standard output from the hook
- Any errors should be caught and returned as JSON

If Claude Code is still affected, temporarily disable the hook by removing it from `~/.claude/settings.json`.
