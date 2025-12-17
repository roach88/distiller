# Distiller

Extract training data from your Claude Code sessions.

Every time you use Claude Code, you're generating valuable training data - task completions, corrections, preferences, and domain knowledge. Distiller captures this signal and transforms it into high-quality training datasets.

## Features

- **Automatic extraction** - Weekly cron job pulls from Claude Code's local cache
- **LLM quality scoring** - Uses Flow-Judge to filter for high-quality examples
- **PII cleaning** - Removes secrets, emails, and sensitive paths
- **Multiple output formats** - SFT conversations, DPO preference pairs
- **Local-first** - All processing happens on your machine
- **Simple CLI** - One command to set up, runs automatically

## Quick Start

```bash
# Clone and install
git clone https://github.com/roach88/distiller.git
cd distiller
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run setup (creates database, installs Ollama + Flow-Judge, sets up weekly cron)
distiller init

# That's it! Distiller will run automatically every Monday at 3am.
# Or run manually anytime:
distiller run
```

## Installation

### Requirements

- Python 3.10+
- macOS or Linux
- ~3GB disk space (for Flow-Judge model)

### Step-by-step

1. **Clone the repository**
   ```bash
   git clone https://github.com/roach88/distiller.git
   cd distiller
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install distiller**
   ```bash
   pip install -e .

   # Optional: S3 support for uploading outputs
   pip install -e ".[s3]"
   ```

4. **Run setup**
   ```bash
   distiller init
   ```

   This will:
   - Create the SQLite database
   - Import your existing Claude Code sessions
   - Install Ollama (if needed)
   - Download the Flow-Judge model (~2.4GB)
   - Set up a weekly cron job

## Usage

### Commands

```bash
distiller                 # Show help
distiller init            # First-time setup
distiller run             # Run extraction (with LLM scoring)
distiller run --no-llm-judge  # Run with heuristics only (faster)
distiller status          # Show database stats and recent outputs
distiller cron            # View/update the weekly schedule
distiller reprocess       # Re-extract everything from scratch
distiller paths           # Show important file locations
```

### Changing the Schedule

```bash
# View current schedule
distiller cron

# Update schedule
distiller cron --hour 9 --day 5    # Fridays at 9am

# Options:
#   --hour: 0-23 (e.g., 9 = 9am, 18 = 6pm)
#   --day:  1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun

# Disable automatic runs
distiller cron --remove
```

### Manual Extraction

```bash
# Standard run (LLM quality scoring enabled by default)
distiller run

# Faster run without LLM scoring
distiller run --no-llm-judge

# Extract only a specific project
distiller run -p myproject

# Higher quality threshold (default: 0.6)
distiller run --min-score 0.8
```

### Reprocessing

When you change settings that affect extraction (like quality thresholds), reprocess everything:

```bash
distiller reprocess                 # Full re-extraction
distiller reprocess --min-score 0.8 # With higher quality bar
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DISTILLER_DB_PATH` | Custom database location | `data/raw_logs.db` |
| `DISTILLER_S3_BUCKET` | S3 bucket for output upload | (none) |

### S3 Upload

To upload outputs to S3:

```bash
# Install S3 dependencies
pip install -e ".[s3]"

# Configure AWS credentials (standard AWS CLI method)
aws configure

# Run with S3 upload
distiller run --s3-bucket my-training-data-bucket

# Or set environment variable
export DISTILLER_S3_BUCKET=my-training-data-bucket
distiller run
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Claude Code              Distiller                  Output      │
│  ───────────              ─────────                  ──────      │
│                                                                  │
│  ~/.claude/projects/  ──▶  SQLite DB  ──▶  Pipeline  ──▶  JSONL  │
│  (session files)          (raw_logs.db)   (scoring)    (training)│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

1. **Capture**: Distiller reads Claude Code's session files from `~/.claude/projects/`
2. **Store**: Sessions are parsed and stored in a local SQLite database
3. **Process**: The pipeline cleans PII, scores quality, and extracts training examples
4. **Output**: High-quality examples are written to JSONL files

### Quality Scoring

By default, Distiller uses two scoring methods:

- **Heuristic scoring**: Fast rule-based signals (tool success rate, response quality, task completion)
- **LLM-as-Judge**: Flow-Judge model evaluates conversation quality

The scores are combined and filtered by threshold (default: 0.6).

### Output Format

Training examples are saved as JSONL in the `output/` directory:

```json
{
  "id": "session_abc123_conv_1",
  "conversations": [
    {"role": "user", "content": "Fix the auth bug in login.ts"},
    {"role": "assistant", "content": "I'll investigate..."}
  ],
  "metadata": {
    "project": "my-app",
    "quality_score": 0.85
  }
}
```

## File Locations

Run `distiller paths` to see all locations:

| Path | Description |
|------|-------------|
| `data/raw_logs.db` | SQLite database with raw sessions |
| `output/` | Generated training data (JSONL files) |
| `output/.distiller_state.json` | Pipeline state for incremental runs |
| `config/pii_patterns.yaml` | PII detection patterns |
| `models/` | Downloaded Flow-Judge model |

## Troubleshooting

### Ollama won't start

```bash
# Start manually
ollama serve

# Then retry
distiller run
```

### Flow-Judge model missing

```bash
# Reinstall the model
distiller setup-model
```

### Database issues

```bash
# Check database location
distiller paths

# Reinitialize (preserves Claude Code data)
rm data/raw_logs.db
distiller init
```

### No sessions found

Make sure you have Claude Code session files:
```bash
ls ~/.claude/projects/
```

If empty, use Claude Code for a while first - sessions are saved automatically.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test
pytest tests/test_ingest.py -v
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.
