# Distiller

Extract high-quality training data from your Claude Code sessions.

Every time you use Claude Code, you generate valuable training signal—task completions, corrections, tool usage patterns, and domain knowledge. Distiller captures this and transforms it into quality-tiered datasets ready for fine-tuning.

## What You Get

```
85 sessions → Pipeline → 528 training examples

output/
├── training_high_quality.jsonl   # 258 examples (score ≥ 0.8)
├── training_good_quality.jsonl   # 191 examples (score 0.7-0.8)
└── training_diverse.jsonl        #  79 examples (score 0.6-0.7)
```

Each example includes rich metadata for filtering and analysis:
- Quality scores (heuristic + LLM judge)
- Turn counts, tool usage stats
- Project source, conversation IDs

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/roach88/distiller.git
cd distiller
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Set up your API key (for LLM quality scoring)
echo "OPENROUTER_API_KEY=your-key-here" > .env

# 3. Run setup
distiller init

# 4. Extract training data
distiller run
```

That's it! Your training data will be in `output/`.

## Installation

### Requirements

- Python 3.10+
- macOS or Linux
- OpenRouter API key (free tier available at [openrouter.ai/keys](https://openrouter.ai/keys))

### Step-by-Step

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

4. **Configure LLM scoring**

   Create a `.env` file with your OpenRouter API key:
   ```bash
   echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env
   ```

   Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys).

5. **Run setup**
   ```bash
   distiller init
   ```

   This will:
   - Create the SQLite database
   - Import your existing Claude Code sessions
   - Set up a weekly cron job (Mondays at 3am)

## Usage

### Commands

```bash
distiller                 # Show help
distiller init            # First-time setup
distiller run             # Run extraction (with LLM scoring)
distiller run --no-llm-judge  # Heuristic scoring only (faster, no API cost)
distiller status          # Show database stats and recent outputs
distiller cron            # View/update the weekly schedule
distiller reprocess       # Re-extract everything from scratch
distiller paths           # Show important file locations
```

### Running Extraction

```bash
# Standard run with LLM quality scoring
distiller run

# Faster run without LLM scoring (uses heuristics only)
distiller run --no-llm-judge

# Extract only a specific project
distiller run -p myproject

# Higher quality threshold (default: 0.6)
distiller run --min-score 0.8

# Upload to S3
distiller run --s3-bucket my-training-bucket
```

### Scheduling

Distiller sets up a weekly cron job during `init`. To modify:

```bash
# View current schedule
distiller cron

# Change schedule (Fridays at 9am)
distiller cron --hour 9 --day 5

# Options:
#   --hour: 0-23 (e.g., 9 = 9am, 18 = 6pm)
#   --day:  1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun

# Disable automatic runs
distiller cron --remove
```

### Reprocessing

When you change settings that affect extraction, reprocess everything:

```bash
distiller reprocess                 # Full re-extraction
distiller reprocess --min-score 0.8 # With higher quality bar
```

## How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                           PIPELINE                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Claude Code         Distiller                         Output        │
│   ───────────         ─────────                         ──────        │
│                                                                       │
│   ~/.claude/     ──▶  Ingest   ──▶  Clean PII  ──▶  Score  ──▶  JSONL │
│   projects/           (SQLite)      (redact)       (LLM)     (tiered) │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Ingest** — Reads Claude Code session files from `~/.claude/projects/`
2. **Parse** — Extracts conversations with user/assistant turns and tool calls
3. **Clean PII** — Removes emails, API keys, secrets, and sensitive paths
4. **Score** — Rates quality using heuristics + LLM judge
5. **Export** — Creates tiered ChatML datasets by quality level

### Quality Scoring

Distiller combines two scoring methods:

**Heuristic Scoring** (fast, local):
- Tool success rate
- Response substance
- Conversation depth
- Task completion signals
- Error rate
- Code quality indicators

**LLM-as-Judge Scoring** (via OpenRouter):
- Task completion (1-5)
- Tool usage effectiveness (1-5)
- Response quality (1-5)
- Code quality (1-5)
- Reasoning explanation

Scores are normalized to 0-1 and combined (50% heuristic + 50% LLM judge).

### Tiered Output

Conversations are automatically sorted into quality tiers:

| Tier | Score Range | Use Case |
|------|-------------|----------|
| `high_quality` | ≥ 0.8 | Production fine-tuning |
| `good_quality` | 0.7 - 0.8 | Augment training data |
| `diverse` | 0.6 - 0.7 | Robustness, edge cases |

Below 0.6 is filtered out by default.

## Output Format

Training examples are saved as ChatML-format JSONL:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert AI coding assistant..."},
    {"role": "user", "content": "Fix the auth bug in login.ts"},
    {"role": "assistant", "content": "I'll investigate the authentication issue...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_abc123", "name": "Read", "content": "..."}
  ],
  "metadata": {
    "conversation_id": "session_abc123",
    "project": "my-app",
    "source": "claude-code",
    "turn_count": 12,
    "message_count": 45,
    "has_tool_calls": true,
    "tool_call_count": 8,
    "quality_scores": {
      "heuristic": {"overall": 0.75, "tool_success_rate": 0.9, ...},
      "llm_judge": {"overall": 0.85, "task_completion": 5, "reasoning": "..."},
      "overall": 0.80
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | API key for LLM scoring | (required for LLM judge) |
| `DISTILLER_DB_PATH` | Custom database location | `data/raw_logs.db` |
| `DISTILLER_S3_BUCKET` | S3 bucket for output upload | (none) |

### LLM Provider Configuration

Edit `config/scoring.yaml` to customize the LLM provider:

```yaml
# Provider: "openrouter", "ollama", or "none"
provider: "openrouter"

# Model selection
# OpenRouter: "anthropic/claude-3.5-sonnet", "openai/gpt-4o", etc.
# Ollama: "llama3", "mistral", etc.
model: "x-ai/grok-code-fast-1"

# Scoring parameters
scoring:
  temperature: 0.1
  max_tokens: 500
  timeout: 30
```

### Using Ollama (Local, Free)

For fully local scoring without API costs:

```bash
# Install Ollama
brew install ollama

# Pull a model
ollama pull llama3

# Update config/scoring.yaml
provider: "ollama"
model: "llama3"

# Run
distiller run
```

### S3 Upload

```bash
# Install S3 dependencies
pip install -e ".[s3]"

# Configure AWS credentials
aws configure

# Run with S3 upload
distiller run --s3-bucket my-training-data-bucket

# Or set environment variable
export DISTILLER_S3_BUCKET=my-training-data-bucket
distiller run
```

## File Locations

Run `distiller paths` to see all locations:

| Path | Description |
|------|-------------|
| `data/raw_logs.db` | SQLite database with raw sessions |
| `output/` | Generated training data (JSONL files) |
| `output/.distiller_state.json` | Pipeline state for incremental runs |
| `config/pii_patterns.yaml` | PII detection patterns |
| `config/scoring.yaml` | LLM provider configuration |
| `.env` | API keys (not committed to git) |

## Troubleshooting

### "OPENROUTER_API_KEY not set"

Create a `.env` file in the project root:
```bash
echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env
```

Or run without LLM scoring:
```bash
distiller run --no-llm-judge
```

### Ollama won't start

```bash
# Start manually
ollama serve

# Then retry
distiller run
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

If empty, use Claude Code for a while first—sessions are saved automatically.

### Low quality scores

If most conversations score below 0.6:
- Lower the threshold: `distiller run --min-score 0.4`
- Check `config/scoring.yaml` model settings
- Try a different LLM model for scoring

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test
pytest tests/test_ingest.py -v

# Regenerate BAML client (after editing baml_src/)
baml-cli generate
```

## Architecture

```
distiller/
├── cli.py              # CLI commands
├── llm_providers.py    # OpenRouter/Ollama abstraction
├── scripts/
│   ├── ingest.py       # Session import
│   ├── parse.py        # Conversation extraction
│   ├── clean_pii.py    # PII redaction
│   ├── score_quality.py # Heuristic + LLM scoring
│   └── export_chatml.py # ChatML + tiered export
├── baml_src/           # BAML definitions for structured LLM output
├── baml_client/        # Auto-generated type-safe client
└── config/
    ├── scoring.yaml    # LLM provider config
    └── pii_patterns.yaml # PII detection rules
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.
