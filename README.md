# Distiller

Extract training data from your Claude Code sessions.

## The Problem

Every time you use Claude Code, you generate valuable training signal:

- **Task completions** — Real problems solved in real codebases
- **Corrections** — When you guide Claude back on track, that's gold
- **Tool usage patterns** — File operations, git commands, test runs
- **Domain knowledge** — Your codebase patterns, preferences, workflows

This data just sits in `~/.claude/projects/`, unused. Meanwhile, fine-tuning requires exactly this kind of high-quality, task-specific data.

## The Solution

Distiller reads your Claude Code sessions, scores them for quality, and exports training-ready datasets:

You get:

- **Quality-tiered outputs** — High/good/diverse quality buckets
- **PII removal** — API keys, emails, secrets automatically redacted
- **Rich metadata** — Turn counts, tool usage, quality scores for filtering
- **ChatML format** — Ready for fine-tuning pipelines

## How It Works

```text
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Claude Code        Distiller Pipeline              Output     │
│   ───────────        ─────────────────              ──────      │
│                                                                 │
│   ~/.claude/    ──▶  Ingest  ──▶  Clean  ──▶  Score  ──▶  JSONL │
│   projects/          (parse)      (PII)      (quality)   (tier) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Pipeline stages:**

1. **Ingest** — Reads session files, extracts conversations
2. **Clean** — Redacts emails, API keys, secrets, sensitive paths
3. **Score** — Rates quality using heuristics + LLM judge
4. **Export** — Creates tiered ChatML datasets

**Quality scoring** combines two methods:

| Method | What it measures |
|--------|------------------|
| **Heuristic** (fast, free) | Tool success rate, response depth, task completion signals |
| **LLM Judge** (via OpenRouter) | Task completion, tool effectiveness, code quality (1-5 scales) |

Scores are normalized to 0-1 and combined. By default, conversations below 0.6 are filtered out but this is configurable.

## Quick Start

```bash
# Clone and install
git clone https://github.com/roach88/distiller.git
cd distiller
uv venv && source .venv/bin/activate
uv pip install -e .

# Set up LLM scoring (optional but recommended)
echo "OPENROUTER_API_KEY=your-key-here" > .env

# Initialize
distiller init

# Extract training data
distiller run
```

Your training data will be in `output/`.

## Installation

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- macOS or Linux
- OpenRouter API key (optional, for LLM scoring — free tier at [openrouter.ai/keys](https://openrouter.ai/keys))

### Detailed Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/roach88/distiller.git
   cd distiller
   ```

2. **Create virtual environment and install**

   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e .

   # Optional: S3 support
   uv pip install -e ".[s3]"
   ```

3. **Configure LLM scoring** (optional)

   ```bash
   echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env
   ```

   Without this, Distiller uses heuristic scoring only (still works, just less accurate).

4. **Initialize**

   ```bash
   distiller init
   ```

   This creates the database, imports existing sessions, and sets up a weekly cron job.

5. **Add to PATH** (for global access)

   The `distiller` command is available when the venv is active. To use it from anywhere without activating, run this from inside the repo:

   ```bash
   # Add to your shell config (run from inside the distiller directory)
   echo "export PATH=\"\$PATH:$PWD/.venv/bin\"" >> ~/.zshrc
   source ~/.zshrc
   ```

   For bash, use `~/.bashrc` instead of `~/.zshrc`.

## Usage

### Commands

```bash
distiller                 # Show help
distiller init            # First-time setup
distiller run             # Extract with LLM scoring
distiller run --no-llm-judge  # Heuristic only (faster, free)
distiller status          # Database stats and recent outputs
distiller cron            # View/update weekly schedule
distiller reprocess       # Re-extract everything
distiller paths           # Show file locations
```

### Options

```bash
# Project filter
distiller run -p myproject

# Quality threshold (default: 0.6)
distiller run --min-score 0.8

# Upload to S3
distiller run --s3-bucket my-training-bucket
```

### Scheduling

Distiller sets up a weekly cron job during `init`:

```bash
# View current schedule
distiller cron

# Change to Fridays at 9am
distiller cron --hour 9 --day 5

# Disable
distiller cron --remove
```

## Output Format

Training data is saved as ChatML-format JSONL:

```text
output/
├── training_high_quality.jsonl   # score ≥ 0.8
├── training_good_quality.jsonl   # score 0.7-0.8
└── training_diverse.jsonl        # score 0.6-0.7
```

Each example:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert AI coding assistant..."},
    {"role": "user", "content": "Fix the auth bug in login.ts"},
    {"role": "assistant", "content": "I'll investigate...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "name": "Read", "content": "..."}
  ],
  "metadata": {
    "conversation_id": "session_abc123",
    "project": "my-app",
    "turn_count": 12,
    "tool_call_count": 8,
    "quality_scores": {
      "heuristic": {"overall": 0.75},
      "llm_judge": {"overall": 0.85, "reasoning": "..."},
      "overall": 0.80
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | LLM scoring API key | (none) |
| `DISTILLER_DB_PATH` | Database location | `data/raw_logs.db` |
| `DISTILLER_S3_BUCKET` | S3 upload bucket | (none) |

### LLM Provider

Edit `config/scoring.yaml`:

```yaml
provider: "openrouter"  # or "ollama" for local
model: "anthropic/claude-3-haiku"

scoring:
  temperature: 0.1
  max_tokens: 500
  timeout: 30
```

### Local Scoring with Ollama

For fully local, free scoring:

```bash
brew install ollama
ollama pull llama3
```

Then in `config/scoring.yaml`:

```yaml
provider: "ollama"
model: "llama3"
```

### S3 Upload

```bash
uv pip install -e ".[s3]"
aws configure
distiller run --s3-bucket my-bucket
```

## Troubleshooting

### "No sessions found"

Check that Claude Code has created sessions:

```bash
ls ~/.claude/projects/
```

If empty, use Claude Code for a while first.

### "OPENROUTER_API_KEY not set"

Either create `.env` with your key, or run without LLM scoring:

```bash
distiller run --no-llm-judge
```

### Low quality scores

- Lower the threshold: `--min-score 0.4`
- Try a different scoring model in `config/scoring.yaml`
- Check that conversations have meaningful exchanges (not just single-turn)

### Database issues

```bash
rm data/raw_logs.db
distiller init
```

## Development

```bash
uv pip install -e ".[dev]"
pytest

# Regenerate BAML client after editing baml_src/
baml-cli generate
```

### Architecture

```text
distiller/
├── cli.py              # CLI entry point
├── llm_providers.py    # OpenRouter/Ollama abstraction
├── scripts/
│   ├── ingest.py       # Session import
│   ├── parse.py        # Conversation extraction
│   ├── clean_pii.py    # PII redaction
│   ├── score_quality.py
│   └── export_chatml.py
├── baml_src/           # LLM scoring prompts
├── baml_client/        # Generated client
└── config/
    ├── scoring.yaml
    └── pii_patterns.yaml
```

## License

MIT
