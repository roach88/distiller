# Distiller Journal

Development decisions, insights, and learnings for the Distiller project.

---

## 2024-12-29: BAML Integration for Quality Scoring

**Decision**: Use BAML for structured LLM-based quality scoring instead of raw API calls.

**Why**:

- BAML provides type-safe structured outputs (no JSON parsing gymnastics)
- Separates prompt engineering from Python code (prompts live in `.baml` files)
- Generated client handles retries, validation, streaming
- Makes scoring criteria explicit and auditable

**Alternative considered**: Direct OpenRouter/Anthropic API calls with JSON mode. Rejected because it couples prompt engineering to Python code and requires manual output validation.

**Files**: `baml_src/`, `baml_client/`, `scripts/score_quality.py`

---

## 2024-12-29: Hybrid Scoring Architecture

**Decision**: Implement both heuristic and LLM-based scoring, composable via config.

**Why**:

- Heuristics are fast and free (good for bulk filtering)
- LLM scoring is accurate but costly (good for final quality gate)
- Config-driven approach lets users choose cost/quality tradeoff

**Pattern**: `config/scoring.yaml` controls which scorers run and their weights.

---

## 2024-12-29: LLM Provider Abstraction

**Decision**: Extract LLM provider logic into `distiller/llm_providers.py`.

**Why**:

- CLI was getting bloated with provider-specific code
- Multiple scripts need LLM access (scoring, future features)
- Single place to manage API keys, model selection, rate limiting

**Insight**: The original CLI had LLM code interleaved with argument parsing. Extracting it revealed the provider abstraction was simpler than expected—just needs model name and API key routing.

---

## 2024-12-29: Export Script Enhancement

**Decision**: `export_chatml.py` now supports quality filtering and richer metadata.

**Why**:

- Training data quality matters more than quantity
- Users need to filter by score thresholds
- Metadata (session info, tool usage) helps with analysis

**Pattern**: Quality scores flow through the pipeline: extract → score → filter → export.
