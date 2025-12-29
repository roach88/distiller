#!/usr/bin/env python3
"""
Distiller - Claude Code Training Data Pipeline

Extracts Claude Code sessions, parses into conversations, removes PII,
scores quality, and exports to ChatML format for fine-tuning.

Usage:
    python pipeline.py                        # Run full pipeline (local)
    python pipeline.py --incremental          # Only process new sessions
    python pipeline.py --s3-bucket my-bucket  # Upload to S3

Environment Variables:
    DISTILLER_S3_BUCKET: Default S3 bucket for uploads
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import our pipeline modules
from scripts.ingest import export_sessions_from_db, get_db_path, ingest_backfill
from scripts.parse import parse_sessions
from scripts.clean_pii import clean_conversations
from scripts.score_quality import score_conversations
from scripts.export_chatml import export_to_chatml, export_tiered_datasets
from scripts.storage import get_storage, get_machine_id, StorageAdapter
from distiller.llm_providers import create_llm_provider, load_scoring_config


STATE_FILE = ".distiller_state.json"


def load_state(state_path: Path) -> dict:
    """Load pipeline state from file."""
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def save_state(state_path: Path, state: dict) -> None:
    """Save pipeline state to file."""
    state_path.write_text(json.dumps(state, indent=2))


def run_pipeline(
    output_dir: str = "output",
    project_filter: Optional[str] = None,
    llm_judge: bool = False,
    min_score: float = 0.6,
    incremental: bool = False,
    storage: Optional[StorageAdapter] = None,
    output_prefix: Optional[str] = None,
    tiered_export: bool = True,
    high_quality_threshold: float = 0.8,
    good_quality_threshold: float = 0.7,
    diverse_quality_threshold: float = 0.6,
) -> dict:
    """
    Full pipeline: ingest -> parse -> clean_pii -> score -> export

    Args:
        output_dir: Output directory for intermediate files
        project_filter: Filter to specific project name
        llm_judge: Use LLM-as-judge scoring (configured in config/scoring.yaml)
        min_score: Minimum quality score threshold (0.0-1.0)
        incremental: Only process sessions since last run
        storage: Storage adapter for output (local or S3)
        output_prefix: Prefix for output filename (default: machine ID)
        tiered_export: Create quality-tiered datasets (default: True)
        high_quality_threshold: Minimum score for high_quality tier (default: 0.8)
        good_quality_threshold: Minimum score for good_quality tier (default: 0.7)
        diverse_quality_threshold: Minimum score for diverse tier (default: 0.6)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # State management for incremental runs
    state_path = output_path / STATE_FILE
    state = load_state(state_path)
    last_run_timestamp = state.get("last_run_timestamp") if incremental else None

    # Generate output filename with prefix
    if output_prefix is None:
        output_prefix = get_machine_id()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_prefix}_{timestamp_str}.jsonl"

    # Define intermediate file paths (temporary)
    sessions_path = output_path / "sessions.json"
    conversations_path = output_path / "conversations.json"
    cleaned_path = output_path / "cleaned.json"
    scored_path = output_path / "scored.json"
    chatml_path = output_path / output_filename
    redaction_log_path = output_path / "redaction_log.json"
    config_path = Path(__file__).parent / "config" / "pii_patterns.yaml"

    # Track run start time
    run_start = time.time()

    # Stage 0: Ingest new sessions from Claude Code logs (incremental)
    claude_dir = Path.home() / ".claude"
    if claude_dir.exists():
        print("Stage 0: Ingesting from Claude Code logs...")
        if incremental and last_run_timestamp:
            print(f"  (incremental: only files modified since {datetime.fromtimestamp(last_run_timestamp)})")
        ingest_result = ingest_backfill(
            claude_dir,
            project_filter=project_filter,
            verbose=False,
            since_timestamp=last_run_timestamp,
        )
        new_sessions = ingest_result.sessions_processed
        print(f"  -> {new_sessions} new sessions ingested, {ingest_result.sessions_skipped} skipped (existing)")

        if incremental and new_sessions == 0:
            print("\nNo new sessions to process.")
            return {"status": "no_new_sessions"}

    # Stage 1: Export from DB
    print("Stage 1: Exporting sessions from database...")
    db_path = get_db_path()
    ingest_stats = export_sessions_from_db(db_path, sessions_path, project_filter)
    print(f"  -> {ingest_stats['sessions_exported']} sessions, {ingest_stats['total_turns']} turns")

    if ingest_stats['sessions_exported'] == 0:
        print("\nNo sessions found.")
        return {"status": "no_sessions"}

    # Stage 2: Parse
    print("Stage 2: Parsing conversations...")
    parse_stats = parse_sessions(sessions_path, conversations_path)
    print(f"  -> {parse_stats['conversations_parsed']} conversations")

    # Stage 3: Clean PII
    print("Stage 3: Cleaning PII...")
    clean_stats = clean_conversations(conversations_path, cleaned_path, config_path, redaction_log_path)
    print(f"  -> {clean_stats['total_redactions']} redactions, {clean_stats['total_replacements']} replacements")
    if clean_stats.get('system_noise_stripped', 0) > 0:
        print(f"  -> {clean_stats['system_noise_stripped']} system noise tags stripped")

    # Stage 4: Score Quality
    # Load LLM provider from config if using LLM judge
    provider = None
    if llm_judge:
        try:
            # Load config and set env vars for BAML
            config = load_scoring_config()
            os.environ['OPENROUTER_MODEL'] = config.get('model', 'x-ai/grok-code-fast-1')
            openrouter_config = config.get('openrouter', {})
            os.environ['OPENROUTER_BASE_URL'] = openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
            os.environ['OPENROUTER_SITE_URL'] = openrouter_config.get('site_url', 'https://github.com/roach88/distiller')
            os.environ['OPENROUTER_APP_NAME'] = openrouter_config.get('app_name', 'distiller')

            provider = create_llm_provider()
            score_method = f"heuristic + LLM judge ({config.get('provider')})"
        except Exception as e:
            print(f"  Warning: Could not load LLM provider: {e}")
            print(f"  Falling back to heuristic-only scoring")
            score_method = "heuristic (LLM provider failed)"
    else:
        score_method = "heuristic"

    print(f"Stage 4: Scoring quality ({score_method})...")
    score_stats = score_conversations(
        cleaned_path,
        scored_path,
        provider=provider,
        min_score=min_score,
        verbose=True,
    )
    print(f"  -> {score_stats['scored']} conversations scored, avg={score_stats.get('average_score', 'N/A')}")
    if score_stats.get('filtered', 0) > 0:
        print(f"  -> {score_stats['filtered']} filtered out (below {min_score} threshold)")

    # Stage 5: Export to ChatML
    print("Stage 5: Exporting to ChatML...")
    if tiered_export:
        # Create quality-tiered datasets
        export_stats = export_tiered_datasets(
            input_path=scored_path,
            output_dir=output_path,
            output_prefix=output_prefix,
            high_quality_threshold=high_quality_threshold,
            good_quality_threshold=good_quality_threshold,
            diverse_quality_threshold=diverse_quality_threshold,
        )
        print(f"  -> Created tiered datasets:")
        for tier_name, tier_stats in export_stats["tiers"].items():
            print(f"     - {tier_name}: {tier_stats['examples_output']} examples ({tier_stats['output_file']})")
    else:
        # Single output file
        export_stats = export_to_chatml(scored_path, chatml_path)
        print(f"  -> {export_stats['examples_output']} training examples, {export_stats['total_messages']} messages")

    # Stage 6: Upload to storage (if S3 configured)
    if tiered_export:
        final_location = {}
        if storage:
            print("Stage 6: Uploading to storage...")
            for tier_name, tier_stats in export_stats["tiers"].items():
                tier_file = Path(tier_stats["output_file"])
                if tier_file.exists():
                    data = tier_file.read_bytes()
                    s3_location = storage.save(tier_file.name, data)
                    final_location[tier_name] = s3_location
                    print(f"  -> Uploaded {tier_name}: {s3_location}")
                    # Remove local file after upload
                    tier_file.unlink()
        else:
            # Local storage - collect file paths
            for tier_name, tier_stats in export_stats["tiers"].items():
                final_location[tier_name] = tier_stats["output_file"]
    else:
        final_location = str(chatml_path)
        if storage:
            print("Stage 6: Uploading to storage...")
            data = chatml_path.read_bytes()
            final_location = storage.save(output_filename, data)
            print(f"  -> Uploaded to: {final_location}")
            # Remove local file after upload
            chatml_path.unlink()

    # Clean up intermediate files
    for intermediate in [conversations_path, cleaned_path, scored_path, redaction_log_path]:
        if intermediate.exists():
            intermediate.unlink()

    # Update state for incremental runs
    state["last_run_timestamp"] = run_start
    state["last_output"] = final_location
    state["last_run"] = datetime.now().isoformat()
    save_state(state_path, state)

    # Print final output location(s)
    if tiered_export:
        print(f"\nDone! Tiered training datasets created:")
        for tier_name, location in final_location.items():
            print(f"  - {tier_name}: {location}")
    else:
        print(f"\nDone! Training data: {final_location}")

    return {
        "status": "success",
        "ingest": ingest_stats,
        "parse": parse_stats,
        "clean_pii": clean_stats,
        "score": score_stats,
        "export": export_stats,
        "output_file": final_location,
        "machine_id": output_prefix,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Distiller - Claude Code Training Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                          # Full run, local output
  python pipeline.py --incremental            # Only new sessions
  python pipeline.py --llm-judge              # Use LLM scoring (config/scoring.yaml)
  python pipeline.py --s3-bucket my-bucket    # Upload to S3
  python pipeline.py --prefix teammate1       # Custom output prefix
        """
    )
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--project", "-p", help="Filter to specific project")
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM-as-judge scoring (configured in config/scoring.yaml)")
    parser.add_argument("--min-score", type=float, default=0.6, help="Minimum quality score (default: 0.6)")
    parser.add_argument("--incremental", "-i", action="store_true", help="Only process new sessions since last run")
    parser.add_argument("--s3-bucket", help="S3 bucket for output (enables S3 upload)")
    parser.add_argument("--s3-prefix", default="training-data", help="S3 key prefix (default: training-data)")
    parser.add_argument("--prefix", help="Output filename prefix (default: username-hostname)")
    parser.add_argument("--json", action="store_true", help="Output stats as JSON")

    # Tiered export options
    parser.add_argument("--no-tiered", action="store_true", help="Disable tiered quality datasets (create single file)")
    parser.add_argument("--high-quality-threshold", type=float, default=0.8, help="Minimum score for high_quality tier (default: 0.8)")
    parser.add_argument("--good-quality-threshold", type=float, default=0.7, help="Minimum score for good_quality tier (default: 0.7)")
    parser.add_argument("--diverse-quality-threshold", type=float, default=0.6, help="Minimum score for diverse tier (default: 0.6)")

    args = parser.parse_args()

    # Set up storage adapter
    if args.s3_bucket:
        storage = get_storage("s3", bucket=args.s3_bucket, prefix=args.s3_prefix)
    else:
        storage = None  # Local storage, file stays in output_dir

    result = run_pipeline(
        output_dir=args.output,
        project_filter=args.project,
        llm_judge=args.llm_judge,
        min_score=args.min_score,
        incremental=args.incremental,
        storage=storage,
        output_prefix=args.prefix,
        tiered_export=not args.no_tiered,
        high_quality_threshold=args.high_quality_threshold,
        good_quality_threshold=args.good_quality_threshold,
        diverse_quality_threshold=args.diverse_quality_threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2))
