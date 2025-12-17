#!/usr/bin/env python3
"""
Remove PII and secrets from conversations.

This script applies regex patterns to detect and redact sensitive information
including emails, API keys, passwords, tokens, and other secrets.

Usage:
    python scripts/clean_pii.py --input conversations.json --output cleaned.json
    python scripts/clean_pii.py --input conversations.json --config pii_patterns.yaml --output cleaned.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_patterns(config_path: Path) -> dict:
    """Load PII patterns from YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compile_patterns(config: dict) -> dict:
    """Compile regex patterns for efficient matching."""
    compiled = {
        "patterns": {},
        "replacements": {},
        "redaction": config.get("redaction", {}),
    }

    # Compile detection patterns
    for name, pattern in config.get("patterns", {}).items():
        try:
            compiled["patterns"][name] = re.compile(pattern)
        except re.error as e:
            print(f"Warning: Invalid pattern '{name}': {e}", file=sys.stderr)

    # Compile replacement patterns
    for name, replacement in config.get("replacements", {}).items():
        if isinstance(replacement, dict):
            try:
                compiled["replacements"][name] = {
                    "pattern": re.compile(replacement["pattern"]),
                    "replacement": replacement["replacement"],
                }
            except re.error as e:
                print(f"Warning: Invalid replacement pattern '{name}': {e}", file=sys.stderr)

    return compiled


# System noise patterns to strip
SYSTEM_NOISE_PATTERNS = [
    # System reminders injected by Claude Code
    (re.compile(r'<system-reminder>.*?</system-reminder>', re.DOTALL), ''),
    # Command messages from slash commands
    (re.compile(r'<command-message>.*?</command-message>', re.DOTALL), ''),
    (re.compile(r'<command-name>.*?</command-name>', re.DOTALL), ''),
    # Tool use status messages
    (re.compile(r'<tool-use-status>.*?</tool-use-status>', re.DOTALL), ''),
]


def strip_system_noise(text: str, stats: dict) -> str:
    """
    Remove system-injected tags that aren't part of the actual conversation.

    Args:
        text: Text to clean
        stats: Stats dict to update with counts

    Returns:
        Text with system noise removed
    """
    if not text:
        return text

    result = text
    for pattern, replacement in SYSTEM_NOISE_PATTERNS:
        matches = pattern.findall(result)
        if matches:
            stats["system_noise"] = stats.get("system_noise", 0) + len(matches)
            result = pattern.sub(replacement, result)

    # Clean up any resulting double newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def redact_text(text: str, compiled: dict, stats: dict) -> str:
    """
    Apply redaction patterns to text.

    Args:
        text: Text to redact
        compiled: Compiled patterns dict
        stats: Stats dict to update with counts

    Returns:
        Redacted text
    """
    if not text:
        return text

    # First strip system noise
    result = strip_system_noise(text, stats)

    # Apply detection patterns (secrets, PII)
    for name, pattern in compiled["patterns"].items():
        matches = pattern.findall(result)
        if matches:
            stats["patterns"][name] = stats["patterns"].get(name, 0) + len(matches)
            redaction = compiled["redaction"].get(name, compiled["redaction"].get("default", "[REDACTED]"))
            result = pattern.sub(redaction, result)

    # Apply replacements (path normalization, etc.)
    for name, replacement in compiled["replacements"].items():
        matches = replacement["pattern"].findall(result)
        if matches:
            stats["replacements"][name] = stats["replacements"].get(name, 0) + len(matches)
            result = replacement["pattern"].sub(replacement["replacement"], result)

    return result


def clean_conversation(conversation: dict, compiled: dict, stats: dict) -> dict:
    """
    Clean a single conversation of PII.

    Returns a new conversation dict with redacted content.
    """
    cleaned = {
        "id": conversation.get("id"),
        "project": conversation.get("project"),
        "project_path": redact_text(
            conversation.get("project_path", ""),
            compiled,
            stats
        ),
        "git_branch": conversation.get("git_branch"),
        "started_at": conversation.get("started_at"),
        "ended_at": conversation.get("ended_at"),
        "turns": [],
        "metadata": conversation.get("metadata", {}),
    }

    for turn in conversation.get("turns", []):
        cleaned_turn = {
            "turn_number": turn.get("turn_number"),
            "role": turn.get("role"),
            "content": redact_text(turn.get("content", ""), compiled, stats),
            "timestamp": turn.get("timestamp"),
        }

        # Clean tool calls if present
        if "tool_calls" in turn:
            cleaned_turn["tool_calls"] = []
            for tc in turn["tool_calls"]:
                cleaned_tc = {
                    "name": tc.get("name"),
                    "input": clean_tool_input(tc.get("input"), compiled, stats),
                }
                cleaned_turn["tool_calls"].append(cleaned_tc)

        # Clean tool results if present
        if "tool_results" in turn:
            cleaned_turn["tool_results"] = []
            for tr in turn["tool_results"]:
                cleaned_tr = {
                    "tool_name": tr.get("tool_name"),
                    "output": redact_text(
                        tr.get("output", ""),
                        compiled,
                        stats
                    ),
                }
                cleaned_turn["tool_results"].append(cleaned_tr)

        cleaned["turns"].append(cleaned_turn)

    return cleaned


def clean_tool_input(tool_input: any, compiled: dict, stats: dict) -> any:
    """Recursively clean tool input, which can be dict, list, or string."""
    if tool_input is None:
        return None

    if isinstance(tool_input, str):
        return redact_text(tool_input, compiled, stats)

    if isinstance(tool_input, dict):
        return {
            k: clean_tool_input(v, compiled, stats)
            for k, v in tool_input.items()
        }

    if isinstance(tool_input, list):
        return [clean_tool_input(item, compiled, stats) for item in tool_input]

    # Numbers, booleans, etc.
    return tool_input


def clean_conversations(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    log_path: Optional[Path] = None
) -> dict:
    """
    Clean all conversations of PII.

    Args:
        input_path: Path to parsed conversations JSON
        output_path: Path to write cleaned conversations
        config_path: Path to PII patterns config
        log_path: Optional path to write redaction log

    Returns:
        Stats dict with counts
    """
    # Load input
    with open(input_path, 'r') as f:
        data = json.load(f)

    conversations = data.get("conversations", [])

    # Load and compile patterns
    config = load_patterns(config_path)
    compiled = compile_patterns(config)

    # Track stats
    stats = {
        "patterns": {},      # Pattern name -> match count
        "replacements": {},  # Replacement name -> match count
        "system_noise": 0,   # System tags stripped
    }

    # Clean conversations
    cleaned_conversations = []
    for conv in conversations:
        cleaned = clean_conversation(conv, compiled, stats)
        cleaned_conversations.append(cleaned)

    # Write cleaned output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({"conversations": cleaned_conversations}, f, indent=2)

    # Write redaction log (counts only, no actual values)
    result = {
        "conversations_cleaned": len(cleaned_conversations),
        "redactions": stats["patterns"],
        "replacements": stats["replacements"],
        "system_noise_stripped": stats["system_noise"],
        "total_redactions": sum(stats["patterns"].values()),
        "total_replacements": sum(stats["replacements"].values()),
    }

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Remove PII and secrets from conversations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with parsed conversations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for cleaned conversations",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/app/config/pii_patterns.yaml"),
        help="Path to PII patterns config (default: /app/config/pii_patterns.yaml)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Output path for redaction log (counts only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        stats = clean_conversations(
            args.input,
            args.output,
            args.config,
            args.log
        )

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Cleaned {stats['conversations_cleaned']} conversations")
            print(f"  Total redactions: {stats['total_redactions']}")
            if stats['redactions']:
                for name, count in stats['redactions'].items():
                    print(f"    - {name}: {count}")
            print(f"  Total replacements: {stats['total_replacements']}")
            if stats['replacements']:
                for name, count in stats['replacements'].items():
                    print(f"    - {name}: {count}")

        return 0

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
