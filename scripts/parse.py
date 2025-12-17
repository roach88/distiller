#!/usr/bin/env python3
"""
Parse raw sessions into structured conversation turns.

This script transforms the raw session data from ingest.py into a structured
format suitable for ML training, with proper conversation turns and tool call pairing.

Usage:
    python scripts/parse.py --input sessions.json --output conversations.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def parse_sessions(input_path: Path, output_path: Path) -> dict:
    """
    Parse raw sessions into structured conversations.

    Args:
        input_path: Path to sessions JSON from ingest.py
        output_path: Path to write parsed conversations

    Returns:
        Stats dict with counts
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    sessions = data.get("sessions", [])
    conversations = []
    total_turns = 0
    total_tool_calls = 0

    for session in sessions:
        conversation = parse_session(session)
        if conversation["turns"]:
            conversations.append(conversation)
            total_turns += len(conversation["turns"])
            total_tool_calls += sum(
                len(turn.get("tool_calls", []))
                for turn in conversation["turns"]
            )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({"conversations": conversations}, f, indent=2)

    return {
        "conversations_parsed": len(conversations),
        "total_turns": total_turns,
        "total_tool_calls": total_tool_calls,
    }


def parse_session(session: dict) -> dict:
    """
    Parse a single session into a structured conversation.

    Transforms raw turns into a clean conversation format:
    - Groups consecutive same-role messages
    - Pairs tool calls with their results
    - Extracts clean text content
    """
    conversation = {
        "id": session.get("id"),
        "project": session.get("project_name"),
        "project_path": session.get("project_path"),
        "git_branch": session.get("git_branch"),
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "turns": [],
        "metadata": {
            "source": "claude-code",
        },
    }

    raw_turns = session.get("turns", [])
    tool_calls_map = build_tool_calls_map(session.get("tool_calls", []))

    # Process turns into conversation format
    i = 0
    turn_number = 0
    while i < len(raw_turns):
        raw_turn = raw_turns[i]
        role = raw_turn.get("role")
        content = raw_turn.get("content", "")
        timestamp = raw_turn.get("timestamp")
        turn_id = raw_turn.get("id")

        # Get tool calls for this turn
        turn_tool_calls = tool_calls_map.get(turn_id, [])

        turn_number += 1
        turn = {
            "turn_number": turn_number,
            "role": role,
            "content": clean_content(content),
            "timestamp": timestamp,
        }

        # Add tool calls if present (assistant turns)
        if role == "assistant" and turn_tool_calls:
            turn["tool_calls"] = [
                format_tool_call(tc) for tc in turn_tool_calls
            ]

            # Add tool results from the tool_output field (populated during ingest)
            tool_results = []
            for tc in turn_tool_calls:
                if tc.get("tool_output"):
                    tool_results.append(format_tool_result(tc))

            if tool_results:
                turn["tool_results"] = tool_results

        conversation["turns"].append(turn)
        i += 1

    return conversation


def build_tool_calls_map(tool_calls: list) -> dict:
    """Build a map of turn_id -> tool_calls for quick lookup."""
    result = {}
    for tc in tool_calls:
        turn_id = tc.get("turn_id")
        if turn_id:
            if turn_id not in result:
                result[turn_id] = []
            result[turn_id].append(tc)
    return result


def format_tool_call(tc: dict) -> dict:
    """Format a tool call for the conversation output."""
    tool_input = tc.get("tool_input")
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            pass

    return {
        "name": tc.get("tool_name"),
        "input": tool_input,
    }


def format_tool_result(tc: dict, max_length: int = 2000) -> dict:
    """
    Format a tool result for the conversation output.

    Args:
        tc: Tool call dict with tool_output
        max_length: Maximum length for output (truncate if longer)
    """
    output = tc.get("tool_output", "")

    # Ensure output is a string
    if output is None:
        output = ""
    elif not isinstance(output, str):
        output = str(output)

    if len(output) > max_length:
        output = output[:max_length] + "... [truncated]"

    return {
        "tool_name": tc.get("tool_name"),
        "output": output,
    }


def clean_content(content: str) -> str:
    """
    Clean message content for training.

    - Strips leading/trailing whitespace
    - Normalizes line endings
    """
    if not content:
        return ""
    return content.strip()




def main():
    parser = argparse.ArgumentParser(
        description="Parse raw sessions into structured conversations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file from ingest.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for parsed conversations",
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

    try:
        stats = parse_sessions(args.input, args.output)

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Parsed {stats['conversations_parsed']} conversations")
            print(f"  Total turns: {stats['total_turns']}")
            print(f"  Total tool calls: {stats['total_tool_calls']}")

        return 0

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
