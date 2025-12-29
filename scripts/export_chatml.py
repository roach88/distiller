#!/usr/bin/env python3
"""
Export cleaned conversations to ChatML format for fine-tuning.

Converts parsed conversations to JSONL ChatML format with:
- Multi-turn conversation support
- Tool calling with OpenAI-compatible format
- <think> tags for reasoning content (optional extraction)

Output format (one JSON object per line):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "Read", "content": "..."},
    ...
  ]
}

Usage:
    python scripts/export_chatml.py --input cleaned.json --output training.jsonl
    python scripts/export_chatml.py --input cleaned.json --output training.jsonl --include-thinking
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional


# Default system prompt for Claude Code style training
DEFAULT_SYSTEM_PROMPT = """You are an expert AI coding assistant. You help users with software engineering tasks including:
- Reading and understanding code
- Writing new code and features
- Debugging and fixing issues
- Refactoring and optimization
- Explaining technical concepts

You have access to tools for reading files, searching code, and executing commands. Use them when helpful."""


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:24]}"


def convert_turn_to_messages(
    turn: dict,
    include_thinking: bool = False,
    next_turn: Optional[dict] = None,
) -> list[dict]:
    """
    Convert a single turn to ChatML message(s).

    May return multiple messages for tool call/result pairs.
    """
    messages = []
    role = turn.get("role")
    content = turn.get("content", "")
    tool_calls = turn.get("tool_calls", [])
    tool_results = turn.get("tool_results", [])

    if role == "user":
        if content:
            messages.append({
                "role": "user",
                "content": content,
            })

    elif role == "assistant":
        # Handle thinking/reasoning extraction
        processed_content = content
        if not include_thinking:
            # Could extract <think> tags here if we want to add them
            # For now, keep content as-is
            pass

        if tool_calls:
            # Assistant message with tool calls
            formatted_tool_calls = []
            tool_id_map = {}  # Map tool name to ID for results

            for tc in tool_calls:
                tool_id = generate_tool_call_id()
                tool_name = tc.get("name", "unknown")
                tool_id_map[tool_name] = tool_id

                # Format arguments as JSON string
                tool_input = tc.get("input", {})
                if isinstance(tool_input, dict):
                    arguments = json.dumps(tool_input)
                else:
                    arguments = json.dumps({"input": tool_input})

                formatted_tool_calls.append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    }
                })

            # Message with tool calls
            msg = {
                "role": "assistant",
                "tool_calls": formatted_tool_calls,
            }
            if processed_content:
                msg["content"] = processed_content
            else:
                msg["content"] = None
            messages.append(msg)

            # Add tool results as separate messages
            for tr in tool_results:
                tool_name = tr.get("tool_name", "unknown")
                tool_id = tool_id_map.get(tool_name, generate_tool_call_id())
                output = tr.get("output", "") or tr.get("output_preview", "")  # Support both field names

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": output,
                })
        else:
            # Regular assistant message
            if processed_content:
                messages.append({
                    "role": "assistant",
                    "content": processed_content,
                })

    return messages


def convert_conversation(
    conversation: dict,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_thinking: bool = False,
    min_turns: int = 2,
    max_turns: Optional[int] = None,
) -> Optional[dict]:
    """
    Convert a conversation to ChatML format.

    Returns None if conversation doesn't meet criteria.
    """
    turns = conversation.get("turns", [])

    # Filter out conversations that are too short
    if len(turns) < min_turns:
        return None

    # Truncate if needed
    if max_turns and len(turns) > max_turns:
        turns = turns[:max_turns]

    messages = []

    # Add system message
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

    # Convert each turn
    for i, turn in enumerate(turns):
        next_turn = turns[i + 1] if i + 1 < len(turns) else None
        turn_messages = convert_turn_to_messages(
            turn,
            include_thinking=include_thinking,
            next_turn=next_turn,
        )
        messages.extend(turn_messages)

    # Validate: must have at least one user and one assistant message
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    # Calculate metadata
    has_tool_calls = any(turn.get("tool_calls") for turn in turns)
    tool_call_count = sum(len(turn.get("tool_calls", [])) for turn in turns)

    # Build metadata with quality scores and computed fields
    metadata = {
        "conversation_id": conversation.get("id"),
        "project": conversation.get("project"),
        "source": "claude-code",
        "turn_count": len(turns),
        "message_count": len(messages),
        "has_tool_calls": has_tool_calls,
        "tool_call_count": tool_call_count,
    }

    # Include quality scores if available
    quality_scores = conversation.get("quality_scores")
    if quality_scores:
        metadata["quality_scores"] = quality_scores

    return {
        "messages": messages,
        "metadata": metadata,
    }


def export_to_chatml(
    input_path: Path,
    output_path: Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_thinking: bool = False,
    min_turns: int = 2,
    max_turns: Optional[int] = None,
    split_long_conversations: bool = True,
    max_messages_per_example: int = 50,
    min_quality_score: Optional[float] = None,
    max_quality_score: Optional[float] = None,
) -> dict:
    """
    Export conversations to ChatML JSONL format.

    Args:
        input_path: Path to cleaned conversations JSON
        output_path: Path to write JSONL output
        system_prompt: System prompt to prepend
        include_thinking: Whether to include <think> tags
        min_turns: Minimum turns per conversation
        max_turns: Maximum turns per conversation (None = unlimited)
        split_long_conversations: Split long conversations into chunks
        max_messages_per_example: Max messages per training example
        min_quality_score: Minimum quality score (filter out below this)
        max_quality_score: Maximum quality score (filter out above this)

    Returns:
        Stats dict with counts
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    conversations = data.get("conversations", [])

    # Filter by quality score if specified
    filtered_conversations = []
    for conv in conversations:
        quality_scores = conv.get("quality_scores", {})
        overall_score = quality_scores.get("overall")

        # Skip if no quality score and filtering is requested
        if (min_quality_score is not None or max_quality_score is not None) and overall_score is None:
            continue

        # Apply min filter
        if min_quality_score is not None and overall_score < min_quality_score:
            continue

        # Apply max filter
        if max_quality_score is not None and overall_score > max_quality_score:
            continue

        filtered_conversations.append(conv)

    conversations = filtered_conversations

    stats = {
        "conversations_input": len(conversations),
        "examples_output": 0,
        "skipped_too_short": 0,
        "skipped_quality_filter": len(data.get("conversations", [])) - len(conversations),
        "total_messages": 0,
        "total_tool_calls": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for conv in conversations:
            turns = conv.get("turns", [])

            if len(turns) < min_turns:
                stats["skipped_too_short"] += 1
                continue

            # Handle long conversations by splitting
            if split_long_conversations and len(turns) > max_messages_per_example:
                # Split into overlapping chunks
                chunk_size = max_messages_per_example
                overlap = 5  # Keep some context between chunks

                for start in range(0, len(turns), chunk_size - overlap):
                    end = min(start + chunk_size, len(turns))
                    chunk_turns = turns[start:end]

                    chunk_conv = {
                        **conv,
                        "turns": chunk_turns,
                        "id": f"{conv.get('id')}_chunk_{start}",
                    }

                    chatml = convert_conversation(
                        chunk_conv,
                        system_prompt=system_prompt,
                        include_thinking=include_thinking,
                        min_turns=min_turns,
                    )

                    if chatml:
                        f.write(json.dumps(chatml) + "\n")
                        stats["examples_output"] += 1
                        stats["total_messages"] += len(chatml["messages"])
                        stats["total_tool_calls"] += sum(
                            len(m.get("tool_calls", []))
                            for m in chatml["messages"]
                        )
            else:
                chatml = convert_conversation(
                    conv,
                    system_prompt=system_prompt,
                    include_thinking=include_thinking,
                    min_turns=min_turns,
                    max_turns=max_turns,
                )

                if chatml:
                    f.write(json.dumps(chatml) + "\n")
                    stats["examples_output"] += 1
                    stats["total_messages"] += len(chatml["messages"])
                    stats["total_tool_calls"] += sum(
                        len(m.get("tool_calls", []))
                        for m in chatml["messages"]
                    )

    return stats


def export_tiered_datasets(
    input_path: Path,
    output_dir: Path,
    output_prefix: str = "training",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_thinking: bool = False,
    min_turns: int = 2,
    max_turns: Optional[int] = None,
    split_long_conversations: bool = True,
    max_messages_per_example: int = 50,
    high_quality_threshold: float = 0.8,
    good_quality_threshold: float = 0.7,
    diverse_quality_threshold: float = 0.6,
) -> dict:
    """
    Export conversations to multiple quality-tiered datasets.

    Creates three datasets by default:
    - high_quality.jsonl: >= 0.8 overall score
    - good_quality.jsonl: >= 0.7 and < 0.8
    - diverse.jsonl: >= 0.6 and < 0.7

    Args:
        input_path: Path to scored conversations JSON
        output_dir: Directory to write tiered JSONL files
        output_prefix: Prefix for output filenames (default: "training")
        system_prompt: System prompt to prepend
        include_thinking: Whether to include <think> tags
        min_turns: Minimum turns per conversation
        max_turns: Maximum turns per conversation
        split_long_conversations: Split long conversations into chunks
        max_messages_per_example: Max messages per training example
        high_quality_threshold: Minimum score for high_quality tier (default: 0.8)
        good_quality_threshold: Minimum score for good_quality tier (default: 0.7)
        diverse_quality_threshold: Minimum score for diverse tier (default: 0.6)

    Returns:
        Combined stats dict with tier breakdowns
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define tiers
    tiers = [
        {
            "name": "high_quality",
            "min_score": high_quality_threshold,
            "max_score": None,
            "output_path": output_dir / f"{output_prefix}_high_quality.jsonl",
        },
        {
            "name": "good_quality",
            "min_score": good_quality_threshold,
            "max_score": high_quality_threshold,
            "output_path": output_dir / f"{output_prefix}_good_quality.jsonl",
        },
        {
            "name": "diverse",
            "min_score": diverse_quality_threshold,
            "max_score": good_quality_threshold,
            "output_path": output_dir / f"{output_prefix}_diverse.jsonl",
        },
    ]

    combined_stats = {
        "tiers": {},
        "total_conversations": 0,
        "total_examples": 0,
    }

    # Export each tier
    for tier in tiers:
        tier_stats = export_to_chatml(
            input_path=input_path,
            output_path=tier["output_path"],
            system_prompt=system_prompt,
            include_thinking=include_thinking,
            min_turns=min_turns,
            max_turns=max_turns,
            split_long_conversations=split_long_conversations,
            max_messages_per_example=max_messages_per_example,
            min_quality_score=tier["min_score"],
            max_quality_score=tier["max_score"],
        )

        combined_stats["tiers"][tier["name"]] = {
            "output_file": str(tier["output_path"]),
            "min_score": tier["min_score"],
            "max_score": tier["max_score"],
            **tier_stats,
        }

        combined_stats["total_conversations"] += tier_stats["conversations_input"]
        combined_stats["total_examples"] += tier_stats["examples_output"]

    return combined_stats


def main():
    parser = argparse.ArgumentParser(
        description="Export conversations to ChatML JSONL format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with cleaned conversations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file for ChatML format",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to include",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="File containing system prompt (overrides --system-prompt)",
    )
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include <think> tags for reasoning content",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum turns per conversation (default: 2)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum turns per conversation",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Don't split long conversations",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=50,
        help="Max messages per training example (default: 50)",
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

    # Load system prompt from file if specified
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        if args.system_prompt_file.exists():
            system_prompt = args.system_prompt_file.read_text().strip()
        else:
            print(f"Warning: System prompt file not found: {args.system_prompt_file}", file=sys.stderr)

    try:
        stats = export_to_chatml(
            args.input,
            args.output,
            system_prompt=system_prompt,
            include_thinking=args.include_thinking,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            split_long_conversations=not args.no_split,
            max_messages_per_example=args.max_messages,
        )

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Exported {stats['examples_output']} training examples")
            print(f"  Input conversations: {stats['conversations_input']}")
            print(f"  Skipped (too short): {stats['skipped_too_short']}")
            print(f"  Total messages: {stats['total_messages']}")
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
