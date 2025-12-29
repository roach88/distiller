#!/usr/bin/env python3
"""
Score conversation quality for training data filtering.

Two scoring methods:
1. Heuristic: Fast rule-based scoring (tool success, response quality, etc.)
2. LLM-as-Judge: Use local Ollama model to evaluate conversations

Usage:
    # Heuristic only (fast)
    python scripts/score_quality.py --input cleaned.json --output scored.json

    # LLM judge (slower, better)
    python scripts/score_quality.py --input cleaned.json --output scored.json --llm-judge

    # Both methods
    python scripts/score_quality.py --input cleaned.json --output scored.json --llm-judge --model qwen2.5:14b

    # Filter by score threshold
    python scripts/score_quality.py --input cleaned.json --output scored.json --min-score 0.6
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import requests

# Import LLM providers
sys.path.insert(0, str(Path(__file__).parent.parent))
from distiller.llm_providers import create_llm_provider, LLMProvider, load_scoring_config

# Import BAML client for structured LLM outputs
try:
    from baml_client import b as baml_client
    BAML_AVAILABLE = True
except ImportError:
    BAML_AVAILABLE = False
    baml_client = None


@dataclass
class HeuristicScores:
    """Individual heuristic quality signals."""
    tool_success_rate: float = 0.0      # % of tool calls with non-empty results
    response_substance: float = 0.0      # Avg response length score
    conversation_depth: float = 0.0      # Multi-turn engagement score
    task_completion: float = 0.0         # Signs of task completion
    error_rate: float = 0.0              # Inverse of error frequency
    code_quality: float = 0.0            # Code block presence and quality

    def overall(self) -> float:
        """Weighted average of all signals."""
        weights = {
            'tool_success_rate': 0.25,
            'response_substance': 0.15,
            'conversation_depth': 0.15,
            'task_completion': 0.20,
            'error_rate': 0.10,
            'code_quality': 0.15,
        }
        total = sum(
            getattr(self, k) * v
            for k, v in weights.items()
        )
        return round(total, 3)

    def to_dict(self) -> dict:
        return {
            'tool_success_rate': self.tool_success_rate,
            'response_substance': self.response_substance,
            'conversation_depth': self.conversation_depth,
            'task_completion': self.task_completion,
            'error_rate': self.error_rate,
            'code_quality': self.code_quality,
            'overall': self.overall(),
        }


def score_tool_success(conversation: dict) -> float:
    """Score based on tool call success rate."""
    turns = conversation.get('turns', [])

    tool_calls = 0
    successful_results = 0

    for turn in turns:
        if 'tool_calls' in turn:
            tool_calls += len(turn['tool_calls'])

        if 'tool_results' in turn:
            for result in turn['tool_results']:
                output = result.get('output', '')
                # Consider successful if output is non-empty and not an error
                if output and len(output) > 10:
                    if not any(err in output.lower() for err in ['error:', 'failed:', 'exception:', 'traceback']):
                        successful_results += 1
                    else:
                        successful_results += 0.3  # Partial credit for error handling

    if tool_calls == 0:
        return 0.5  # Neutral for conversations without tools

    return min(1.0, successful_results / tool_calls)


def score_response_substance(conversation: dict) -> float:
    """Score based on response quality and length."""
    turns = conversation.get('turns', [])

    assistant_turns = [t for t in turns if t.get('role') == 'assistant']
    if not assistant_turns:
        return 0.0

    scores = []
    for turn in assistant_turns:
        content = turn.get('content', '') or ''
        length = len(content)

        # Score based on length (diminishing returns)
        if length < 50:
            score = 0.2
        elif length < 200:
            score = 0.5
        elif length < 500:
            score = 0.7
        elif length < 1500:
            score = 0.9
        else:
            score = 1.0

        # Bonus for structured content
        if '```' in content:
            score = min(1.0, score + 0.1)
        if any(marker in content for marker in ['1.', '- ', '* ', '##']):
            score = min(1.0, score + 0.05)

        scores.append(score)

    return sum(scores) / len(scores)


def score_conversation_depth(conversation: dict) -> float:
    """Score based on conversation engagement depth."""
    turns = conversation.get('turns', [])

    user_turns = len([t for t in turns if t.get('role') == 'user'])
    assistant_turns = len([t for t in turns if t.get('role') == 'assistant'])

    # Ideal: multiple back-and-forth exchanges
    exchanges = min(user_turns, assistant_turns)

    if exchanges <= 1:
        return 0.3
    elif exchanges <= 3:
        return 0.5
    elif exchanges <= 6:
        return 0.7
    elif exchanges <= 12:
        return 0.9
    else:
        return 1.0


def score_task_completion(conversation: dict) -> float:
    """Score based on signs of task completion."""
    turns = conversation.get('turns', [])

    # Look for completion signals in assistant responses
    completion_signals = [
        'done', 'complete', 'finished', 'successfully',
        'implemented', 'created', 'fixed', 'updated',
        'here is', "here's", 'the result', 'output:',
    ]

    # Look for follow-up questions or issues (negative signals)
    issue_signals = [
        "doesn't work", "still broken", "error", "failed",
        "try again", "wrong", "not what i",
    ]

    last_assistant_turns = [
        t for t in turns[-5:] if t.get('role') == 'assistant'
    ]
    last_user_turns = [
        t for t in turns[-3:] if t.get('role') == 'user'
    ]

    score = 0.5  # Neutral baseline

    # Check assistant completion signals
    for turn in last_assistant_turns:
        content = (turn.get('content', '') or '').lower()
        for signal in completion_signals:
            if signal in content:
                score += 0.1
                break

    # Check for user complaints (negative)
    for turn in last_user_turns:
        content = (turn.get('content', '') or '').lower()
        for signal in issue_signals:
            if signal in content:
                score -= 0.15
                break

    return max(0.0, min(1.0, score))


def score_error_rate(conversation: dict) -> float:
    """Score inversely based on error frequency."""
    turns = conversation.get('turns', [])

    total_content = 0
    error_content = 0

    error_patterns = [
        r'error:', r'exception:', r'traceback',
        r'failed to', r'could not', r'unable to',
        r'syntax error', r'type error', r'name error',
    ]

    for turn in turns:
        content = turn.get('content', '') or ''
        total_content += len(content)

        for pattern in error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                error_content += 100  # Penalty per error type found

        # Also check tool results
        for result in turn.get('tool_results', []):
            output = result.get('output', '') or ''
            for pattern in error_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    error_content += 50

    if total_content == 0:
        return 0.5

    error_ratio = error_content / total_content
    # Invert: low errors = high score
    return max(0.0, 1.0 - min(1.0, error_ratio * 10))


def score_code_quality(conversation: dict) -> float:
    """Score based on code presence and quality."""
    turns = conversation.get('turns', [])

    code_blocks = 0
    quality_signals = 0

    for turn in turns:
        if turn.get('role') != 'assistant':
            continue

        content = turn.get('content', '') or ''

        # Count code blocks
        blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)
        code_blocks += len(blocks)

        for lang, code in blocks:
            # Quality signals in code
            if lang:  # Has language specified
                quality_signals += 1
            if len(code.strip()) > 50:  # Substantial code
                quality_signals += 1
            if 'def ' in code or 'function ' in code or 'class ' in code:
                quality_signals += 1  # Has functions/classes
            if '#' in code or '//' in code:
                quality_signals += 0.5  # Has comments

    if code_blocks == 0:
        return 0.5  # Neutral for non-code conversations

    avg_quality = quality_signals / code_blocks if code_blocks else 0
    return min(1.0, 0.4 + avg_quality * 0.2)


def compute_heuristic_scores(conversation: dict) -> HeuristicScores:
    """Compute all heuristic quality scores for a conversation."""
    return HeuristicScores(
        tool_success_rate=score_tool_success(conversation),
        response_substance=score_response_substance(conversation),
        conversation_depth=score_conversation_depth(conversation),
        task_completion=score_task_completion(conversation),
        error_rate=score_error_rate(conversation),
        code_quality=score_code_quality(conversation),
    )


# ============ LLM-as-Judge Scoring ============

JUDGE_PROMPT = """Evaluate this AI coding assistant conversation. Rate 1-5 for each criterion.

Conversation:
---
{conversation}
---

IMPORTANT: Respond with ONLY valid JSON, no other text. Use this exact format:
{{"task_completion": 4, "tool_use": 3, "response_quality": 4, "code_quality": 3, "reasoning": "brief reason"}}

Criteria:
- task_completion: Did assistant complete the request? (1=failed, 5=perfect)
- tool_use: Were tools used effectively? (1=poor, 5=excellent)
- response_quality: Were responses clear and helpful? (1=poor, 5=excellent)
- code_quality: Was code correct? (1=poor, 5=excellent, 3=no code)

JSON response:"""


def format_conversation_for_judge(conversation: dict, max_length: int = 4000) -> str:
    """Format conversation for LLM judge evaluation."""
    turns = conversation.get('turns', [])
    lines = []

    for turn in turns:
        role = turn.get('role', 'unknown').upper()
        content = turn.get('content', '') or ''

        # Truncate long content
        if len(content) > 500:
            content = content[:500] + '...'

        lines.append(f"[{role}]: {content}")

        # Include tool calls summary
        if 'tool_calls' in turn:
            for tc in turn['tool_calls']:
                lines.append(f"  -> Tool: {tc.get('name', 'unknown')}")

        # Include tool results summary
        if 'tool_results' in turn:
            for tr in turn['tool_results']:
                output = tr.get('output', '')[:200]
                lines.append(f"  <- Result: {output}...")

    result = '\n'.join(lines)

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length] + '\n... [truncated]'

    return result


def call_llm_provider(prompt: str, provider: Optional[LLMProvider], timeout: int = 120) -> Optional[str]:
    """Call LLM provider for evaluation."""
    if provider is None:
        return None

    return provider.score_conversation(prompt, timeout=timeout)


def parse_judge_response(response: str) -> Optional[dict]:
    """Parse LLM judge JSON response."""
    if not response:
        return None

    # Try to extract JSON from response
    try:
        # Look for JSON object in response
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    return None


def score_with_llm_judge(
    conversation: dict,
    provider: Optional[LLMProvider]
) -> Optional[dict]:
    """Score a conversation using LLM-as-judge with BAML for structured outputs."""
    if provider is None:
        return None

    formatted = format_conversation_for_judge(conversation)

    # Use BAML if available for guaranteed structured output
    if BAML_AVAILABLE and baml_client:
        try:
            # Call BAML function - returns typed ConversationScore object
            # BAML client is async, so we need to run it in the event loop
            score = asyncio.run(baml_client.JudgeConversation(conversation=formatted))

            # Normalize scores to 0-1 range
            normalized = {
                'task_completion': max(1, min(5, score.task_completion)) / 5.0,
                'tool_use': max(1, min(5, score.tool_use)) / 5.0,
                'response_quality': max(1, min(5, score.response_quality)) / 5.0,
                'code_quality': max(1, min(5, score.code_quality)) / 5.0,
                'reasoning': score.reasoning,
            }

            # Calculate overall
            normalized['overall'] = (
                normalized['task_completion'] +
                normalized['tool_use'] +
                normalized['response_quality'] +
                normalized['code_quality']
            ) / 4.0

            return normalized

        except Exception as e:
            # BAML failed, fall back to manual parsing
            print(f"BAML scoring failed: {e}, falling back to manual parsing", file=sys.stderr)

    # Fallback: manual prompt + JSON parsing (legacy method)
    prompt = JUDGE_PROMPT.format(conversation=formatted)
    response = call_llm_provider(prompt, provider=provider)
    if not response:
        return None

    scores = parse_judge_response(response)
    if not scores:
        return None

    # Normalize scores to 0-1 range
    normalized = {}
    for key in ['task_completion', 'tool_use', 'response_quality', 'code_quality']:
        if key in scores:
            try:
                val = float(scores[key])
                val = max(1, min(5, val))  # Clamp to 1-5
                normalized[key] = val / 5.0
            except (ValueError, TypeError):
                pass

    # Calculate overall
    if normalized:
        normalized['overall'] = sum(normalized.values()) / len(normalized)
        normalized['reasoning'] = scores.get('reasoning', '')

    return normalized


# ============ Main Scoring Pipeline ============

def score_conversations(
    input_path: Path,
    output_path: Path,
    provider: Optional[LLMProvider] = None,
    min_score: Optional[float] = None,
    verbose: bool = False,
) -> dict:
    """
    Score all conversations and optionally filter by quality.

    Args:
        input_path: Path to cleaned conversations JSON
        output_path: Path to write scored conversations
        provider: LLM provider for scoring (None = heuristic only)
        min_score: Minimum score threshold for filtering
        verbose: Print progress

    Returns:
        Stats dict
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    conversations = data.get('conversations', [])
    scored = []
    filtered_count = 0

    stats = {
        'total': len(conversations),
        'scored': 0,
        'filtered': 0,
        'llm_judge_failures': 0,
        'score_distribution': {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0},
    }

    for i, conv in enumerate(conversations):
        if verbose and (i + 1) % 10 == 0:
            print(f"Scoring {i + 1}/{len(conversations)}...")

        # Compute heuristic scores
        heuristic = compute_heuristic_scores(conv)

        # Optionally compute LLM judge scores
        llm_scores = None
        if provider:
            llm_scores = score_with_llm_judge(conv, provider=provider)
            if llm_scores is None:
                stats['llm_judge_failures'] += 1

        # Combine scores
        if llm_scores:
            # Average heuristic and LLM scores
            overall = (heuristic.overall() + llm_scores.get('overall', 0)) / 2
        else:
            overall = heuristic.overall()

        # Build scored conversation
        scored_conv = {
            **conv,
            'quality_scores': {
                'heuristic': heuristic.to_dict(),
                'llm_judge': llm_scores,
                'overall': round(overall, 3),
            }
        }

        # Filter by threshold
        if min_score is not None and overall < min_score:
            filtered_count += 1
            stats['filtered'] += 1
            continue

        scored.append(scored_conv)
        stats['scored'] += 1

        # Track distribution
        if overall < 0.2:
            stats['score_distribution']['0.0-0.2'] += 1
        elif overall < 0.4:
            stats['score_distribution']['0.2-0.4'] += 1
        elif overall < 0.6:
            stats['score_distribution']['0.4-0.6'] += 1
        elif overall < 0.8:
            stats['score_distribution']['0.6-0.8'] += 1
        else:
            stats['score_distribution']['0.8-1.0'] += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'conversations': scored}, f, indent=2)

    # Calculate average score
    if scored:
        avg_score = sum(c['quality_scores']['overall'] for c in scored) / len(scored)
        stats['average_score'] = round(avg_score, 3)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Score conversation quality for training data filtering"
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
        help="Output JSON file with scored conversations",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM-as-judge scoring (configured in config/scoring.yaml)",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM scoring, use heuristics only",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum quality score threshold (default: 0.6)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress",
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

    # Load LLM provider from config
    provider = None
    if args.llm_judge and not args.no_llm_judge:
        try:
            # Load config first - this is the source of truth
            config = load_scoring_config()

            # Set environment variables from config for BAML to use
            # This ensures BAML and the fallback provider use the same settings
            import os
            os.environ['OPENROUTER_MODEL'] = config.get('model', 'x-ai/grok-code-fast-1')
            openrouter_config = config.get('openrouter', {})
            os.environ['OPENROUTER_BASE_URL'] = openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
            os.environ['OPENROUTER_SITE_URL'] = openrouter_config.get('site_url', 'https://github.com/roach88/distiller')
            os.environ['OPENROUTER_APP_NAME'] = openrouter_config.get('app_name', 'distiller')

            # Create provider (for fallback path)
            provider = create_llm_provider()

            if args.verbose:
                print(f"Using LLM provider: {config.get('provider')} with model {config.get('model')}")
                if BAML_AVAILABLE:
                    print("  ✓ BAML enabled for structured outputs (guaranteed JSON parsing)")
                    print(f"  ✓ Config loaded from config/scoring.yaml")
                else:
                    print("  ⚠ BAML not available, using fallback JSON parsing")
        except Exception as e:
            print(f"Warning: Could not load LLM provider: {e}", file=sys.stderr)
            print("Falling back to heuristic-only scoring", file=sys.stderr)

    try:
        stats = score_conversations(
            args.input,
            args.output,
            provider=provider,
            min_score=args.min_score,
            verbose=args.verbose,
        )

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Scored {stats['scored']} conversations")
            if stats.get('filtered', 0) > 0:
                print(f"  Filtered out: {stats['filtered']} (below threshold)")
            if stats.get('llm_judge_failures', 0) > 0:
                print(f"  LLM judge failures: {stats['llm_judge_failures']}")
            print(f"  Average score: {stats.get('average_score', 'N/A')}")
            print(f"  Distribution:")
            for bucket, count in stats['score_distribution'].items():
                bar = '█' * (count * 20 // max(1, stats['scored']))
                print(f"    {bucket}: {count:3d} {bar}")

        return 0

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
