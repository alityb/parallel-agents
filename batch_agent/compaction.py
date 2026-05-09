"""Message compaction — reduces context length by summarizing old tool results.

Phase 2 implementation: uses a heuristic truncation approach rather than a model call.
Every `compact_interval` turns, tool results older than `keep_recent_turns` are
truncated to a brief summary. This prevents context from growing unboundedly in
long-running agents.

Future: replace with a lightweight model call (e.g. Llama-3.2-3B) for true summarization.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .spec import Message

logger = logging.getLogger(__name__)

# Configuration
COMPACT_INTERVAL = 3       # Run compaction every N turns
KEEP_RECENT_TURNS = 2      # Keep full content for the last N turns
MAX_SUMMARY_CHARS = 200    # Max chars per compacted tool result


def should_compact(turn: int) -> bool:
    """Returns True if compaction should be triggered at this turn."""
    return turn > 0 and turn % COMPACT_INTERVAL == 0


def compact_messages(messages: list[Message], current_turn: int) -> list[Message]:
    """Compact old tool results in the message history.

    Keeps the first user message and all recent messages intact.
    Older tool_result messages are truncated.

    Returns a new list (does not mutate the original).
    """
    if len(messages) <= 3:
        return messages[:]

    # Find which messages are "old" vs "recent"
    # We track turn boundaries by counting assistant responses
    turn_boundaries: list[int] = []
    for i, msg in enumerate(messages):
        if msg.role in ("assistant", "assistant_raw"):
            turn_boundaries.append(i)

    # Keep messages from the last `keep_recent_turns` assistant responses onward
    if len(turn_boundaries) > KEEP_RECENT_TURNS:
        cutoff_idx = turn_boundaries[-KEEP_RECENT_TURNS]
    else:
        return messages[:]

    compacted: list[Message] = []
    compacted_count = 0

    for i, msg in enumerate(messages):
        if i >= cutoff_idx:
            # Recent: keep as-is
            compacted.append(msg)
        elif msg.role == "tool_result" and i > 0:
            # Old tool result: compact it
            summary = _summarize_tool_result(msg.content)
            compacted.append(Message(role="tool_result", content=summary))
            compacted_count += 1
        else:
            # Keep user messages and assistant messages (they're short context)
            compacted.append(msg)

    if compacted_count > 0:
        logger.info("Compacted %d old tool result(s) at turn %d", compacted_count, current_turn)

    return compacted


def _summarize_tool_result(content: str) -> str:
    """Create a compact summary of a tool result."""
    try:
        blocks = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        blocks = [{"type": "tool_result", "tool_use_id": "unknown", "content": content}]

    summarized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        tool_content = block.get("content", "")
        tool_id = block.get("tool_use_id", "unknown")

        if len(tool_content) > MAX_SUMMARY_CHARS:
            truncated = tool_content[:MAX_SUMMARY_CHARS]
            summary = f"[COMPACTED: {len(tool_content)} chars → {MAX_SUMMARY_CHARS}] {truncated}..."
        else:
            summary = tool_content

        summarized_blocks.append({
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": summary,
        })

    return json.dumps(summarized_blocks)
