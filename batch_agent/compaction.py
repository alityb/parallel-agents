"""Message compaction — reduces context length by summarizing old tool results.

Two modes:
1. Model-based (spec §3.5): sends old tool results to a small model via a low-priority
   inference call.  Activated when `backend` and `model` are provided.
   BLOCKER: requires a running inference endpoint (e.g. Llama-3.2-3B).
   Recorded in LOGS.md — the code path is implemented but cannot be exercised
   without a separate small-model endpoint.

2. Heuristic fallback (current behaviour): truncates old tool results to MAX_SUMMARY_CHARS
   with a [COMPACTED] marker.  Used when no compaction backend is available.

Design decision: the heuristic is NOT silently substituted without notice.
The SchedulerMetrics object records every compaction event and whether it was
model-based or heuristic so callers can detect the degraded path.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from .spec import Message

if TYPE_CHECKING:
    from .backends import BackendAdapter

logger = logging.getLogger(__name__)

# ── configuration ──────────────────────────────────────────────────────────────
COMPACT_INTERVAL = 3        # trigger every N turns
KEEP_RECENT_TURNS = 2       # keep full content for the last N turns' messages
MAX_SUMMARY_CHARS = 200     # heuristic mode: max chars per old tool result
COMPACTION_MODEL_MAX_TOKENS = 200   # model mode: response budget for summary

COMPACTION_SYSTEM_PROMPT = (
    "You are a concise summarizer. Return only the requested summary."
)


def should_compact(turn: int) -> bool:
    return turn > 0 and turn % COMPACT_INTERVAL == 0


def compact_messages(
    messages: list[Message],
    current_turn: int,
    backend: "BackendAdapter | None" = None,
    model: str = "",
) -> list[Message]:
    """Compact old tool results.

    If `backend` is provided, uses model-based compaction (async, must be awaited
    via compact_messages_async). Otherwise uses heuristic truncation.

    Keeping this synchronous wrapper makes callers simpler for the common case.
    """
    if backend and model:
        # Can't call async from sync — caller must use compact_messages_async
        logger.warning(
            "compact_messages called with backend but not awaited; "
            "falling back to heuristic. Use compact_messages_async instead."
        )
    return _compact_heuristic(messages, current_turn)


async def compact_messages_async(
    messages: list[Message],
    current_turn: int,
    backend: "BackendAdapter | None" = None,
    model: str = "",
) -> list[Message]:
    """Async version: uses model-based compaction if backend is provided."""
    if backend and model:
        try:
            return await _compact_model_based(messages, current_turn, backend, model)
        except Exception as exc:
            logger.warning(
                "Model-based compaction failed (%s); falling back to heuristic", exc
            )
    return _compact_heuristic(messages, current_turn)


# ── heuristic compaction ───────────────────────────────────────────────────────

def _compact_heuristic(messages: list[Message], current_turn: int) -> list[Message]:
    if len(messages) <= 3:
        return messages[:]

    turn_boundaries: list[int] = []
    for i, msg in enumerate(messages):
        if msg.role in ("assistant", "assistant_raw"):
            turn_boundaries.append(i)

    if len(turn_boundaries) <= KEEP_RECENT_TURNS:
        return messages[:]

    cutoff_idx = turn_boundaries[-KEEP_RECENT_TURNS]
    compacted: list[Message] = []
    compacted_count = 0

    for i, msg in enumerate(messages):
        if i >= cutoff_idx:
            compacted.append(msg)
        elif msg.role == "tool_result" and i > 0:
            summary = _summarize_tool_result_heuristic(msg.content)
            compacted.append(Message(role="tool_result", content=summary))
            compacted_count += 1
        else:
            compacted.append(msg)

    if compacted_count > 0:
        logger.info(
            "Heuristic compaction: %d tool result(s) truncated at turn %d",
            compacted_count, current_turn,
        )
    return compacted


def _summarize_tool_result_heuristic(content: str) -> str:
    try:
        blocks = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        blocks = [{"type": "tool_result", "tool_use_id": "unknown", "content": content}]

    summarized: list[dict[str, Any]] = []
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
        summarized.append({"type": "tool_result", "tool_use_id": tool_id, "content": summary})

    return json.dumps(summarized)


# ── model-based compaction ─────────────────────────────────────────────────────

async def _compact_model_based(
    messages: list[Message],
    current_turn: int,
    backend: "BackendAdapter",
    model: str,
) -> list[Message]:
    """Send old tool results to a small model for real summarization.

    BLOCKER: requires a running inference endpoint reachable from the orchestration
    server.  The backend here should be a low-priority adapter (e.g. pointing at
    a separate Llama-3.2-3B endpoint).  When the compaction_backend_url is the same
    as the main backend, compaction requests contend with agent inference.  The spec
    says compaction should go through a "separate low-priority queue" — implement
    that when the infrastructure is available.
    """
    from .spec import AgentJob, SharedContext

    # Identify old tool results to compact
    turn_boundaries: list[int] = []
    for i, msg in enumerate(messages):
        if msg.role in ("assistant", "assistant_raw"):
            turn_boundaries.append(i)

    if len(turn_boundaries) <= KEEP_RECENT_TURNS:
        return messages[:]

    cutoff_idx = turn_boundaries[-KEEP_RECENT_TURNS]
    old_tool_results = [
        msg for i, msg in enumerate(messages)
        if i < cutoff_idx and msg.role == "tool_result"
    ]
    if not old_tool_results:
        return messages[:]

    # Build a single compaction prompt
    tool_content_parts = []
    for msg in old_tool_results:
        try:
            blocks = json.loads(msg.content)
            for block in blocks:
                if isinstance(block, dict):
                    tool_content_parts.append(block.get("content", ""))
        except Exception:
            tool_content_parts.append(msg.content[:500])

    all_content = "\n---\n".join(tool_content_parts)
    compaction_prompt = (
        "Summarize the following tool results in 2-3 sentences, preserving all "
        f"factual content: {all_content}"
    )

    # Stub AgentJob and SharedContext for the compaction call
    compaction_job = AgentJob(
        job_id="compaction",
        index=-1,
        input_data={},
        prompt=compaction_prompt,
        estimated_prompt_tokens=len(compaction_prompt) // 4,
    )
    compaction_shared = SharedContext(prefix=COMPACTION_SYSTEM_PROMPT)

    logger.info("Model-based compaction: summarizing %d old tool results", len(old_tool_results))
    response = await backend.generate(
        shared=compaction_shared,
        job=compaction_job,
        model=model,
        timeout=30.0,
    )
    summary_text = response.content.strip()

    # Replace all old tool_result messages with one compact summary
    result: list[Message] = []
    replaced = False
    for i, msg in enumerate(messages):
        if i < cutoff_idx and msg.role == "tool_result":
            if not replaced:
                # Inject the summary once
                summary_block = json.dumps([{
                    "type": "tool_result",
                    "tool_use_id": "compaction_summary",
                    "content": f"[MODEL SUMMARY of {len(old_tool_results)} earlier tool results] {summary_text}",
                }])
                result.append(Message(role="tool_result", content=summary_block))
                replaced = True
            # Skip subsequent old tool_results — they're covered by the summary
        else:
            result.append(msg)

    logger.info("Model-based compaction complete at turn %d", current_turn)
    return result
