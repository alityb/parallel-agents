from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from . import BackendAdapter, BackendResponse, ParsedToolCall
from batch_agent.spec import AgentJob, Message, SharedContext

logger = logging.getLogger(__name__)


class AnthropicBackend(BackendAdapter):
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.anthropic.com") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> BackendResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic:// backend")

        system: str | list[dict[str, Any]]
        if shared.prefix:
            system = [{"type": "text", "text": shared.prefix, "cache_control": {"type": "ephemeral"}}]
        else:
            system = ""

        # Build messages list for the API
        if messages is not None:
            api_messages = _messages_to_api(messages)
        else:
            api_messages = [{"role": "user", "content": job.prompt}]

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": api_messages,
        }

        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
        raw = response.json()

        content = _extract_text(raw)
        tool_calls = _extract_tool_calls(raw)
        stop_reason = raw.get("stop_reason", "")

        return BackendResponse(content=content, raw=raw, tool_calls=tool_calls, stop_reason=stop_reason)


def _messages_to_api(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message objects to Anthropic API message format."""
    api_msgs: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool_result":
            # Tool results are sent as user messages with tool_result content blocks
            # The content field contains JSON-encoded list of tool result blocks
            try:
                blocks = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                blocks = [{"type": "text", "text": msg.content}]
            api_msgs.append({"role": "user", "content": blocks})
        elif msg.role == "assistant_raw":
            # Raw assistant content blocks (preserves tool_use blocks for multi-turn)
            try:
                blocks = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                blocks = [{"type": "text", "text": msg.content}]
            api_msgs.append({"role": "assistant", "content": blocks})
        else:
            api_msgs.append({"role": msg.role, "content": msg.content})
    return api_msgs


def _extract_text(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in raw.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_tool_calls(raw: dict[str, Any]) -> list[ParsedToolCall]:
    """Parse tool_use content blocks into ParsedToolCall objects.

    Does not skip malformed blocks — logs them and returns with error=True.
    """
    tool_calls: list[ParsedToolCall] = []
    for block in raw.get("content", []):
        if block.get("type") != "tool_use":
            continue

        block_id = block.get("id", "")
        name = block.get("name", "")
        raw_input = block.get("input")

        # Validate required fields
        if not block_id:
            logger.warning("tool_use block missing 'id': %s", block)
            tool_calls.append(ParsedToolCall(
                id="unknown",
                name=name or "unknown",
                args={},
                error=True,
                error_message="tool_use block missing 'id' field",
            ))
            continue

        if not name:
            logger.warning("tool_use block missing 'name': %s", block)
            tool_calls.append(ParsedToolCall(
                id=block_id,
                name="unknown",
                args={},
                error=True,
                error_message="tool_use block missing 'name' field",
            ))
            continue

        # Parse input/args
        if raw_input is None:
            logger.warning("tool_use block '%s' has null input", name)
            tool_calls.append(ParsedToolCall(
                id=block_id,
                name=name,
                args={},
                error=True,
                error_message="tool_use block has null 'input' field",
            ))
            continue

        if not isinstance(raw_input, dict):
            logger.warning("tool_use block '%s' has non-dict input: %s", name, type(raw_input))
            tool_calls.append(ParsedToolCall(
                id=block_id,
                name=name,
                args={},
                error=True,
                error_message=f"tool_use block 'input' is {type(raw_input).__name__}, expected dict",
            ))
            continue

        # Valid tool call
        tool_calls.append(ParsedToolCall(
            id=block_id,
            name=name,
            args=raw_input,
            error=False,
        ))

    return tool_calls
