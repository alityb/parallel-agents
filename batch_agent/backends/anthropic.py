from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from . import BackendAdapter, BackendResponse, ParsedToolCall, StreamingToolCall
from ..spec import AgentJob, Message, SharedContext
from ..utils import DEFAULT_MAX_TOKENS, strip_preamble_headers

# Capabilities shared by all API-mode (non-self-hosted) backends.
_API_MODE_CAPABILITIES = {
    "prefix_pinning": False,
    "kvflow": False,
    "diff_kv": False,
    "nvext_agent_hints": False,
    "max_safe_concurrent": 5,
}

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
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> BackendResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic:// backend")

        system: str | list[dict[str, Any]]
        if shared.prefix:
            system_prompt = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        else:
            system = ""

        # Build messages list for the API
        if messages is not None:
            api_messages = _messages_to_api(messages)
        else:
            api_messages = [{"role": "user", "content": job.prompt}]

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": DEFAULT_MAX_TOKENS,
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

    async def generate_streaming(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        tool_queue: Any | None = None,
    ) -> BackendResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic:// backend")

        system: str | list[dict[str, Any]]
        if shared.prefix:
            system_prompt = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        else:
            system = ""

        api_messages = _messages_to_api(messages) if messages is not None else [{"role": "user", "content": job.prompt}]
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": api_messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.base_url}/v1/messages", json=payload, headers=headers) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" not in content_type:
                        raw = json.loads((await response.aread()).decode())
                        return await _emit_anthropic_response(raw, tool_queue)

                    text_parts: list[str] = []
                    content_blocks: list[dict[str, Any]] = []
                    active_blocks: dict[int, dict[str, Any]] = {}
                    tool_calls: list[ParsedToolCall] = []
                    stop_reason = ""

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data:
                            continue
                        event = json.loads(data)
                        event_type = event.get("type")
                        index = int(event.get("index", 0))

                        if event_type == "content_block_start":
                            block = dict(event.get("content_block") or {})
                            if block.get("type") == "tool_use":
                                block.setdefault("input_json", "")
                            active_blocks[index] = block
                        elif event_type == "content_block_delta":
                            block = active_blocks.setdefault(index, {})
                            delta = event.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                text_parts.append(text)
                                block["type"] = "text"
                                block["text"] = block.get("text", "") + text
                            elif delta.get("type") == "input_json_delta":
                                block["input_json"] = block.get("input_json", "") + delta.get("partial_json", "")
                        elif event_type == "content_block_stop":
                            block = active_blocks.pop(index, {})
                            if block.get("type") == "tool_use":
                                parsed = _tool_block_to_call(block)
                                tool_calls.append(parsed)
                                if tool_queue is not None:
                                    await tool_queue.put(StreamingToolCall(tool_call=parsed))
                                content_blocks.append({
                                    "type": "tool_use",
                                    "id": parsed.id,
                                    "name": parsed.name,
                                    "input": parsed.args,
                                })
                            elif block:
                                content_blocks.append(block)
                        elif event_type == "message_delta":
                            delta = event.get("delta") or {}
                            stop_reason = delta.get("stop_reason") or stop_reason
                        elif event_type == "message_stop":
                            break

                    if tool_queue is not None:
                        await tool_queue.put(StreamingToolCall(is_final=True))
                    raw = {
                        "content": content_blocks,
                        "stop_reason": stop_reason or ("tool_use" if tool_calls else "end_turn"),
                    }
                    return BackendResponse(
                        content="".join(part for part in text_parts if part),
                        raw=raw,
                        tool_calls=tool_calls,
                        stop_reason=raw["stop_reason"],
                    )
        finally:
            if tool_queue is not None:
                await tool_queue.put(StreamingToolCall(is_final=True))

    def backend_capabilities(self) -> dict[str, Any]:
        return _API_MODE_CAPABILITIES.copy()


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


async def _emit_anthropic_response(raw: dict[str, Any], tool_queue: Any | None) -> BackendResponse:
    content = _extract_text(raw)
    tool_calls = _extract_tool_calls(raw)
    stop_reason = raw.get("stop_reason", "")
    if tool_queue is not None:
        for tool_call in tool_calls:
            await tool_queue.put(StreamingToolCall(tool_call=tool_call))
        await tool_queue.put(StreamingToolCall(is_final=True))
    return BackendResponse(content=content, raw=raw, tool_calls=tool_calls, stop_reason=stop_reason)


def _tool_block_to_call(block: dict[str, Any]) -> ParsedToolCall:
    raw_input = block.get("input")
    if raw_input is None:
        raw_json = block.get("input_json", "") or "{}"
        try:
            raw_input = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            return ParsedToolCall(
                id=block.get("id", "unknown"),
                name=block.get("name", "unknown"),
                args={},
                error=True,
                error_message=f"malformed input JSON: {exc}",
            )
    if not isinstance(raw_input, dict):
        return ParsedToolCall(
            id=block.get("id", "unknown"),
            name=block.get("name", "unknown"),
            args={},
            error=True,
            error_message=f"tool_use block 'input' is {type(raw_input).__name__}, expected dict",
        )
    return ParsedToolCall(
        id=block.get("id", "unknown"),
        name=block.get("name", "unknown"),
        args=raw_input,
        error=False,
    )
