from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx

from . import BackendAdapter, BackendResponse, ParsedToolCall
from .anthropic import _API_MODE_CAPABILITIES
from ..spec import AgentJob, Message, SharedContext
from ..utils import strip_preamble_headers

logger = logging.getLogger(__name__)


class OpenAIBackend(BackendAdapter):
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openai.com") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_url(cls, url: str) -> "OpenAIBackend":
        parsed = urlparse(url)
        base_url = f"https://{parsed.netloc}" if parsed.netloc else "https://api.openai.com"
        return cls(base_url=base_url)

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
            raise RuntimeError("OPENAI_API_KEY is required for openai:// backend")

        api_messages: list[dict[str, Any]] = []
        if shared.prefix:
            system_prompt = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            api_messages.append({"role": "system", "content": system_prompt})

        if messages is not None:
            api_messages.extend(_messages_to_openai(messages))
        else:
            api_messages.append({"role": "user", "content": job.prompt})

        payload: dict[str, Any] = {"model": model, "messages": api_messages}

        # Convert Anthropic-format tool schemas to OpenAI format if provided
        if tools:
            payload["tools"] = _convert_tools_to_openai(tools)

        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
        raw = response.json()

        choice = raw["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "")
        tool_calls = _extract_tool_calls(message)

        # Map OpenAI finish_reason to Anthropic-like stop_reason for is_final
        stop_reason = "tool_use" if finish_reason == "tool_calls" or tool_calls else finish_reason

        return BackendResponse(
            content=content,
            raw=raw,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )

    def backend_capabilities(self) -> dict[str, Any]:
        return _API_MODE_CAPABILITIES.copy()


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message objects to OpenAI API format."""
    api_msgs: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool_result":
            # Tool results — parse the JSON list of tool_result blocks
            try:
                blocks = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                blocks = [{"tool_use_id": "unknown", "content": msg.content}]
            for block in blocks:
                api_msgs.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", "unknown"),
                    "content": block.get("content", ""),
                })
        elif msg.role == "assistant_raw":
            # Reconstruct assistant message with tool_calls from stored content blocks
            try:
                blocks = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                api_msgs.append({"role": "assistant", "content": msg.content})
                continue

            text_parts = []
            tool_calls_list = []
            for block in blocks:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls_list.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "\n".join(text_parts)
            else:
                assistant_msg["content"] = None
            if tool_calls_list:
                assistant_msg["tool_calls"] = tool_calls_list
            api_msgs.append(assistant_msg)
        else:
            api_msgs.append({"role": msg.role, "content": msg.content})
    return api_msgs


def _extract_tool_calls(message: dict[str, Any]) -> list[ParsedToolCall]:
    """Parse OpenAI-format tool_calls from assistant message."""
    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return []

    parsed: list[ParsedToolCall] = []
    for tc in raw_calls:
        call_id = tc.get("id", "")
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        if not call_id:
            logger.warning("OpenAI tool_call missing 'id': %s", tc)
            parsed.append(ParsedToolCall(
                id="unknown", name=name or "unknown", args={},
                error=True, error_message="tool_call missing 'id'",
            ))
            continue

        if not name:
            logger.warning("OpenAI tool_call missing function name: %s", tc)
            parsed.append(ParsedToolCall(
                id=call_id, name="unknown", args={},
                error=True, error_message="tool_call missing function name",
            ))
            continue

        # Parse arguments JSON
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError as exc:
            logger.warning("OpenAI tool_call '%s' has malformed arguments: %s", name, exc)
            parsed.append(ParsedToolCall(
                id=call_id, name=name, args={},
                error=True, error_message=f"malformed arguments JSON: {exc}",
            ))
            continue

        if not isinstance(args, dict):
            logger.warning("OpenAI tool_call '%s' arguments not a dict: %s", name, type(args))
            parsed.append(ParsedToolCall(
                id=call_id, name=name, args={},
                error=True, error_message=f"arguments is {type(args).__name__}, expected dict",
            ))
            continue

        parsed.append(ParsedToolCall(id=call_id, name=name, args=args, error=False))

    return parsed


def _convert_tools_to_openai(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-format tool schemas to OpenAI function calling format."""
    openai_tools: list[dict[str, Any]] = []
    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools
