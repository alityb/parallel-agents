from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx

from . import BackendAdapter, BackendResponse, ParsedToolCall, StreamingToolCall
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
        _apply_request_extensions(payload, metadata)

        # Convert Anthropic-format tool schemas to OpenAI format if provided
        if tools:
            payload["tools"] = _convert_tools_to_openai(tools)

        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
        raw = response.json()

        choice = raw["choices"][0]
        message = choice.get("message") or {"content": choice.get("text", "")}
        content = _normalise_content(message.get("content"))
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
            raise RuntimeError("OPENAI_API_KEY is required for openai:// backend")

        api_messages: list[dict[str, Any]] = []
        if shared.prefix:
            system_prompt = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            api_messages.append({"role": "system", "content": system_prompt})

        if messages is not None:
            api_messages.extend(_messages_to_openai(messages))
        else:
            api_messages.append({"role": "user", "content": job.prompt})

        payload: dict[str, Any] = {"model": model, "messages": api_messages, "stream": True}
        _apply_request_extensions(payload, metadata)
        if tools:
            payload["tools"] = _convert_tools_to_openai(tools)

        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" not in content_type:
                        raw = json.loads((await response.aread()).decode())
                        return await _emit_openai_response(raw, tool_queue)

                    content_parts: list[str] = []
                    tool_chunks: dict[int, dict[str, str]] = {}
                    finish_reason = ""
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        event = json.loads(data)
                        choice = (event.get("choices") or [{}])[0]
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        for chunk in delta.get("tool_calls") or []:
                            index = int(chunk.get("index", 0))
                            entry = tool_chunks.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            if chunk.get("id"):
                                entry["id"] = chunk["id"]
                            func = chunk.get("function") or {}
                            if func.get("name"):
                                entry["name"] = func["name"]
                            if func.get("arguments"):
                                entry["arguments"] += func["arguments"]
                        finish_reason = choice.get("finish_reason") or finish_reason

                    tool_calls = _tool_chunks_to_calls(tool_chunks)
                    if tool_queue is not None:
                        for tool_call in tool_calls:
                            await tool_queue.put(StreamingToolCall(tool_call=tool_call))
                        await tool_queue.put(StreamingToolCall(is_final=True))
                    raw = {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "".join(content_parts) or None,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                                    }
                                    for tc in tool_calls if not tc.error
                                ],
                            },
                            "finish_reason": finish_reason,
                        }]
                    }
                    return BackendResponse(
                        content="".join(content_parts),
                        raw=raw,
                        tool_calls=tool_calls,
                        stop_reason="tool_use" if finish_reason == "tool_calls" or tool_calls else finish_reason,
                    )
        finally:
            if tool_queue is not None:
                # If an exception is raised before the normal final event, unblock
                # the scheduler's dispatch loop.
                await tool_queue.put(StreamingToolCall(is_final=True))

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
        raw_args = func.get("arguments", "{}")

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

        # Parse arguments JSON. SGLang can return a dict here instead of the
        # OpenAI stringified-JSON convention.
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError) as exc:
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


async def _emit_openai_response(raw: dict[str, Any], tool_queue: Any | None) -> BackendResponse:
    choice = raw["choices"][0]
    message = choice.get("message") or {"content": choice.get("text", "")}
    content = _normalise_content(message.get("content"))
    finish_reason = choice.get("finish_reason", "")
    tool_calls = _extract_tool_calls(message)
    if tool_queue is not None:
        for tool_call in tool_calls:
            await tool_queue.put(StreamingToolCall(tool_call=tool_call))
        await tool_queue.put(StreamingToolCall(is_final=True))
    return BackendResponse(
        content=content,
        raw=raw,
        tool_calls=tool_calls,
        stop_reason="tool_use" if finish_reason == "tool_calls" or tool_calls else finish_reason,
    )


def _tool_chunks_to_calls(tool_chunks: dict[int, dict[str, str]]) -> list[ParsedToolCall]:
    parsed: list[ParsedToolCall] = []
    for index in sorted(tool_chunks):
        chunk = tool_chunks[index]
        call_id = chunk.get("id", "") or f"call_{index}"
        name = chunk.get("name", "")
        arguments = chunk.get("arguments", "") or "{}"
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as exc:
            parsed.append(ParsedToolCall(
                id=call_id,
                name=name or "unknown",
                args={},
                error=True,
                error_message=f"malformed arguments JSON: {exc}",
            ))
            continue
        if not isinstance(args, dict):
            parsed.append(ParsedToolCall(
                id=call_id,
                name=name or "unknown",
                args={},
                error=True,
                error_message=f"arguments is {type(args).__name__}, expected dict",
            ))
            continue
        parsed.append(ParsedToolCall(id=call_id, name=name, args=args, error=False))
    return parsed


def _normalise_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


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


def _apply_request_extensions(payload: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    if not metadata:
        return
    extensions = metadata.get("request_extensions")
    if isinstance(extensions, dict):
        payload.update(extensions)
