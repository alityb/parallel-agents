"""NVIDIA Dynamo-compatible backend adapter.

Dynamo exposes an OpenAI/vLLM-compatible HTTP surface plus NVIDIA-specific
extensions such as ``nvext.agent_hints`` and streaming ``tool_call_dispatch``
events. This adapter keeps the vLLM request shape and adds those extensions
only when BatchSpec enables them.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from . import BackendResponse, ParsedToolCall, StreamingToolCall, _http_url_from_scheme
from .openai import OpenAIBackend, _convert_tools_to_openai, _emit_openai_response, _messages_to_openai
from .vllm import VLLMBackend
from ..spec import AgentJob, Message, SharedContext
from ..utils import NO_API_KEY, strip_preamble_headers


def _build_nvext_hints(job: Any) -> dict[str, Any] | None:
    """Map KVFlow Advisor state to nvext.agent_hints for Dynamo backends.

    Fields:
        latency_sensitivity: 0.0-1.0, higher = needs faster response
        priority: integer, higher = schedule sooner
        speculative_prefill: True if KV blocks are expected to be cached
        osl: estimated output sequence length in tokens
    """
    state = getattr(job, "state", None)
    if state is None:
        return None
    steps_to_execution = getattr(state, "steps_to_execution", None)
    if steps_to_execution is None:
        return None

    max_turns = getattr(state, "max_turns", getattr(job, "max_turns", 1))
    turn = getattr(state, "turn", 0)
    kv_key = getattr(state, "kv_key", None)
    return {
        "nvext": {
            "agent_hints": {
                "latency_sensitivity": min(1.0, 1.0 / max(0.1, float(steps_to_execution))),
                "priority": int(max_turns) - int(turn),
                "speculative_prefill": kv_key is not None,
                "osl": 512,
            }
        }
    }


def _build_nvext_hints_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata or not metadata.get("nvext_agent_hints"):
        return None
    steps_to_execution = metadata.get("steps_to_execution")
    if steps_to_execution is None:
        return None
    return {
        "nvext": {
            "agent_hints": {
                "latency_sensitivity": min(1.0, 1.0 / max(0.1, float(steps_to_execution))),
                "priority": int(metadata.get("max_turns", 1)) - int(metadata.get("turn", 0)),
                "speculative_prefill": metadata.get("kv_key") is not None,
                "osl": 512,
            }
        }
    }


class DynamoBackend(VLLMBackend):
    def __init__(self, api_key: str | None = None, base_url: str = "http://localhost:8000", **kwargs: Any) -> None:
        super().__init__(api_key=api_key or NO_API_KEY, base_url=base_url, **kwargs)

    @classmethod
    def from_url(cls, url: str) -> "DynamoBackend":
        return cls(api_key=NO_API_KEY, base_url=_http_url_from_scheme(url, "dynamo"))

    def backend_capabilities(self) -> dict[str, Any]:
        caps = super().backend_capabilities()
        caps.update({
            "nvext_agent_hints": True,
            "streaming_tool_dispatch": True,
            "strip_anthropic_preamble": True,
        })
        return caps

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
        metadata = dict(metadata or {})
        hints = _build_nvext_hints_from_metadata(metadata)
        if hints:
            metadata["request_extensions"] = hints
        # Dynamo accepts nvext.agent_hints but rejects vLLM's top-level
        # request_id extension. Call the OpenAI-compatible base directly so
        # VLLMBackend._with_vllm_request_id() is not applied.
        return await OpenAIBackend.generate(
            self,
            shared=shared,
            job=job,
            messages=messages,
            model=model,
            tools=tools,
            metadata=metadata,
            timeout=timeout,
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
        metadata = dict(metadata or {})
        hints = _build_nvext_hints_from_metadata(metadata)
        if hints:
            metadata["request_extensions"] = hints

        api_messages: list[dict[str, Any]] = []
        if shared.prefix:
            system_prompt = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            api_messages.append({"role": "system", "content": system_prompt})
        if messages is not None:
            api_messages.extend(_messages_to_openai(messages))
        else:
            api_messages.append({"role": "user", "content": job.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "stream": True,
        }
        if metadata.get("max_tokens") is not None:
            payload["max_tokens"] = metadata["max_tokens"]
        if hints:
            payload.update(hints)
        if tools:
            payload["tools"] = _convert_tools_to_openai(tools)

        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raw = json.loads((await response.aread()).decode())
                    return await _emit_openai_response(raw, tool_queue)
                return await self._parse_dynamo_stream(response, tool_queue)

    async def _parse_dynamo_stream(self, response: httpx.Response, tool_queue: Any | None) -> BackendResponse:
        content_parts: list[str] = []
        tool_calls: list[ParsedToolCall] = []
        event_name = ""
        finish_reason = ""
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
                if event_name == "error":
                    raise RuntimeError("Dynamo streaming error event")
                continue
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            event_type = event.get("type") or event_name
            if event_type == "error":
                raise RuntimeError(f"Dynamo streaming error: {event}")
            if event_type == "tool_call_dispatch":
                parsed = _parse_dynamo_tool_call(event.get("tool_call") or event)
                tool_calls.append(parsed)
                if tool_queue is not None:
                    await tool_queue.put(StreamingToolCall(tool_call=parsed))
                continue

            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            finish_reason = choice.get("finish_reason") or finish_reason

        if tool_queue is not None:
            await tool_queue.put(StreamingToolCall(is_final=True))
        return BackendResponse(
            content="".join(content_parts),
            raw={"content": "".join(content_parts), "tool_calls": [tc.__dict__ for tc in tool_calls]},
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else (finish_reason or "stop"),
        )


def _parse_dynamo_tool_call(raw: dict[str, Any]) -> ParsedToolCall:
    call_id = raw.get("id", "unknown")
    function = raw.get("function") or {}
    name = raw.get("name") or function.get("name", "unknown")
    args_raw = raw.get("args", raw.get("arguments", function.get("arguments", {})))
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw or "{}")
        except json.JSONDecodeError as exc:
            return ParsedToolCall(id=call_id, name=name, args={}, error=True, error_message=str(exc))
    else:
        args = args_raw or {}
    if not isinstance(args, dict):
        return ParsedToolCall(
            id=call_id,
            name=name,
            args={},
            error=True,
            error_message=f"arguments is {type(args).__name__}, expected dict",
        )
    return ParsedToolCall(id=call_id, name=name, args=args, error=False)
