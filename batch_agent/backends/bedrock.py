"""AWS Bedrock Converse backend adapter.

Uses the Bedrock Converse API via boto3 (both converse_stream for real calls
and converse for test/fallback).  Credentials come entirely from the standard
boto3 credential chain — no hardcoded keys.

URL format:
  bedrock://us-east-1/anthropic.claude-sonnet-4-5   explicit region + model override
  bedrock://us-east-1                                explicit region, model from BatchSpec
  bedrock://anthropic.claude-sonnet-4-5              model override, region from AWS config
  bedrock://                                         region and model from AWS config / BatchSpec

The model parameter passed to generate() always wins over the URL-parsed model.

Prompt caching:
  Bedrock supports Anthropic prompt caching via cachePoint blocks in the Converse
  system array.  Only Claude (anthropic.*) models support this; all others silently
  skip the cachePoint injection.

Streaming:
  converse_stream is used by default.  boto3 is synchronous, so stream iteration
  runs in asyncio.to_thread().  converse_stream support on Bedrock depends on the
  model; Claude and Llama 3.x are supported.  Titan and some older models are not —
  the adapter falls back to converse() if converse_stream raises an unsupported error.

Tool calling:
  Bedrock Converse uses camelCase keys and a different nesting than Anthropic native:
    toolUse.toolUseId  (vs tool_use.id)
    toolUse.input      (dict, already parsed — not a JSON string)
    Bedrock streaming delivers input as a concatenated JSON string across deltas.

KV cache control:
  Bedrock is a managed service.  warm_prefix is a no-op (no /v1/completions endpoint).
  Prefix caching effectiveness depends on Bedrock's internal implementation with cache
  points; there is no way to pin blocks or query hit rates.  get_cache_metrics always
  returns {} for Bedrock backends.

LOGS.md references:
  - Bedrock prompt caching (Claude on Bedrock) added ~2024-07.  Llama/Titan: not supported.
  - converse_stream: Claude 3/3.5, Llama 3.x supported; older Titan not.
  - warm_prefix and KV pin: not applicable to managed Bedrock — same limitation as
    the Anthropic API adapter (degraded mode per spec §3.4.3).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from . import BackendAdapter, BackendResponse, ParsedToolCall
from ..spec import AgentJob, Message, SharedContext
from ..utils import (
    INTERNAL_HTTP_TIMEOUT,
    NO_API_KEY,
    DEFAULT_MAX_TOKENS,
    prefix_hash,
    strip_preamble_headers,
)

logger = logging.getLogger(__name__)

_MODE_LIMITATIONS = """
Bedrock backend mode limitations:
1. Standard Bedrock quotas are low for large Claude/Opus profiles. Start with
   max_concurrent=1-3 to avoid ThrottlingException. The AIMD controller handles
   bursts and quota discovery, but sustained higher parallelism requires an AWS
   Bedrock quota increase.
2. KVFlow prefetch and TokenDance diff KV do not apply. Bedrock is managed and
   exposes no KV cache block IDs, pinning API, prefetch API, or cache eviction
   controls.
3. In Bedrock mode the SDK value is tool deduplication, structured output
   validation, retry/failure handling, result streaming, and Bedrock cachePoint
   management. It is not GPU scheduling efficiency or KV cache co-design.
"""


@dataclass
class BedrockConcurrencyController:
    """AIMD quota-aware controller for Bedrock managed-service throttling."""

    max_concurrent_ceiling: int = 3
    current_limit: int = 1
    clock: Any = time.monotonic
    quiet_window_seconds: float = 60.0
    last_throttle_at: float | None = None
    last_increase_at: float | None = None

    def record_throttle(self) -> int:
        self.last_throttle_at = self.clock()
        self.current_limit = max(1, self.current_limit // 2)
        return self.current_limit

    def maybe_increase(self) -> int:
        now = self.clock()
        if self.current_limit >= self.max_concurrent_ceiling:
            return self.current_limit
        if self.last_throttle_at is not None and now - self.last_throttle_at < self.quiet_window_seconds:
            return self.current_limit
        if self.last_increase_at is not None and now - self.last_increase_at < self.quiet_window_seconds:
            return self.current_limit
        self.current_limit += 1
        self.last_increase_at = now
        return self.current_limit


# ── model-capability tables ────────────────────────────────────────────────────

def _supports_prompt_caching(model_id: str) -> bool:
    """Only anthropic.claude-* models support cachePoint on Bedrock."""
    parts = model_id.lower().split(".")
    vendor = parts[1] if parts and parts[0] in {"us", "eu", "apac"} and len(parts) > 1 else parts[0]
    return vendor == "anthropic"


def _supports_streaming(model_id: str) -> bool:
    """converse_stream is supported for most modern Bedrock models.

    Conservative allow-list: claude, llama3.x, mistral.
    Falls back to converse() on any error anyway.
    """
    parts = model_id.lower().split(".")
    vendor = parts[1] if parts and parts[0] in {"us", "eu", "apac"} and len(parts) > 1 else parts[0]
    return vendor in {"anthropic", "meta", "mistral", "amazon"}


# ── adapter ────────────────────────────────────────────────────────────────────

class BedrockBackend(BackendAdapter):

    def __init__(
        self,
        region: str | None = None,
        model_id_override: str | None = None,
        max_concurrent_ceiling: int = 3,
        *,
        _client_factory: Any = None,  # injectable for testing
    ) -> None:
        self.region = region
        self.model_id_override = model_id_override
        self._client_factory = _client_factory  # if None, uses boto3 default chain
        self.request_metrics: list[dict[str, Any]] = []
        self.request_payloads: list[dict[str, Any]] = []
        self.concurrency_controller = BedrockConcurrencyController(
            max_concurrent_ceiling=max(1, max_concurrent_ceiling)
        )

    @classmethod
    def from_url(cls, url: str) -> "BedrockBackend":
        """Parse bedrock://[region/][model-id] URL.

        Heuristic: if the first path component contains a dot it is a model ID
        (e.g. "anthropic.claude-sonnet-4-5"); otherwise it is a region
        (e.g. "us-east-1").
        """
        parsed = urlparse(url)
        netloc = parsed.netloc  # may be "us-east-1" or "anthropic.claude-x"
        path = parsed.path.lstrip("/")  # may be empty or "anthropic.claude-x"

        region: str | None = None
        model_id_override: str | None = None

        if netloc and path:
            # bedrock://region/model-id
            region = netloc
            model_id_override = path
        elif netloc and "." in netloc:
            # bedrock://anthropic.claude-x  — no region, model in netloc
            model_id_override = netloc
        elif netloc:
            # bedrock://us-east-1  — just a region
            region = netloc
        # else: bedrock:// — use defaults from env/config

        return cls(region=region, model_id_override=model_id_override)

    # ── BackendAdapter interface ───────────────────────────────────────────────

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        """Bedrock is a managed service — no warm_prefix endpoint.

        Returns a stable hash of the prefix for use as kv_key (e.g. by KVFlow),
        but does NOT pre-fill any server-side cache.  Actual cache warming happens
        implicitly via Bedrock's cachePoint mechanism on the first real call.
        """
        if not shared.prefix:
            return None
        prefix = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
        return prefix_hash(prefix)

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
        # Model resolution: explicit call wins over URL override
        model_id = self.model_id_override if self.model_id_override else model

        client = self._get_client()

        # Build Converse payload
        payload: dict[str, Any] = {"modelId": model_id}

        # System prompt with optional cachePoint (Claude on Bedrock only)
        if shared.prefix:
            system_prompt = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            system_block: dict[str, Any] = {"text": system_prompt}
            if _supports_prompt_caching(model_id):
                payload["system"] = [system_block, {"cachePoint": {"type": "default"}}]
            else:
                payload["system"] = [system_block]

        # Messages
        if messages is not None:
            payload["messages"] = _messages_to_bedrock(messages)
        else:
            payload["messages"] = [{"role": "user", "content": [{"text": job.prompt}]}]

        # Tool configuration (Anthropic-format schemas → Bedrock toolSpec)
        if tools:
            payload["toolConfig"] = {"tools": _convert_tools_to_bedrock(tools)}

        # Inference configuration
        payload["inferenceConfig"] = {"maxTokens": DEFAULT_MAX_TOKENS}
        self.request_payloads.append(payload)

        # Try streaming first; fall back to non-streaming on error
        try:
            if _supports_streaming(model_id):
                text, tool_calls, stop_reason, raw = await asyncio.wait_for(
                    asyncio.to_thread(_sync_stream, client, payload),
                    timeout=timeout,
                )
            else:
                text, tool_calls, stop_reason, raw = await asyncio.wait_for(
                    asyncio.to_thread(_sync_converse, client, payload),
                    timeout=timeout,
                )
        except Exception as exc:
            # If streaming fails (model doesn't support it), fall back to converse
            msg = str(exc).lower()
            if "throttling" in msg or "too many requests" in msg:
                self.concurrency_controller.record_throttle()
            if "prompt caching" in msg or "cachepoint" in msg:
                logger.debug("Bedrock cachePoint rejected for %s; retrying without cachePoint", model_id)
                payload = _without_cache_point(payload)
                if _supports_streaming(model_id):
                    text, tool_calls, stop_reason, raw = await asyncio.wait_for(
                        asyncio.to_thread(_sync_stream, client, payload),
                        timeout=timeout,
                    )
                else:
                    text, tool_calls, stop_reason, raw = await asyncio.wait_for(
                        asyncio.to_thread(_sync_converse, client, payload),
                        timeout=timeout,
                    )
            elif "stream" in msg or "unsupported" in msg:
                logger.debug(
                    "converse_stream failed (%s), falling back to converse()", exc
                )
                text, tool_calls, stop_reason, raw = await asyncio.wait_for(
                    asyncio.to_thread(_sync_converse, client, payload),
                    timeout=timeout,
                )
            else:
                raise

        self.concurrency_controller.maybe_increase()
        self.request_metrics.append({
            **raw.get("metrics", {}),
            "usage": raw.get("usage", {}),
            "stop_reason": stop_reason,
            "cachePointRequested": any("cachePoint" in block for block in payload.get("system", [])),
        })
        return BackendResponse(
            content=text,
            raw=raw,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )

    async def get_cache_metrics(self) -> dict[str, float]:
        """Bedrock does not expose KV cache internals — always empty."""
        return {"recommended_concurrency": float(self.concurrency_controller.maybe_increase())}

    async def send_prefetch_hints(self, hints: list[dict[str, Any]]) -> None:
        """No-op: Bedrock is a managed service with no KV prefetch API."""
        logger.debug("send_prefetch_hints: no-op for Bedrock backend (%d hints ignored)", len(hints))

    def backend_capabilities(self) -> dict[str, Any]:
        # See LOGS.md "Authoritative Bedrock cache latency isolation".
        # Bedrock cachePoint produced confirmed cache token reads/writes, but
        # cache-hit TTFT was not lower for a ~1,200-token prefix because managed
        # queue/model latency dominated observed first-token latency.
        return {
            "prefix_pinning": False,
            "prompt_cache_token_savings": True,
            "prompt_cache_latency_benefit": False,
            "kvflow": False,
            "diff_kv": False,
            "max_safe_concurrent": self.concurrency_controller.current_limit,
        }

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for the Bedrock backend: pip install boto3"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.region:
            kwargs["region_name"] = self.region
        kwargs["config"] = Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=10,
        )
        return boto3.client("bedrock-runtime", **kwargs)


def _without_cache_point(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the payload with Converse cachePoint blocks removed."""
    copied = dict(payload)
    if "system" in copied:
        copied["system"] = [block for block in copied["system"] if "cachePoint" not in block]
    return copied


# ── sync helpers (run inside asyncio.to_thread) ────────────────────────────────

def _sync_stream(
    client: Any, payload: dict[str, Any]
) -> tuple[str, list[ParsedToolCall], str, dict[str, Any]]:
    """Run converse_stream and collect the full response."""
    started = time.monotonic()
    response = client.converse_stream(**payload)
    stream = response.get("stream", [])

    texts: list[str] = []
    tool_blocks: dict[int, dict[str, Any]] = {}   # block index → accumulator
    stop_reason = "end_turn"
    last_event: dict[str, Any] = {}
    ttft_seconds: float | None = None
    usage: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    raw_events: list[dict[str, Any]] = []

    for event in stream:
        last_event = event
        raw_events.append(event)

        if "contentBlockStart" in event:
            data = event["contentBlockStart"]
            idx: int = data["contentBlockIndex"]
            start = data.get("start", {})
            if "toolUse" in start:
                tu = start["toolUse"]
                tool_blocks[idx] = {
                    "toolUseId": tu.get("toolUseId", ""),
                    "name": tu.get("name", ""),
                    "input_json": "",
                }

        elif "contentBlockDelta" in event:
            data = event["contentBlockDelta"]
            idx = data["contentBlockIndex"]
            delta = data.get("delta", {})
            if "text" in delta:
                if ttft_seconds is None:
                    ttft_seconds = time.monotonic() - started
                texts.append(delta["text"])
            elif "toolUse" in delta and idx in tool_blocks:
                if ttft_seconds is None:
                    ttft_seconds = time.monotonic() - started
                tool_blocks[idx]["input_json"] += delta["toolUse"].get("input", "")

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
        elif "metadata" in event:
            metadata = event["metadata"]
            usage = metadata.get("usage", {})
            metrics = metadata.get("metrics", {})

    text = "".join(texts)
    tool_calls = _parse_bedrock_tool_blocks(tool_blocks)
    # Normalize Bedrock stop reasons to match our is_final convention
    if stop_reason == "tool_use":
        normalized_stop = "tool_use"
    else:
        normalized_stop = stop_reason  # "end_turn", "max_tokens", "stop_sequence"

    total_seconds = time.monotonic() - started
    return text, tool_calls, normalized_stop, {
        "last_event": last_event,
        "events": raw_events,
        "usage": usage,
        "metrics": {
            **metrics,
            "ttft_seconds": ttft_seconds,
            "total_seconds": total_seconds,
        },
    }


def _sync_converse(
    client: Any, payload: dict[str, Any]
) -> tuple[str, list[ParsedToolCall], str, dict[str, Any]]:
    """Run converse (non-streaming) and return the full response."""
    response = client.converse(**payload)
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks: list[dict[str, Any]] = message.get("content", [])
    stop_reason = response.get("stopReason", "end_turn")

    texts: list[str] = []
    tool_blocks: dict[int, dict[str, Any]] = {}

    for idx, block in enumerate(content_blocks):
        if "text" in block:
            texts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            raw_input = tu.get("input", {})
            # converse() returns input as a dict (already parsed); stream delivers JSON string
            tool_blocks[idx] = {
                "toolUseId": tu.get("toolUseId", ""),
                "name": tu.get("name", ""),
                "input_json": json.dumps(raw_input) if isinstance(raw_input, dict) else str(raw_input),
            }

    text = "".join(texts)
    if stop_reason == "tool_use":
        normalized_stop = "tool_use"
    else:
        normalized_stop = stop_reason

    response.setdefault("metrics", {})["ttft_seconds"] = None
    return text, _parse_bedrock_tool_blocks(tool_blocks), normalized_stop, response


def _parse_bedrock_tool_blocks(
    tool_blocks: dict[int, dict[str, Any]]
) -> list[ParsedToolCall]:
    """Convert accumulated Bedrock toolUse blocks → ParsedToolCall list.

    Does NOT skip malformed blocks — logs them and returns with error=True.
    """
    tool_calls: list[ParsedToolCall] = []

    for idx in sorted(tool_blocks.keys()):
        tb = tool_blocks[idx]
        tool_id = tb.get("toolUseId", "")
        name = tb.get("name", "")
        input_json = tb.get("input_json", "")

        if not tool_id:
            logger.warning("Bedrock toolUse block at index %d missing toolUseId", idx)
            tool_calls.append(ParsedToolCall(
                id="unknown", name=name or "unknown", args={},
                error=True, error_message="toolUse missing toolUseId",
            ))
            continue

        if not name:
            logger.warning("Bedrock toolUse block at index %d missing name", idx)
            tool_calls.append(ParsedToolCall(
                id=tool_id, name="unknown", args={},
                error=True, error_message="toolUse missing name",
            ))
            continue

        try:
            args = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            logger.warning("Bedrock toolUse '%s' has malformed input JSON: %s", name, exc)
            tool_calls.append(ParsedToolCall(
                id=tool_id, name=name, args={},
                error=True, error_message=f"malformed input JSON: {exc}",
            ))
            continue

        if not isinstance(args, dict):
            logger.warning(
                "Bedrock toolUse '%s' input is %s, expected dict", name, type(args).__name__
            )
            tool_calls.append(ParsedToolCall(
                id=tool_id, name=name, args={},
                error=True, error_message=f"input is {type(args).__name__}, expected dict",
            ))
            continue

        tool_calls.append(ParsedToolCall(id=tool_id, name=name, args=args, error=False))

    return tool_calls


# ── message format conversion ──────────────────────────────────────────────────

def _messages_to_bedrock(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message objects to Bedrock Converse message format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            result.append({"role": "user", "content": [{"text": msg.content}]})

        elif msg.role == "assistant":
            result.append({"role": "assistant", "content": [{"text": msg.content}]})

        elif msg.role == "assistant_raw":
            # JSON-encoded list of Anthropic-format content blocks
            try:
                blocks: list[dict[str, Any]] = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                result.append({"role": "assistant", "content": [{"text": msg.content}]})
                continue

            bedrock_content: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    bedrock_content.append({"text": block.get("text", "")})
                elif block.get("type") == "tool_use":
                    bedrock_content.append({
                        "toolUse": {
                            "toolUseId": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                        }
                    })
            if bedrock_content:
                result.append({"role": "assistant", "content": bedrock_content})

        elif msg.role == "tool_result":
            # JSON-encoded list of Anthropic tool_result blocks → Bedrock toolResult
            try:
                blocks_raw: list[dict[str, Any]] = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                result.append({"role": "user", "content": [{"text": msg.content}]})
                continue

            bedrock_tool_results: list[dict[str, Any]] = []
            for block in blocks_raw:
                if not isinstance(block, dict):
                    continue
                tool_use_id = block.get("tool_use_id", "unknown")
                content_text = block.get("content", "")
                is_error = block.get("is_error", False)

                tr: dict[str, Any] = {
                    "toolUseId": tool_use_id,
                    "content": [{"text": content_text if isinstance(content_text, str) else json.dumps(content_text)}],
                }
                if is_error:
                    tr["status"] = "error"
                bedrock_tool_results.append({"toolResult": tr})

            if bedrock_tool_results:
                result.append({"role": "user", "content": bedrock_tool_results})

    return result


def _convert_tools_to_bedrock(
    anthropic_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert Anthropic-format tool schemas to Bedrock toolSpec format.

    Anthropic:  {name, description, input_schema: {type, properties, required}}
    Bedrock:    {toolSpec: {name, description, inputSchema: {json: {type, properties, required}}}}
    """
    bedrock_tools: list[dict[str, Any]] = []
    for tool in anthropic_tools:
        bedrock_tools.append({
            "toolSpec": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": {
                    "json": tool.get("input_schema", {"type": "object", "properties": {}})
                },
            }
        })
    return bedrock_tools
