"""SGLang backend adapter.

SGLang (>= 0.2) supports an OpenAI-compatible API at /v1/chat/completions,
so most of our OpenAI adapter works unchanged.  The key differences:

1. warm_prefix: SGLang uses RadixAttention — prefix KV is cached automatically
   at the token-tree level. A zero-token warm-up request still helps prime the
   tree, but we don't need to call /internal/pin_blocks (RadixAttention keeps
   frequently-accessed prefix nodes hot by its own scoring).

2. /generate: SGLang also provides a native /generate endpoint with a richer
   payload (sampling_params, rid, etc.).  We use the OpenAI-compat path by
   default for portability; set use_native=True to route through /generate.

3. send_prefetch_hints: SGLang's RadixAttention tracks prefix trees.
   Prefetching for a specific agent means requesting that its context subtree
   be promoted in the radix tree's LRU.  We call POST /internal/prefetch_radix
   if available; silently skip otherwise (same as vLLM path).

4. get_cache_metrics: SGLang exposes Prometheus metrics at /metrics,
   including sglang:token_usage and cache hit stats.

HARDWARE BLOCKER: all hardware-dependent paths (warm_prefix against a real
SGLang server, prefetch hints, metrics) are implemented but cannot be live-
tested without a GPU and SGLang installation.  Recorded in LOGS.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from . import BackendAdapter, BackendResponse, ParsedToolCall
from .openai import OpenAIBackend, _extract_tool_calls, _messages_to_openai, _convert_tools_to_openai
from batch_agent.spec import AgentJob, Message, SharedContext

logger = logging.getLogger(__name__)


class SGLangBackend(OpenAIBackend):
    """Full SGLang adapter.

    Inherits OpenAI-compat generate() from OpenAIBackend.
    Overrides warm_prefix, get_cache_metrics, and adds send_prefetch_hints.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:30000",
        use_native: bool = False,
    ) -> None:
        super().__init__(api_key=api_key or "EMPTY", base_url=base_url)
        self.use_native = use_native

    @classmethod
    def from_url(cls, url: str) -> "SGLangBackend":
        parsed = urlparse(url)
        scheme = "http" if parsed.scheme == "sglang" else parsed.scheme
        base_url = f"{scheme}://{parsed.netloc}"
        return cls(base_url=base_url)

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        if not shared.prefix:
            return None
        prefix_hash = hashlib.sha256(shared.prefix.encode("utf-8")).hexdigest()
        # SGLang RadixAttention caches automatically; a warm-up call primes the tree
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                if self.use_native:
                    # Native /generate endpoint
                    response = await client.post(
                        f"{self.base_url}/generate",
                        json={
                            "text": shared.prefix,
                            "sampling_params": {"max_new_tokens": 0},
                        },
                        headers={"authorization": f"Bearer {self.api_key}"},
                    )
                else:
                    # OpenAI-compat: zero-token completion
                    response = await client.post(
                        f"{self.base_url}/v1/completions",
                        json={"model": model, "prompt": shared.prefix, "max_tokens": 0},
                        headers={"authorization": f"Bearer {self.api_key}"},
                    )
                if response.status_code in (200, 400):
                    # 400 is acceptable for max_tokens=0 on some backends
                    logger.debug("SGLang prefix warmed, hash=%s", prefix_hash[:12])
            except Exception as e:
                logger.debug("SGLang warm_prefix failed: %s (continuing)", e)
        return prefix_hash

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
        if self.use_native:
            return await self._generate_native(
                shared=shared, job=job, messages=messages,
                model=model, tools=tools, metadata=metadata, timeout=timeout,
            )
        # Default: OpenAI-compat path
        return await super().generate(
            shared=shared, job=job, messages=messages,
            model=model, tools=tools, metadata=metadata, timeout=timeout,
        )

    async def _generate_native(
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
        """SGLang native /generate endpoint.

        Uses system_prompt + input_ids pattern native to SGLang.
        Tool calling in native mode is not yet supported by SGLang natively;
        falls back to prompt-based tool injection if tools are requested.
        """
        system_text = shared.prefix or ""
        if messages is not None:
            # Build a text prompt from message history
            prompt_parts = []
            if system_text:
                prompt_parts.append(f"System: {system_text}\n")
            for msg in messages:
                role = msg.role
                if role in ("assistant_raw", "tool_result"):
                    role = "assistant" if role == "assistant_raw" else "tool"
                prompt_parts.append(f"{role.capitalize()}: {msg.content}\n")
            prompt_parts.append("Assistant: ")
            prompt = "".join(prompt_parts)
        else:
            prompt = f"{system_text}\n\nUser: {job.prompt}\nAssistant: " if system_text else job.prompt

        payload: dict[str, Any] = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 4096,
                "temperature": 0.0,
                "stop": ["\nUser:", "\nSystem:"],
            },
        }
        if tools:
            # Inject tool descriptions as text (native mode doesn't have structured tools)
            tool_desc = json.dumps(tools, indent=2)
            payload["text"] = f"Available tools:\n{tool_desc}\n\n{prompt}"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/generate",
                json=payload,
                headers={"authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
        raw = response.json()
        content = raw.get("text", "")
        return BackendResponse(content=content, raw=raw, stop_reason="end_turn")

    async def get_cache_metrics(self) -> dict[str, float]:
        """Parse SGLang Prometheus metrics for cache stats."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/metrics")
                if response.status_code != 200:
                    return {}
            text = response.text
            metrics: dict[str, float] = {}
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                m = re.match(r"^(sglang:\w+)\s+([\d.e+\-]+)", line)
                if not m:
                    continue
                name, value_str = m.group(1), m.group(2)
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                if "cache_hit" in name.lower():
                    metrics["prefix_cache_hit_rate"] = value
                elif "token_usage" in name:
                    metrics["gpu_utilization"] = value
            return metrics
        except Exception as e:
            logger.debug("SGLang get_cache_metrics failed: %s", e)
            return {}

    async def send_prefetch_hints(self, hints: list[Any]) -> None:
        """Send KVFlow prefetch hints to SGLang's RadixAttention tree.

        Calls POST /internal/prefetch_radix — promotes context subtrees in the
        radix tree's eviction score so they survive longer in cache.

        HARDWARE BLOCKER: silently no-ops if endpoint is unavailable.
        """
        if not hints:
            return
        payload_hints = [h.to_dict() if hasattr(h, "to_dict") else h for h in hints]
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/internal/prefetch_radix",
                    json={"hints": payload_hints},
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                if response.status_code != 200:
                    logger.debug("SGLang prefetch_radix returned %d", response.status_code)
        except Exception as e:
            logger.debug("send_prefetch_hints (sglang) skipped: %s", e)
