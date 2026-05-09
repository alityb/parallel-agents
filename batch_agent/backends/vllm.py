"""vLLM backend adapter — OpenAI-compatible HTTP client with prefix warming,
block pinning, adaptive concurrency metrics, and KVFlow prefetch support.

Prefix block pinning (Drift 6):
  After warm_prefix(), calls POST /internal/pin_blocks with the prefix hash.
  This requires a one-line patch to vllm/core/block_manager_v1.py (or v2):

    In BlockSpaceManager.__init__:
        self.pinned_block_hashes: set[str] = set()
    In BlockSpaceManager.get_physical_blocks / free_blocks / can_allocate:
        skip eviction for blocks whose hash is in self.pinned_block_hashes

  The /internal/pin_blocks route is provided in backends/vllm_patch/prefetch_route.py.
  HARDWARE BLOCKER: patch cannot be tested without a running vLLM server and GPU.
  Recorded in LOGS.md — the code path is implemented but the actual eviction prevention
  requires the patched BlockManager running on real hardware.

Adaptive concurrency (Drift 3):
  get_cache_metrics() polls GET /v1/metrics (Prometheus text) and returns
  prefix_cache_hit_rate and gpu_utilization as floats in [0, 1].
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from .openai import OpenAIBackend
from batch_agent.spec import Message, SharedContext

logger = logging.getLogger(__name__)


class VLLMBackend(OpenAIBackend):

    @classmethod
    def from_url(cls, url: str) -> "VLLMBackend":
        parsed = urlparse(url)
        scheme = "http" if parsed.scheme == "vllm" else parsed.scheme
        base_url = f"{scheme}://{parsed.netloc}"
        return cls(api_key="EMPTY", base_url=base_url)

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        if not shared.prefix:
            return None
        prefix_hash = hashlib.sha256(shared.prefix.encode("utf-8")).hexdigest()
        async with httpx.AsyncClient(timeout=30) as client:
            # vLLM ≥0.6: use chat completions with max_tokens=1 to force prefix KV fill.
            # Legacy zero-token /v1/completions no longer works in vLLM 0.20+.
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "system", "content": shared.prefix},
                                     {"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    },
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                logger.debug("warm_prefix via chat/completions: prefix hash=%s", prefix_hash[:12])
            except Exception as e:
                logger.debug("warm_prefix failed (%s) — continuing without pre-warm", e)

        # Drift 6: pin the prefix block (requires vllm_patch on the server side).
        async with httpx.AsyncClient(timeout=5) as pclient:
            try:
                pin_resp = await pclient.post(
                    f"{self.base_url}/internal/pin_blocks",
                    json={"kv_keys": [prefix_hash]},
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                if pin_resp.status_code == 200:
                    logger.debug("Pinned prefix block %s", prefix_hash[:12])
            except Exception as e:
                logger.debug("Pin blocks call skipped (%s) — vLLM patch not installed", e)

        return prefix_hash

    def backend_capabilities(self) -> dict[str, Any]:
        return {
            "prefix_pinning": True,
            "kvflow": True,
            "diff_kv": True,
            "max_safe_concurrent": 64,
        }

    async def get_cache_metrics(self) -> dict[str, float]:
        """Poll vLLM Prometheus metrics endpoint for prefix cache hit rate and GPU util."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/metrics")
                if response.status_code != 200:
                    return {}
                text = response.text
            metrics: dict[str, float] = {}
            # Parse Prometheus text format
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                m = re.match(r"^(vllm:\w+)(?:\{[^}]*\})?\s+([\d.e+\-]+)", line)
                if not m:
                    continue
                name, value_str = m.group(1), m.group(2)
                if name.endswith("_created"):
                    continue  # skip timestamp gauges
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                if ("prefix_cache_queries_total" in name or "prefix_cache_hits_total" in name) \
                        and "external" not in name:
                    # These are token-level counters; we compute rate after both are parsed
                    if "queries" in name:
                        metrics["_prefix_cache_queries_tokens"] = value
                    else:
                        metrics["_prefix_cache_hits_tokens"] = value
                elif "prefix_cache_hit_rate" in name:
                    metrics["prefix_cache_hit_rate"] = value
                elif "gpu_cache_usage" in name:
                    metrics["gpu_utilization"] = value
            # Derive hit rate from token counters when a pre-computed rate is absent
            if "prefix_cache_hit_rate" not in metrics:
                q = metrics.pop("_prefix_cache_queries_tokens", 0)
                h = metrics.pop("_prefix_cache_hits_tokens", 0)
                if q > 0:
                    metrics["prefix_cache_hit_rate"] = h / q
                    metrics["prefix_cache_hit_tokens"] = h
                    metrics["prefix_cache_query_tokens"] = q
            return metrics
        except Exception as e:
            logger.debug("get_cache_metrics failed: %s", e)
            return {}

    async def send_prefetch_hints(self, hints: list[Any]) -> None:
        """Send KVFlow prefetch hints to vLLM /internal/prefetch.

        Requires vllm_patch/prefetch_route.py installed on the vLLM server.
        HARDWARE BLOCKER: silently no-ops if endpoint is unavailable.
        """
        if not hints:
            return
        payload_hints = [h.to_dict() if hasattr(h, "to_dict") else h for h in hints]
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/internal/prefetch",
                    json={"hints": payload_hints},
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                if response.status_code != 200:
                    logger.debug("Prefetch endpoint returned %d", response.status_code)
        except Exception as e:
            logger.debug("send_prefetch_hints skipped (%s)", e)

    async def get_queue_metrics(self) -> dict[str, Any]:
        """Return vLLM request queue depth from the Prometheus /metrics endpoint."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.base_url}/metrics")
                if response.status_code != 200:
                    return {}
                text = response.text
            metrics: dict[str, Any] = {}
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                m = re.match(r"^(vllm:\w+)(?:\{[^}]*\})?\s+([\d.e+\-]+)", line)
                if not m or m.group(1).endswith("_created"):
                    continue
                name, val_str = m.group(1), m.group(2)
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                if "num_requests_waiting" in name:
                    metrics["requests_waiting"] = int(val)
                elif "num_requests_running" in name:
                    metrics["requests_running"] = int(val)
            return metrics
        except Exception as e:
            logger.debug("get_queue_metrics failed: %s", e)
            return {}
