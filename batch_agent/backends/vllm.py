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
            # Warm the prefix KV cache (zero-token completion forces prefill)
            response = await client.post(
                f"{self.base_url}/v1/completions",
                json={"model": model, "prompt": shared.prefix, "max_tokens": 0},
                headers={"authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()

            # Drift 6: pin the prefix block so it is never LRU-evicted.
            # Requires vllm_patch/prefetch_route.py to be installed on the vLLM server.
            # HARDWARE BLOCKER: silently skips if /internal/pin_blocks is not available.
            try:
                pin_response = await client.post(
                    f"{self.base_url}/internal/pin_blocks",
                    json={"kv_keys": [prefix_hash]},
                    headers={"authorization": f"Bearer {self.api_key}"},
                    timeout=5.0,
                )
                if pin_response.status_code == 200:
                    logger.debug("Pinned prefix block %s", prefix_hash[:12])
                else:
                    logger.debug(
                        "Pin blocks endpoint returned %d (patch not installed?)",
                        pin_response.status_code,
                    )
            except Exception as e:
                logger.debug("Pin blocks call skipped (%s) — vLLM patch not installed", e)

        return prefix_hash

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
                m = re.match(r"^(vllm:\w+)\s+([\d.e+\-]+)", line)
                if not m:
                    continue
                name, value_str = m.group(1), m.group(2)
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                if "prefix_cache_hit_rate" in name:
                    metrics["prefix_cache_hit_rate"] = value
                elif "gpu_cache_usage" in name:
                    metrics["gpu_utilization"] = value
            return metrics
        except Exception as e:
            logger.debug("get_cache_metrics failed: %s", e)
            return {}

    async def send_prefetch_hints(self, hints: list[dict[str, Any]]) -> None:
        """Send KVFlow prefetch hints to vLLM /internal/prefetch.

        Requires vllm_patch/prefetch_route.py installed on the vLLM server.
        HARDWARE BLOCKER: silently no-ops if endpoint is unavailable.
        """
        if not hints:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/internal/prefetch",
                    json={"hints": hints},
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                if response.status_code != 200:
                    logger.debug("Prefetch endpoint returned %d", response.status_code)
        except Exception as e:
            logger.debug("send_prefetch_hints skipped (%s)", e)
