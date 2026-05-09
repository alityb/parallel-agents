"""vLLM backend adapter — OpenAI-compatible HTTP client with prefix warming,
block pinning, adaptive concurrency metrics, and KVFlow prefetch support.

Prefix block pinning (Drift 6):
  After warm_prefix(), calls POST /internal/pin_blocks with the prefix hash.
  Requires the vllm_patch/prefetch_route.py to be installed on the vLLM server.
  HARDWARE BLOCKER: patch cannot be tested without a running vLLM server and GPU.

Adaptive concurrency (Drift 3):
  get_cache_metrics() polls GET /v1/metrics (Prometheus text) and returns
  prefix_cache_hit_rate and gpu_utilization as floats in [0, 1].
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from . import _http_url_from_scheme
from .openai import OpenAIBackend
from ..spec import SharedContext
from ..utils import (
    INTERNAL_HTTP_TIMEOUT,
    NO_API_KEY,
    PREFIX_WARM_TIMEOUT,
    parse_prometheus_metrics,
    prefix_hash,
)

logger = logging.getLogger(__name__)


class VLLMBackend(OpenAIBackend):

    @classmethod
    def from_url(cls, url: str) -> "VLLMBackend":
        return cls(api_key=NO_API_KEY, base_url=_http_url_from_scheme(url, "vllm"))

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        if not shared.prefix:
            return None
        phash = prefix_hash(shared.prefix)
        async with httpx.AsyncClient(timeout=PREFIX_WARM_TIMEOUT) as client:
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
                logger.debug("warm_prefix via chat/completions: prefix hash=%s", phash[:12])
            except Exception as e:
                logger.debug("warm_prefix failed (%s) — continuing without pre-warm", e)

        # Drift 6: pin the prefix block (requires vllm_patch on the server side).
        async with httpx.AsyncClient(timeout=INTERNAL_HTTP_TIMEOUT) as pclient:
            try:
                pin_resp = await pclient.post(
                    f"{self.base_url}/internal/pin_blocks",
                    json={"kv_keys": [phash]},
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                if pin_resp.status_code == 200:
                    logger.debug("Pinned prefix block %s", phash[:12])
            except Exception as e:
                logger.debug("Pin blocks call skipped (%s) — vLLM patch not installed", e)

        return phash

    def backend_capabilities(self) -> dict[str, Any]:
        return {
            "prefix_pinning": True,
            "kvflow": True,
            "diff_kv": True,
            "max_safe_concurrent": 64,
        }

    async def _scrape_vllm_metrics(self) -> dict[str, float]:
        """Shared Prometheus scraper for all vLLM /metrics calls."""
        async with httpx.AsyncClient(timeout=INTERNAL_HTTP_TIMEOUT) as client:
            response = await client.get(f"{self.base_url}/metrics")
            if response.status_code != 200:
                return {}
            return parse_prometheus_metrics(response.text, prefix="vllm:")

    async def get_cache_metrics(self) -> dict[str, float]:
        """Poll vLLM Prometheus metrics for prefix cache hit rate and GPU util."""
        try:
            raw = await self._scrape_vllm_metrics()
            metrics: dict[str, float] = {}
            for name, value in raw.items():
                if ("prefix_cache_queries_total" in name or
                        "prefix_cache_hits_total" in name) and "external" not in name:
                    if "queries" in name:
                        metrics["_q"] = value
                    else:
                        metrics["_h"] = value
                elif "prefix_cache_hit_rate" in name:
                    metrics["prefix_cache_hit_rate"] = value
                elif "gpu_cache_usage" in name:
                    metrics["gpu_utilization"] = value

            if "prefix_cache_hit_rate" not in metrics:
                q = metrics.pop("_q", 0)
                h = metrics.pop("_h", 0)
                if q > 0:
                    metrics["prefix_cache_hit_rate"] = h / q
                    metrics["prefix_cache_hit_tokens"] = h
                    metrics["prefix_cache_query_tokens"] = q
            return metrics
        except Exception as e:
            logger.debug("get_cache_metrics failed: %s", e)
            return {}

    async def get_queue_metrics(self) -> dict[str, Any]:
        """Return vLLM request queue depth from the Prometheus /metrics endpoint."""
        try:
            raw = await self._scrape_vllm_metrics()
            result: dict[str, Any] = {}
            for name, val in raw.items():
                if "num_requests_waiting" in name:
                    result["requests_waiting"] = int(val)
                elif "num_requests_running" in name:
                    result["requests_running"] = int(val)
            return result
        except Exception as e:
            logger.debug("get_queue_metrics failed: %s", e)
            return {}

    async def send_prefetch_hints(self, hints: list[Any]) -> None:
        """Send KVFlow prefetch hints to vLLM /internal/prefetch.

        Requires vllm_patch/prefetch_route.py installed on the vLLM server.
        HARDWARE BLOCKER: silently no-ops if endpoint is unavailable.
        """
        if not hints:
            return
        payload = [h.to_dict() if hasattr(h, "to_dict") else h for h in hints]
        try:
            async with httpx.AsyncClient(timeout=INTERNAL_HTTP_TIMEOUT) as client:
                response = await client.post(
                    f"{self.base_url}/internal/prefetch",
                    json={"hints": payload},
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                if response.status_code != 200:
                    logger.debug("Prefetch endpoint returned %d", response.status_code)
        except Exception as e:
            logger.debug("send_prefetch_hints skipped (%s)", e)
