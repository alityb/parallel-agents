"""BackpressureController and auto-calibration for the Wave Scheduler.

Replaces the fixed dispatch-token gate (which serialised agents into waves)
with a continuous flow-controller that dispatches jobs as fast as the backend
can absorb them.

Two mechanisms:

1. BackpressureController
   Dispatches a new agent whenever requests_waiting < queue_depth_ceiling.
   For vLLM this is polled from /metrics.  For mock/API backends get_queue_metrics()
   returns {} which means "always has capacity" — the controller is effectively a
   no-op and max_inflight becomes the only limiting factor.

2. calibrate_max_inflight
   Runs a 5-second ramp (8 → 16 → 32 → 64 → 128) before the first wave,
   measures throughput at each level, and returns the throughput-maximising value.
   Result is cached per backend URL so it is not re-run on every BatchAgent.run().
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backends import BackendAdapter
    from .spec import AgentJob, SharedContext

logger = logging.getLogger(__name__)

# Cache: backend_url → calibrated max_inflight
_calibration_cache: dict[str, int] = {}


class BackpressureController:
    """Rate-limits dispatch based on backend queue depth.

    Dispatching is paused whenever:
        requests_waiting >= queue_depth_ceiling

    For backends that do not expose queue metrics (Anthropic, OpenAI, Bedrock)
    ``get_queue_metrics()`` returns {} and this controller is a transparent
    pass-through — max_inflight is the only throttle.

    For vLLM, ``requests_waiting`` comes from the Prometheus /metrics endpoint.
    """

    def __init__(
        self,
        queue_depth_ceiling: int = 8,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.queue_depth_ceiling = queue_depth_ceiling
        self.poll_interval = poll_interval_seconds
        self._last_metrics: dict[str, Any] = {}
        self._total_waits: int = 0

    async def wait_for_capacity(self, backend: "BackendAdapter") -> None:
        """Block until the backend queue has room for one more request."""
        while True:
            metrics = await backend.get_queue_metrics()
            self._last_metrics = metrics
            waiting = metrics.get("requests_waiting", 0)
            if waiting < self.queue_depth_ceiling:
                return
            self._total_waits += 1
            await asyncio.sleep(self.poll_interval)

    def has_capacity_sync(self, last_metrics: dict[str, Any]) -> bool:
        """Non-blocking check against cached metrics (used in unit tests)."""
        return last_metrics.get("requests_waiting", 0) < self.queue_depth_ceiling


async def calibrate_max_inflight(
    backend: "BackendAdapter",
    shared: "SharedContext",
    model: str,
    backend_url: str = "",
    levels: list[int] | None = None,
    duration_per_level: float = 1.0,
    min_requests: int = 10,
) -> int:
    """Ramp concurrency from low to high; return the throughput-peak value.

    Caches the result per backend_url so repeated BatchAgent.run() calls on the
    same endpoint skip calibration.

    Args:
        backend: The BackendAdapter to calibrate against.
        shared: SharedContext (system prompt) for the test requests.
        model: Model identifier.
        backend_url: Cache key (e.g. "vllm://localhost:8000").
        levels: Concurrency levels to test.  Defaults to [8, 16, 32, 64, 128].
        duration_per_level: Seconds to test each level.
        min_requests: Minimum requests to complete before trusting a measurement.

    Returns:
        The concurrency level that maximised throughput.
    """
    if backend_url and backend_url in _calibration_cache:
        cached = _calibration_cache[backend_url]
        logger.info("calibrate_max_inflight: using cached value %d for %s", cached, backend_url)
        return cached

    if levels is None:
        levels = [8, 16, 32, 64, 128]

    best_concurrency = levels[0]
    best_throughput = 0.0
    logger.info("calibrate_max_inflight: starting ramp over %s", levels)

    # Import here to avoid circular at module load
    from .spec import AgentJob

    async def _one_request(sem: asyncio.Semaphore, idx: int) -> float:
        t0 = time.monotonic()
        async with sem:
            await backend.generate(
                shared=shared,
                job=AgentJob(
                    job_id=f"calib-{idx}",
                    index=idx,
                    input_data={},
                    prompt="ping",
                    estimated_prompt_tokens=2,
                ),
                model=model,
                timeout=10.0,
            )
        return time.monotonic() - t0

    for level in levels:
        sem = asyncio.Semaphore(level)
        completed = 0
        t_start = time.monotonic()

        async def _worker(idx: int) -> None:
            nonlocal completed
            while time.monotonic() - t_start < duration_per_level:
                await _one_request(sem, idx)
                completed += 1

        workers = [asyncio.create_task(_worker(i)) for i in range(level)]
        await asyncio.sleep(duration_per_level + 0.2)
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        elapsed = min(time.monotonic() - t_start, duration_per_level + 0.2)
        throughput = completed / elapsed if elapsed > 0 else 0.0
        logger.info(
            "calibrate level=%d: completed=%d in %.2fs → %.1f req/s",
            level, completed, elapsed, throughput,
        )

        if completed < min_requests:
            # Too few requests at this level — backend may be slow; stop ramping
            logger.info("calibrate: stopping ramp (insufficient requests at level %d)", level)
            break

        if throughput > best_throughput * 1.05:   # 5% improvement threshold
            best_throughput = throughput
            best_concurrency = level
        elif throughput < best_throughput * 0.90:  # 10% drop → we hit the wall
            logger.info(
                "calibrate: throughput dropped %.1f → %.1f at level %d; stopping",
                best_throughput, throughput, level,
            )
            break

    logger.info(
        "calibrate_max_inflight: chose %d (%.1f req/s) for %s",
        best_concurrency, best_throughput, backend_url or "backend",
    )
    if backend_url:
        _calibration_cache[backend_url] = best_concurrency
    return best_concurrency


def clear_calibration_cache(backend_url: str | None = None) -> None:
    """Clear calibration cache (for testing or when backend config changes)."""
    if backend_url:
        _calibration_cache.pop(backend_url, None)
    else:
        _calibration_cache.clear()
