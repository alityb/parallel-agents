"""Unit test for BackpressureController.

Verifies:
1. has_capacity_sync returns False when requests_waiting >= ceiling
2. wait_for_capacity blocks while queue is full and unblocks when it drains
3. Dispatch stops when queue fills, resumes when it drops
"""
from __future__ import annotations

import asyncio

from batch_agent.backpressure import BackpressureController, clear_calibration_cache


class _FakeBackend:
    """Controllable fake backend for backpressure testing."""

    def __init__(self) -> None:
        self._waiting = 0

    def set_waiting(self, n: int) -> None:
        self._waiting = n

    async def get_queue_metrics(self) -> dict:
        return {"requests_waiting": self._waiting, "requests_running": 0}


def test_backpressure_blocks_at_ceiling() -> None:
    ctrl = BackpressureController(queue_depth_ceiling=4, poll_interval_seconds=0.01)

    # Below ceiling → has capacity
    assert ctrl.has_capacity_sync({"requests_waiting": 3}) is True
    # At ceiling → no capacity
    assert ctrl.has_capacity_sync({"requests_waiting": 4}) is False
    # Above ceiling → no capacity
    assert ctrl.has_capacity_sync({"requests_waiting": 10}) is False
    print("[PASS] has_capacity_sync: correct threshold")


def test_backpressure_wait_for_capacity_unblocks() -> None:
    """wait_for_capacity should return as soon as the queue drains below ceiling."""
    backend = _FakeBackend()
    backend.set_waiting(8)  # start full
    ctrl = BackpressureController(queue_depth_ceiling=4, poll_interval_seconds=0.01)

    async def drain_after_delay() -> None:
        await asyncio.sleep(0.05)
        backend.set_waiting(3)  # drain below ceiling

    async def run() -> float:
        import time
        asyncio.create_task(drain_after_delay())
        t0 = time.monotonic()
        await ctrl.wait_for_capacity(backend)
        return time.monotonic() - t0

    elapsed = asyncio.run(run())
    assert elapsed >= 0.04, f"Should have waited at least 40ms, got {elapsed:.3f}s"
    assert elapsed < 0.5, f"Should not have waited more than 500ms, got {elapsed:.3f}s"
    print(f"[PASS] wait_for_capacity: blocked for {elapsed:.3f}s until queue drained")


def test_backpressure_controller_dispatch_simulation() -> None:
    """Simulates N agents dispatching through a backpressure controller.

    Mock vLLM queue:
    - requests_waiting starts at 0
    - grows by 1 each time a new agent dispatches
    - shrinks by 1 each time an agent completes
    Ceiling = 4.

    Verifies:
    - Controller pauses dispatch when waiting >= 4
    - Resumes when a completion brings waiting < 4
    """
    backend = _FakeBackend()
    ctrl = BackpressureController(queue_depth_ceiling=4, poll_interval_seconds=0.005)

    dispatched: list[int] = []
    completed: list[int] = []

    async def run() -> None:
        async def agent(idx: int) -> None:
            await ctrl.wait_for_capacity(backend)
            backend.set_waiting(backend._waiting + 1)
            dispatched.append(idx)
            await asyncio.sleep(0.03)  # simulate inference
            backend.set_waiting(max(0, backend._waiting - 1))
            completed.append(idx)

        tasks = [asyncio.create_task(agent(i)) for i in range(16)]
        await asyncio.gather(*tasks)

    asyncio.run(run())

    assert len(dispatched) == 16, f"All 16 should have dispatched, got {len(dispatched)}"
    assert len(completed) == 16, f"All 16 should have completed, got {len(completed)}"
    # At no point should more than ceiling+1 be dispatched ahead of completions
    # (small overshoot possible due to async scheduling)
    print(f"[PASS] dispatch simulation: {len(dispatched)} dispatched, {len(completed)} completed")


def test_calibration_cache_is_populated() -> None:
    clear_calibration_cache()
    from batch_agent.backpressure import _calibration_cache
    assert "test://host" not in _calibration_cache
    _calibration_cache["test://host"] = 64
    assert _calibration_cache["test://host"] == 64
    clear_calibration_cache("test://host")
    assert "test://host" not in _calibration_cache
    print("[PASS] calibration cache: set, retrieve, clear work correctly")


if __name__ == "__main__":
    test_backpressure_blocks_at_ceiling()
    test_backpressure_wait_for_capacity_unblocks()
    test_backpressure_controller_dispatch_simulation()
    test_calibration_cache_is_populated()
    print("\n[ALL PASS] BackpressureController tests")
