"""PrioritySemaphore — asyncio semaphore where waiters are served by priority.

Lower priority value = served sooner (min-heap). Used by the Wave Scheduler so
that agents with fewer turns remaining (nearer completion) jump ahead of fresh agents
when competing for an inference slot.
"""
from __future__ import annotations

import asyncio
import heapq
import logging

logger = logging.getLogger(__name__)


class PrioritySemaphore:
    """Semaphore where acquirers are served by priority (lower value = served first).

    Thread-safety: asyncio-safe (single-threaded event loop assumed).
    Capacity can be adjusted dynamically via set_capacity() for adaptive concurrency.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._active = 0  # slots currently held
        self._waiters: list[tuple[float, int, asyncio.Future]] = []  # heap
        self._counter = 0  # tiebreaker for equal priorities

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    async def acquire(self, priority: float = 0.0) -> None:
        """Acquire a slot. Lower priority = served before higher-priority waiters."""
        if self._active < self._capacity:
            self._active += 1
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, self._counter, fut))
        self._counter += 1
        try:
            await fut
        except asyncio.CancelledError:
            # Remove from waiters if cancelled
            self._waiters = [(p, c, f) for p, c, f in self._waiters if f is not fut]
            heapq.heapify(self._waiters)
            raise
        # _active already incremented by release() before waking us

    def release(self) -> None:
        """Release a slot. Wakes the highest-priority waiter, if any."""
        if self._waiters:
            # Find the first non-done future
            while self._waiters:
                priority, counter, fut = heapq.heappop(self._waiters)
                if not fut.done():
                    fut.set_result(None)
                    # _active stays the same: released slot immediately given to waiter
                    return
            # All waiters were cancelled/done
            self._active -= 1
        else:
            self._active -= 1

    def set_capacity(self, new_capacity: int) -> None:
        """Adjust capacity. Increases take effect immediately; decreases are gradual."""
        new_capacity = max(1, new_capacity)
        if new_capacity > self._capacity:
            delta = new_capacity - self._capacity
            self._capacity = new_capacity
            # Wake up to delta waiters immediately
            for _ in range(delta):
                if self._waiters:
                    while self._waiters:
                        _, _, fut = heapq.heappop(self._waiters)
                        if not fut.done():
                            self._active += 1
                            fut.set_result(None)
                            break
                # else: extra capacity will be picked up by next acquire()
        else:
            self._capacity = new_capacity
            # Existing holders keep their slots; capacity shrinks as they release
