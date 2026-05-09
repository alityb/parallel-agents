"""KVFlow Advisor — scheduler-to-inference prefetch hints.

The advisor scans agent state every 500ms, finds agents in TOOL_WAIT, estimates
when each will need GPU again using ToolPool P75 latency, and emits prefetch hints
to the backend before the agent reactivates.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .state import AgentStatus, InMemoryStateStore
from .tools.pool import ToolPool


@dataclass(frozen=True)
class PrefetchHint:
    job_id: str
    kv_key: str
    priority: float
    eta_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrefetchBackend(Protocol):
    async def send_prefetch_hints(self, hints: list[PrefetchHint]) -> None: ...


class KVFlowAdvisor:
    def __init__(
        self,
        *,
        state_store: InMemoryStateStore,
        tool_pool: ToolPool,
        backend: PrefetchBackend,
        prefetch_horizon: float = 2.0,
        interval_seconds: float = 0.5,
    ) -> None:
        self.state_store = state_store
        self.tool_pool = tool_pool
        self.backend = backend
        self.prefetch_horizon = prefetch_horizon
        self.interval_seconds = interval_seconds
        self._running = False
        self.last_hints: list[PrefetchHint] = []

    async def run(self) -> None:
        self._running = True
        while self._running:
            await self.emit_once()
            await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False

    async def emit_once(self) -> list[PrefetchHint]:
        hints = self.compute_hints()
        self.last_hints = hints
        if hints:
            await self.backend.send_prefetch_hints(hints)
        return hints

    def compute_hints(self) -> list[PrefetchHint]:
        now = time.time()
        hints: list[PrefetchHint] = []

        for state in self.state_store.all_in_status(AgentStatus.TOOL_WAIT):
            if not state.kv_key:
                continue
            eta = self._estimate_steps_to_execution(state)
            state.steps_to_execution = eta
            state.estimated_next_activation = now + eta
            if eta >= self.prefetch_horizon:
                continue
            # Shorter ETA means higher priority; keep numeric priority monotonic.
            priority = 1.0 / max(eta, 0.001)
            hints.append(PrefetchHint(
                job_id=state.job_id,
                kv_key=state.kv_key,
                priority=priority,
                eta_seconds=eta,
            ))

        hints.sort(key=lambda h: h.eta_seconds)
        return hints

    def _estimate_steps_to_execution(self, state: Any) -> float:
        if state.tool_calls_pending:
            latencies = []
            for call in state.tool_calls_pending:
                p75 = self.tool_pool.p75_latency(call.name)
                if p75 is not None:
                    latencies.append(p75)
            if latencies:
                return max(latencies)
        if state.steps_to_execution is not None:
            return state.steps_to_execution
        if state.p75_tool_wait() is not None:
            return state.p75_tool_wait() or 0.0
        return 0.5
