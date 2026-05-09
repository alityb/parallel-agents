from __future__ import annotations

import asyncio

from batch_agent.kvflow import KVFlowAdvisor
from batch_agent.spec import ToolCall
from batch_agent.state import AgentStatus, InMemoryStateStore
from batch_agent.tools.pool import ToolPool


class RecordingBackend:
    def __init__(self) -> None:
        self.hints = []

    async def send_prefetch_hints(self, hints):
        self.hints.extend(hints)


def test_kvflow_advisor_orders_by_shortest_eta_and_excludes_outside_horizon() -> None:
    store = InMemoryStateStore()
    pool = ToolPool()
    backend = RecordingBackend()

    # Seed latency history. P75 short < medium < long.
    for value in [0.20, 0.25, 0.30, 0.35]:
        pool._record_latency("short_tool", value)
    for value in [0.80, 0.90, 1.00, 1.10]:
        pool._record_latency("medium_tool", value)
    for value in [2.5, 3.0, 3.5, 4.0]:
        pool._record_latency("long_tool", value)

    short = store.create("job-short")
    short.status = AgentStatus.TOOL_WAIT
    short.kv_key = "kv-short"
    short.tool_calls_pending = [ToolCall(name="short_tool", args={})]

    medium = store.create("job-medium")
    medium.status = AgentStatus.TOOL_WAIT
    medium.kv_key = "kv-medium"
    medium.tool_calls_pending = [ToolCall(name="medium_tool", args={})]

    long = store.create("job-long")
    long.status = AgentStatus.TOOL_WAIT
    long.kv_key = "kv-long"
    long.tool_calls_pending = [ToolCall(name="long_tool", args={})]

    advisor = KVFlowAdvisor(
        state_store=store,
        tool_pool=pool,
        backend=backend,
        prefetch_horizon=2.0,
    )

    hints = asyncio.run(advisor.emit_once())

    assert [h.job_id for h in hints] == ["job-short", "job-medium"]
    assert "job-long" not in [h.job_id for h in hints]
    assert hints[0].eta_seconds < hints[1].eta_seconds
    assert short.estimated_next_activation is not None
    assert medium.steps_to_execution is not None
    assert backend.hints == hints
