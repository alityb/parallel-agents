from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel

from batch_agent.backends import BackendAdapter, BackendResponse, ParsedToolCall
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool


class Output(BaseModel):
    value: int


class MockPrefetchBackend(BackendAdapter):
    def __init__(self) -> None:
        self.prefetched: set[str] = set()
        self.requests_after_prefetch = 0
        self.total_reactivation_requests = 0

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        return "shared-kv"

    async def send_prefetch_hints(self, hints) -> None:
        for h in hints:
            self.prefetched.add(h.kv_key)

    async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, metadata=None, timeout=None) -> BackendResponse:
        tool_results = sum(1 for m in (messages or []) if m.role == "tool_result")
        if tool_results > 0:
            self.total_reactivation_requests += 1
            if metadata and metadata.get("kv_key") in self.prefetched:
                self.requests_after_prefetch += 1
        if tool_results < 2:
            return BackendResponse(
                content="",
                raw={"content": [{"type": "tool_use", "id": f"tool-{job.index}-{tool_results}", "name": "slow_tool", "input": {}}]},
                tool_calls=[ParsedToolCall(id=f"tool-{job.index}-{tool_results}", name="slow_tool", args={})],
                stop_reason="tool_use",
            )
        body = json.dumps({"value": job.index})
        return BackendResponse(content=body, raw={"content": [{"type": "text", "text": body}]}, stop_reason="end_turn")


@Tool.define(name="slow_tool", cacheable=False)
async def slow_tool() -> str:
    await asyncio.sleep(0.3)
    return "ok"


def test_prefetch_accuracy_with_300ms_tool_wait() -> None:
    backend = MockPrefetchBackend()
    pool = ToolPool()
    # Seed ETA so advisor has P75 data before first TOOL_WAIT.
    for value in [0.25, 0.30, 0.30, 0.35]:
        pool._record_latency("slow_tool", value)

    spec = BatchSpec(
        task="Run {i}",
        inputs=[{"i": i} for i in range(20)],
        tools=[Tool.registry["slow_tool"]],
        output_schema=Output,
        model="mock",
        backend="vllm://mock",
        max_concurrent=8,
        max_turns=3,
        timeout_per_turn=5,
        timeout_per_tool=2,
        kvflow=True,
    )
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), backend, tool_pool=pool).run())
    assert all(r.ok for r in results)
    hit_rate = backend.requests_after_prefetch / backend.total_reactivation_requests
    assert hit_rate >= 0.80
