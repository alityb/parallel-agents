from __future__ import annotations

import asyncio
import json
import time

from pydantic import BaseModel

from batch_agent.backends import BackendAdapter, BackendResponse
from batch_agent.distributed import DistributedWaveScheduler, NodeStopped
from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext
from batch_agent.state import AgentStatus, RedisStreamsStateStore


class MockRedis:
    def __init__(self) -> None:
        self.values = {}
        self.expiry = {}
        self.stream = []

    def _cleanup(self, key: str) -> None:
        expires = self.expiry.get(key)
        if expires is not None and time.time() >= expires:
            self.values.pop(key, None)
            self.expiry.pop(key, None)

    def get(self, key: str):
        self._cleanup(key)
        return self.values.get(key)

    def set(self, key: str, value, nx: bool = False, ex: int | None = None, px: int | None = None):
        self._cleanup(key)
        if nx and key in self.values:
            return False
        self.values[key] = value
        if ex is not None:
            self.expiry[key] = time.time() + ex
        if px is not None:
            self.expiry[key] = time.time() + (px / 1000)
        return True

    def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.expiry.pop(key, None)

    def xadd(self, stream: str, fields: dict) -> None:
        self.stream.append((stream, fields))


class Output(BaseModel):
    value: int


class MockBackend(BackendAdapter):
    async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout=None) -> BackendResponse:
        await asyncio.sleep(0.001)
        body = json.dumps({"value": job.index})
        return BackendResponse(content=body, raw={"content": [{"type": "text", "text": body}]}, stop_reason="end_turn")


def test_distributed_wave_scheduler_failover_completes_after_node_loss() -> None:
    redis = MockRedis()
    spec = BatchSpec(
        task="Process {i}",
        inputs=[{"i": i} for i in range(100)],
        output_schema=Output,
        model="mock",
        backend="mock://",
        distributed=True,
        max_concurrent=10,
    )
    nodes = ["node-a", "node-b"]
    node_a = DistributedWaveScheduler(spec=spec, backend=MockBackend(), redis_client=redis, node_id="node-a", nodes=nodes, lease_ttl_seconds=0.05)
    node_b = DistributedWaveScheduler(spec=spec, backend=MockBackend(), redis_client=redis, node_id="node-b", nodes=nodes, lease_ttl_seconds=0.05)

    async def run() -> None:
        try:
            await node_a.run(stop_after=30)
        except NodeStopped:
            pass
        await node_b.run()
        # Let node-a leases expire, then surviving node picks up all remaining jobs.
        await asyncio.sleep(0.06)
        await node_b.run(failover=True)

    asyncio.run(run())

    store = RedisStreamsStateStore(redis, node_id="observer")
    completed = 0
    for i in range(100):
        state = store.load(f"job-{i}")
        if state and state.status == AgentStatus.COMPLETE:
            completed += 1
    loss_rate = (100 - completed) / 100
    assert completed >= 95
    assert loss_rate <= 0.05
