from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pydantic import BaseModel

from batch_agent.backends import BackendAdapter, BackendResponse, ParsedToolCall
from batch_agent.compiler import TaskCompiler
from batch_agent.distributed import DistributedWaveScheduler
from batch_agent.kvflow import KVFlowAdvisor
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext, ToolCall
from batch_agent.state import AgentStatus, InMemoryStateStore, RedisStreamsStateStore
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool
from batch_agent.backends.vllm_patch.diff_cache_engine import AgentKVSnapshot, DiffCacheEngine
from tests.integration.test_distributed_scheduler import MockRedis


class StructuredOutput(BaseModel):
    value: str


class MultiTurnBackend(BackendAdapter):
    def __init__(self, events: dict[str, float]) -> None:
        self.events = events

    async def generate(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None = None,
        model: str,
        tools=None,
        timeout: float | None = None,
    ) -> BackendResponse:
        turn = 1 + sum(1 for message in messages or [] if message.role == "tool_result")
        self.events[f"generate_job_{job.index}_turn_{turn}"] = time.monotonic()
        await asyncio.sleep(0.005)
        if messages and any(message.role == "tool_result" for message in messages):
            return BackendResponse(content=json.dumps({"value": f"done-{job.index}"}), stop_reason="end_turn")
        tool_call = ParsedToolCall(
            id=f"call-{job.index}",
            name="optimization_contract_slow_tool",
            args={"key": job.index},
        )
        return BackendResponse(content="", tool_calls=[tool_call], stop_reason="tool_use")


def test_multi_turn_structured_output_and_tool_wait_slot_release() -> None:
    events: dict[str, float] = {}

    @Tool.define(name="optimization_contract_slow_tool", cacheable=False)
    async def slow_tool(key: int) -> str:
        events[f"tool_start_{key}"] = time.monotonic()
        await asyncio.sleep(0.06)
        events[f"tool_end_{key}"] = time.monotonic()
        return f"tool-{key}"

    spec = BatchSpec(
        task="Process {item}",
        inputs=[{"item": "a"}, {"item": "b"}],
        output_schema=StructuredOutput,
        tools=[Tool.registry["optimization_contract_slow_tool"]],
        max_concurrent=1,
        max_turns=2,
        streaming_tool_dispatch=False,
    )

    results = asyncio.run(
        WaveScheduler(TaskCompiler().compile(spec), MultiTurnBackend(events)).run()
    )

    assert [result.output.value for result in results] == ["done-0", "done-1"]
    assert events["generate_job_1_turn_1"] < events["tool_end_0"]


def test_tool_pool_coalesces_identical_calls() -> None:
    calls = 0

    @Tool.define(name="optimization_contract_shared_read", cacheable=True)
    async def shared_read(path: str) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return f"contents:{path}"

    pool = ToolPool()

    async def run() -> list[Any]:
        return await asyncio.gather(
            *[pool.call("optimization_contract_shared_read", {"path": "README.md"}) for _ in range(25)]
        )

    results = asyncio.run(run())

    assert set(results) == {"contents:README.md"}
    assert calls == 1


def test_warm_prefix_called_once_for_shared_prefix() -> None:
    class WarmBackend(BackendAdapter):
        def __init__(self) -> None:
            self.warmed: list[str] = []

        async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
            self.warmed.append(shared.prefix)
            return "kv-shared"

        async def generate(
            self,
            *,
            shared: SharedContext,
            job: AgentJob,
            messages: list[Message] | None = None,
            model: str,
            tools=None,
            timeout: float | None = None,
        ) -> BackendResponse:
            return BackendResponse(content=json.dumps({"value": str(job.index)}), stop_reason="end_turn")

    backend = WarmBackend()
    spec = BatchSpec(
        task="Shared instruction. Review {file}",
        inputs=[{"file": f"f{i}.py"} for i in range(5)],
        output_schema=StructuredOutput,
        max_concurrent=5,
    )

    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), backend).run())

    assert len(results) == 5
    assert len(backend.warmed) == 1
    assert backend.warmed[0].startswith("Shared instruction")


def test_streaming_result_delivery_yields_before_slowest_agent_finishes() -> None:
    class VariableBackend(BackendAdapter):
        async def generate(
            self,
            *,
            shared: SharedContext,
            job: AgentJob,
            messages: list[Message] | None = None,
            model: str,
            tools=None,
            timeout: float | None = None,
        ) -> BackendResponse:
            if job.index == 1:
                await asyncio.sleep(0.08)
            return BackendResponse(content=json.dumps({"value": str(job.index)}), stop_reason="end_turn")

    spec = BatchSpec(
        task="Do {x}",
        inputs=[{"x": "fast"}, {"x": "slow"}],
        output_schema=StructuredOutput,
        max_concurrent=2,
    )

    async def run() -> tuple[int, float, float]:
        scheduler = WaveScheduler(TaskCompiler().compile(spec), VariableBackend())
        started = time.monotonic()
        async for result in scheduler.stream():
            first_elapsed = time.monotonic() - started
            return result.index, first_elapsed, time.monotonic() - started
        raise AssertionError("no result yielded")

    first_index, first_elapsed, _ = asyncio.run(run())

    assert first_index == 0
    assert first_elapsed < 0.06


def test_kvflow_mock_emits_hints_only_not_verified_prefetch() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.hints = []

        async def send_prefetch_hints(self, hints):
            self.hints.extend(hints)

    store = InMemoryStateStore()
    pool = ToolPool()
    backend = RecordingBackend()
    state = store.create("job-kv")
    state.status = AgentStatus.TOOL_WAIT
    state.kv_key = "kv-key"
    state.tool_calls_pending = [ToolCall(name="search", args={})]
    pool._record_latency("search", 0.4)

    hints = asyncio.run(
        KVFlowAdvisor(state_store=store, tool_pool=pool, backend=backend).emit_once()
    )

    assert len(hints) == 1
    assert backend.hints == hints
    assert hints[0].kv_key == "kv-key"


def test_tokendance_mock_diff_kv_reduces_stored_blocks() -> None:
    shared = tuple(range(128))
    snapshots = [
        AgentKVSnapshot(job_id=f"job-{i}", tokens=shared + tuple(range(1000 + i * 16, 1016 + i * 16)), turn=2)
        for i in range(20)
    ]
    engine = DiffCacheEngine(block_size=16)

    asyncio.run(engine.all_gather(snapshots, completion_fraction=1.0))
    stats = engine.stats(snapshots)

    assert stats.full_blocks == 180
    assert stats.stored_unique_blocks == 28
    assert stats.compression_ratio > 6.0


def test_distributed_mock_redis_orchestration_completes_shards() -> None:
    class JSONBackend(BackendAdapter):
        async def generate(
            self,
            *,
            shared: SharedContext,
            job: AgentJob,
            messages: list[Message] | None = None,
            model: str,
            tools=None,
            timeout: float | None = None,
        ) -> BackendResponse:
            return BackendResponse(content=json.dumps({"value": str(job.index)}), stop_reason="end_turn")

    redis = MockRedis()
    spec = BatchSpec(
        task="Do {x}",
        inputs=[{"x": i} for i in range(20)],
        output_schema=StructuredOutput,
        model="mock",
        backend="mock://",
        max_concurrent=4,
    )
    nodes = ["node-a", "node-b"]

    async def run() -> None:
        await asyncio.gather(
            DistributedWaveScheduler(
                spec=spec,
                backend=JSONBackend(),
                redis_client=redis,
                node_id="node-a",
                nodes=nodes,
            ).run(),
            DistributedWaveScheduler(
                spec=spec,
                backend=JSONBackend(),
                redis_client=redis,
                node_id="node-b",
                nodes=nodes,
            ).run(),
        )

    asyncio.run(run())

    store = RedisStreamsStateStore(redis, node_id="observer")
    completed = [
        store.load(f"job-{i}")
        for i in range(20)
        if store.load(f"job-{i}") and store.load(f"job-{i}").status == AgentStatus.COMPLETE
    ]
    assert len(completed) == 20
