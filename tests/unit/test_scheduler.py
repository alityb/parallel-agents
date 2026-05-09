from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from batch_agent.backends import BackendAdapter, BackendResponse, ParsedToolCall, StreamingToolCall
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext
from batch_agent.tools import Tool


class Payload(BaseModel):
    value: str


class FakeBackend(BackendAdapter):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None) -> BackendResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return BackendResponse(content=f'{{"value": "{job.index}"}}')


def test_scheduler_returns_results_in_input_order() -> None:
    spec = BatchSpec(task="Do {x}", inputs=[{"x": "a"}, {"x": "b"}], output_schema=Payload, max_concurrent=1)
    plan = TaskCompiler().compile(spec)
    backend = FakeBackend()

    results = asyncio.run(WaveScheduler(plan, backend).run())

    assert [result.output.value for result in results] == ["0", "1"]
    assert backend.max_active == 1


def test_scheduler_returns_failures_as_data_for_oversized_job() -> None:
    spec = BatchSpec(task="Do {x}", inputs=[{"x": "x" * 100}], model_max_context=5, min_response_tokens=1)
    plan = TaskCompiler().compile(spec)

    results = asyncio.run(WaveScheduler(plan, FakeBackend()).run())

    assert results[0].ok is False
    assert results[0].error is not None
    assert results[0].error.type == "OVERSIZED"


def test_streaming_tool_dispatch_starts_tool_before_generate_returns() -> None:
    events: dict[str, float] = {}

    @Tool.define(name="stream_timing_tool", cacheable=False)
    async def stream_timing_tool() -> str:
        events["tool_start"] = time.monotonic()
        await asyncio.sleep(0.01)
        return "ok"

    class StreamingBackend(BackendAdapter):
        async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None) -> BackendResponse:
            await asyncio.sleep(0.05)
            if messages and any(m.role == "tool_result" for m in messages):
                return BackendResponse(content='{"value": "done"}', stop_reason="end_turn")
            events["generate_return"] = time.monotonic()
            tc = ParsedToolCall(id="call_1", name="stream_timing_tool", args={})
            return BackendResponse(content="", tool_calls=[tc], stop_reason="tool_use")

        async def generate_streaming(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None, tool_queue=None, metadata=None) -> BackendResponse:
            if messages and any(m.role == "tool_result" for m in messages):
                return await self.generate(shared=shared, job=job, messages=messages, model=model, tools=tools, timeout=timeout)
            tc = ParsedToolCall(id="call_1", name="stream_timing_tool", args={})
            await tool_queue.put(StreamingToolCall(tool_call=tc))
            await asyncio.sleep(0.05)
            events["generate_return"] = time.monotonic()
            await tool_queue.put(StreamingToolCall(is_final=True))
            return BackendResponse(content="", tool_calls=[tc], stop_reason="tool_use")

    spec = BatchSpec(
        task="Do {x}",
        inputs=[{"x": "a"}],
        output_schema=Payload,
        max_turns=2,
        tools=[Tool.registry["stream_timing_tool"]],
        streaming_tool_dispatch=True,
    )

    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), StreamingBackend()).run())

    assert results[0].ok
    assert events["tool_start"] < events["generate_return"]


def test_streaming_tool_dispatch_disabled_waits_for_generate_return() -> None:
    events: dict[str, float] = {}

    @Tool.define(name="nonstream_timing_tool", cacheable=False)
    async def nonstream_timing_tool() -> str:
        events["tool_start"] = time.monotonic()
        await asyncio.sleep(0.01)
        return "ok"

    class NonStreamingBackend(BackendAdapter):
        async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None) -> BackendResponse:
            await asyncio.sleep(0.05)
            if messages and any(m.role == "tool_result" for m in messages):
                return BackendResponse(content='{"value": "done"}', stop_reason="end_turn")
            events["generate_return"] = time.monotonic()
            tc = ParsedToolCall(id="call_1", name="nonstream_timing_tool", args={})
            return BackendResponse(content="", tool_calls=[tc], stop_reason="tool_use")

    spec = BatchSpec(
        task="Do {x}",
        inputs=[{"x": "a"}],
        output_schema=Payload,
        max_turns=2,
        tools=[Tool.registry["nonstream_timing_tool"]],
        streaming_tool_dispatch=False,
    )

    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), NonStreamingBackend()).run())

    assert results[0].ok
    assert events["tool_start"] > events["generate_return"]


def test_scheduler_preserves_tool_calls_without_backend_raw_blocks() -> None:
    seen_messages: list[list[Message]] = []

    @Tool.define(name="rawless_tool", cacheable=False)
    async def rawless_tool() -> str:
        return "ok"

    class RawlessToolBackend(BackendAdapter):
        async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None) -> BackendResponse:
            seen_messages.append(list(messages or []))
            if messages and any(m.role == "tool_result" for m in messages):
                return BackendResponse(content='{"value": "done"}', raw={"usage": {}}, stop_reason="end_turn")
            tc = ParsedToolCall(id="call_1", name="rawless_tool", args={})
            return BackendResponse(content="", raw={"usage": {}}, tool_calls=[tc], stop_reason="tool_use")

    spec = BatchSpec(
        task="Do {x}",
        inputs=[{"x": "a"}],
        output_schema=Payload,
        max_turns=2,
        tools=[Tool.registry["rawless_tool"]],
        streaming_tool_dispatch=False,
    )

    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), RawlessToolBackend()).run())

    assert results[0].ok
    second_turn_messages = seen_messages[1]
    assistant_raw = [m for m in second_turn_messages if m.role == "assistant_raw"]
    assert assistant_raw
    assert "rawless_tool" in assistant_raw[0].content
