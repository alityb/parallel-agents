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


def test_streamed_tool_call_is_dispatched_before_final_response() -> None:
    events: dict[str, float] = {}

    @Tool.define(name="integration_stream_tool", cacheable=False)
    async def integration_stream_tool() -> str:
        events["tool_start"] = time.monotonic()
        await asyncio.sleep(0.10)
        events["tool_done"] = time.monotonic()
        return "streamed"

    class MockStreamingBackend(BackendAdapter):
        async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None) -> BackendResponse:
            await asyncio.sleep(0.10)
            if messages and any(m.role == "tool_result" for m in messages):
                return BackendResponse(content='{"value": "done"}', stop_reason="end_turn")
            tc = ParsedToolCall(id="call_stream", name="integration_stream_tool", args={})
            return BackendResponse(content="", tool_calls=[tc], stop_reason="tool_use")

        async def generate_streaming(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None, tool_queue=None, metadata=None) -> BackendResponse:
            if messages and any(m.role == "tool_result" for m in messages):
                return await self.generate(shared=shared, job=job, messages=messages, model=model, tools=tools, timeout=timeout)
            tc = ParsedToolCall(id="call_stream", name="integration_stream_tool", args={})
            await asyncio.sleep(0.02)
            await tool_queue.put(StreamingToolCall(tool_call=tc))
            events["tool_emitted"] = time.monotonic()
            await asyncio.sleep(0.10)
            events["response_done"] = time.monotonic()
            await tool_queue.put(StreamingToolCall(is_final=True))
            return BackendResponse(content="", tool_calls=[tc], stop_reason="tool_use")

    spec = BatchSpec(
        task="Do {x}",
        inputs=[{"x": "a"}],
        output_schema=Payload,
        max_turns=2,
        tools=[Tool.registry["integration_stream_tool"]],
        streaming_tool_dispatch=True,
    )

    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), MockStreamingBackend()).run())

    assert results[0].ok
    assert events["tool_start"] >= events["tool_emitted"]
    assert events["tool_start"] < events["response_done"]
