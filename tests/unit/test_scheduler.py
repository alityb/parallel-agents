from __future__ import annotations

import asyncio

from pydantic import BaseModel

from batch_agent.backends import BackendAdapter, BackendResponse
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext


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
