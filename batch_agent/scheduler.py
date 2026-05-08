from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator

from .backends import BackendAdapter
from .repair import parse_and_validate_output
from .spec import AgentError, AgentJob, AgentResult, ExecutionPlan
from .state import AgentStatus, InMemoryStateStore


class WaveScheduler:
    def __init__(self, plan: ExecutionPlan, backend: BackendAdapter) -> None:
        self.plan = plan
        self.backend = backend
        self.states = InMemoryStateStore()
        self._semaphore = asyncio.Semaphore(plan.spec.max_concurrent)

    async def run(self) -> list[AgentResult]:
        results: list[AgentResult | None] = [None] * len(self.plan.jobs)
        async for result in self.stream():
            results[result.index] = result
        return [result for result in results if result is not None]

    async def stream(self) -> AsyncIterator[AgentResult]:
        await self.backend.warm_prefix(self.plan.shared, self.plan.spec.model)
        queue: asyncio.Queue[AgentResult | None] = asyncio.Queue()
        tasks = [asyncio.create_task(self._execute_to_queue(job, queue)) for job in self.plan.jobs]
        waiter = asyncio.create_task(self._finish_when_done(tasks, queue))

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await waiter

    async def _finish_when_done(self, tasks: list[asyncio.Task[None]], queue: asyncio.Queue[AgentResult | None]) -> None:
        await asyncio.gather(*tasks)
        await queue.put(None)

    async def _execute_to_queue(self, job: AgentJob, queue: asyncio.Queue[AgentResult | None]) -> None:
        result = await self._execute(job)
        callback = self.plan.spec.on_result
        if callback:
            maybe_awaitable = callback(result)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        await queue.put(result)

    async def _execute(self, job: AgentJob) -> AgentResult:
        state = self.states.create(job.job_id)
        state.set_status(AgentStatus.PREFLIGHT)

        if job.oversized:
            error = AgentError("OVERSIZED", "prompt exceeds configured model context budget", retryable=False)
            state.error = error
            state.set_status(AgentStatus.FAILED)
            return AgentResult(job_id=job.job_id, index=job.index, output=None, error=error, attempts=0)

        attempts = self.plan.spec.max_retries + 1
        for attempt in range(1, attempts + 1):
            state.retry_count = attempt - 1
            try:
                state.set_status(AgentStatus.RUNNING)
                async with self._semaphore:
                    response = await self.backend.generate(
                        shared=self.plan.shared,
                        job=job,
                        model=self.plan.spec.model,
                        timeout=self.plan.spec.timeout_per_agent,
                    )
                state.turn += 1
                output = parse_and_validate_output(response.content, self.plan.spec.output_schema)
                state.output = output
                state.set_status(AgentStatus.COMPLETE)
                return AgentResult(job_id=job.job_id, index=job.index, output=output, attempts=attempt)
            except Exception as exc:
                retryable = attempt < attempts
                if retryable:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
                    continue
                error = AgentError(type=exc.__class__.__name__, message=str(exc), retryable=False)
                state.error = error
                state.set_status(AgentStatus.FAILED)
                return AgentResult(job_id=job.job_id, index=job.index, output=None, error=error, attempts=attempt)

        raise AssertionError("unreachable")
