from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from .backends import BackendAdapter, backend_from_url
from .compiler import TaskCompiler
from .scheduler import WaveScheduler
from .spec import AgentResult, BatchSpec
from .tools import Tool


class BatchAgent:
    @classmethod
    async def run(cls, **kwargs: Any) -> list[AgentResult]:
        spec = BatchSpec(**kwargs)
        scheduler = cls._scheduler(spec)
        return await scheduler.run()

    @classmethod
    async def stream(cls, **kwargs: Any) -> AsyncIterator[AgentResult]:
        spec = BatchSpec(**kwargs)
        scheduler = cls._scheduler(spec)
        async for result in scheduler.stream():
            yield result

    @classmethod
    def _scheduler(cls, spec: BatchSpec, backend: BackendAdapter | None = None) -> WaveScheduler:
        plan = TaskCompiler().compile(spec)
        return WaveScheduler(plan, backend or backend_from_url(spec.backend))


__all__ = ["AgentResult", "BatchAgent", "BatchSpec", "Tool"]
