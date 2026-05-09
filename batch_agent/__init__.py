from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from .backends import BackendAdapter, backend_from_url
from .compiler import TaskCompiler
from .repair import parse_and_validate_output
from .scheduler import WaveScheduler
from .spec import AgentJob, AgentResult, BatchSpec, ExecutionPlan, Message, SharedContext
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
    async def run_with_reduce(cls, **kwargs: Any) -> tuple[list[AgentResult], Any]:
        """Run batch agents, then run a reduce agent that sees all results.

        Returns (individual_results, reduce_output).
        """
        reduce_prompt = kwargs.pop("reduce", None)
        reduce_schema = kwargs.pop("reduce_schema", None)
        if not reduce_prompt:
            raise ValueError("run_with_reduce requires a 'reduce' prompt")

        spec = BatchSpec(**kwargs)
        backend = backend_from_url(spec.backend)
        scheduler = cls._scheduler(spec, backend)
        results = await scheduler.run()

        # Build reduce input: all successful outputs
        successful_outputs = []
        for r in results:
            if r.ok:
                output = r.output
                if hasattr(output, "model_dump"):
                    output = output.model_dump()
                elif hasattr(output, "dict"):
                    output = output.dict()
                successful_outputs.append(output)

        # Format the reduce prompt
        reduce_text = reduce_prompt.format(n=len(successful_outputs))
        results_json = json.dumps(successful_outputs, default=str, ensure_ascii=False)
        full_reduce_prompt = f"{reduce_text}\n\nResults:\n{results_json}"

        # Build shared context for reduce agent
        reduce_shared = SharedContext(prefix=spec.system_prompt or "")
        if reduce_schema:
            from .compiler import SCHEMA_INSTRUCTION
            schema_dict = None
            if hasattr(reduce_schema, "model_json_schema"):
                schema_dict = reduce_schema.model_json_schema()
            elif hasattr(reduce_schema, "schema"):
                schema_dict = reduce_schema.schema()
            elif isinstance(reduce_schema, dict):
                schema_dict = reduce_schema
            if schema_dict:
                reduce_shared = SharedContext(
                    prefix=(spec.system_prompt or "") + f"\n\n{SCHEMA_INSTRUCTION}\nSchema:\n{json.dumps(schema_dict)}",
                    schema=schema_dict,
                )

        # Run reduce agent
        reduce_job = AgentJob(
            job_id="reduce-0",
            index=0,
            input_data={},
            prompt=full_reduce_prompt,
            estimated_prompt_tokens=len(full_reduce_prompt) // 4,
            max_turns=spec.max_turns,
        )

        reduce_response = await backend.generate(
            shared=reduce_shared,
            job=reduce_job,
            model=spec.model,
            timeout=spec.timeout_per_agent,
        )

        reduce_output = parse_and_validate_output(reduce_response.content, reduce_schema)
        return results, reduce_output

    @classmethod
    def _scheduler(cls, spec: BatchSpec, backend: BackendAdapter | None = None) -> WaveScheduler:
        plan = TaskCompiler().compile(spec)
        return WaveScheduler(plan, backend or backend_from_url(spec.backend))


__all__ = ["AgentResult", "BatchAgent", "BatchSpec", "Tool"]
