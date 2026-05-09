from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .backends import BackendAdapter, backend_from_url
from .compiler import SCHEMA_INSTRUCTION, TaskCompiler
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

        The reduce agent uses the same multi-turn loop and tool support as map agents
        (Drift 9 fix — previously used a one-shot backend.generate() call).
        """
        reduce_prompt_template = kwargs.pop("reduce", None)
        reduce_schema = kwargs.pop("reduce_schema", None)
        if not reduce_prompt_template:
            raise ValueError("run_with_reduce requires a 'reduce' prompt")

        spec = BatchSpec(**kwargs)
        backend = backend_from_url(spec.backend)
        scheduler = cls._scheduler(spec, backend)
        results = await scheduler.run()

        # Build reduce input — all results (successful AND failed, per spec §3.4)
        all_outputs = []
        for r in results:
            if r.ok:
                output = r.output
                if hasattr(output, "model_dump"):
                    output = output.model_dump()
                elif hasattr(output, "dict"):
                    output = output.dict()
                all_outputs.append({"status": "ok", "index": r.index, "output": output})
            else:
                all_outputs.append({
                    "status": "error",
                    "index": r.index,
                    "error": {"type": r.error.type, "message": r.error.message},
                })

        reduce_text = reduce_prompt_template.format(n=len(results))
        results_json = json.dumps(all_outputs, default=str, ensure_ascii=False)
        full_reduce_prompt = f"{reduce_text}\n\nResults:\n{results_json}"

        # Build schema-injected prefix for the reduce agent
        reduce_prefix = spec.system_prompt or ""
        if reduce_schema:
            schema_dict: dict[str, Any] | None = None
            if hasattr(reduce_schema, "model_json_schema"):
                schema_dict = reduce_schema.model_json_schema()
            elif hasattr(reduce_schema, "schema"):
                schema_dict = reduce_schema.schema()
            elif isinstance(reduce_schema, dict):
                schema_dict = reduce_schema
            if schema_dict:
                reduce_prefix = (
                    reduce_prefix + f"\n\n{SCHEMA_INSTRUCTION}\n"
                    f"Schema:\n{json.dumps(schema_dict)}"
                ).strip()

        reduce_shared = SharedContext(prefix=reduce_prefix)

        # Drift 9 fix: run reduce through the full WaveScheduler multi-turn loop
        # Pass results as an input variable rather than embedding in task to avoid
        # format() interpreting JSON keys as template variables.
        reduce_spec = BatchSpec(
            task="{reduce_prompt}",
            inputs=[{"reduce_prompt": full_reduce_prompt}],
            system_prompt=reduce_prefix,
            tools=spec.tools,
            output_schema=reduce_schema,
            model=spec.model,
            backend=spec.backend,
            max_concurrent=1,
            max_turns=spec.max_turns,
            max_retries=spec.max_retries,
            timeout_per_agent=spec.timeout_per_agent,
            timeout_per_turn=spec.timeout_per_turn,
            timeout_per_tool=spec.timeout_per_tool,
            no_hoist=True,  # don't auto-hoist the single input
        )
        reduce_scheduler = cls._scheduler(reduce_spec, backend)
        reduce_results = await reduce_scheduler.run()
        if not reduce_results or not reduce_results[0].ok:
            error = reduce_results[0].error if reduce_results else None
            raise RuntimeError(f"Reduce agent failed: {error}")

        return results, reduce_results[0].output

    @classmethod
    def _scheduler(
        cls, spec: BatchSpec, backend: BackendAdapter | None = None
    ) -> WaveScheduler:
        plan = TaskCompiler().compile(spec)
        return WaveScheduler(plan, backend or backend_from_url(spec.backend))


__all__ = ["AgentResult", "BatchAgent", "BatchSpec", "Tool"]
