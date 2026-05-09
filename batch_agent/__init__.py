from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .backends import BackendAdapter, backend_from_url
from .compiler import SCHEMA_INSTRUCTION, TaskCompiler
from .repair import parse_and_validate_output
from .scheduler import WaveScheduler
from .spec import AgentResult, BatchSpec, SharedContext
from .tools import Tool
from .utils import extract_schema, to_jsonable


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
        """Run batch agents then a reduce agent that sees all results.

        Returns (individual_results, reduce_output).
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
                all_outputs.append({"status": "ok", "index": r.index,
                                    "output": to_jsonable(r.output)})
            else:
                all_outputs.append({"status": "error", "index": r.index,
                                    "error": {"type": r.error.type,
                                              "message": r.error.message}})

        reduce_text = reduce_prompt_template.format(n=len(results))
        results_json = json.dumps(all_outputs, default=str, ensure_ascii=False)
        full_reduce_prompt = f"{reduce_text}\n\nResults:\n{results_json}"

        # Build schema-injected prefix for the reduce agent
        reduce_prefix = spec.system_prompt or ""
        schema_dict = extract_schema(reduce_schema)
        if schema_dict:
            reduce_prefix = (
                reduce_prefix + f"\n\n{SCHEMA_INSTRUCTION}\n"
                f"Schema:\n{json.dumps(schema_dict)}"
            ).strip()

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
            no_hoist=True,
        )
        reduce_scheduler = cls._scheduler(reduce_spec, backend)
        reduce_results = await reduce_scheduler.run()
        if not reduce_results or not reduce_results[0].ok:
            error = reduce_results[0].error if reduce_results else None
            raise RuntimeError(f"Reduce agent failed: {error}")

        return results, reduce_results[0].output

    @classmethod
    async def run_with_map_reduce(
        cls,
        *,
        plan_prompt: str,
        plan_inputs: dict[str, Any] | None = None,
        plan_output_schema: Any,
        planner_model: str | None = None,
        task: str,
        output_schema: Any,
        worker_model: str | None = None,
        reduce: str,
        reduce_schema: Any,
        reducer_model: str | None = None,
        tools: list[Any] | tuple[Any, ...] = (),
        model: str,
        backend: str,
        max_concurrent: int = 64,
        max_turns: int = 6,
        on_result: Any | None = None,
        checkpoint_dir: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[AgentResult], Any]:
        """Run a three-tier plan → map → reduce agent topology."""
        plan_inputs = plan_inputs or {}
        stage_checkpoint = _stage_checkpoint_factory(checkpoint_dir)

        planner_results = await cls.run(
            task=plan_prompt,
            inputs=[plan_inputs],
            tools=tools,
            output_schema=plan_output_schema,
            model=planner_model or model,
            backend=backend,
            max_concurrent=1,
            max_turns=max_turns,
            checkpoint_dir=stage_checkpoint("plan"),
            **kwargs,
        )
        if not planner_results or not planner_results[0].ok:
            error = planner_results[0].error if planner_results else None
            raise RuntimeError(f"Planner agent failed: {error}")

        planner_output = planner_results[0].output
        items = _extract_plan_items(planner_output)
        map_inputs = [{"item": item, "index": index} for index, item in enumerate(items)]

        map_results = await cls.run(
            task=task,
            inputs=map_inputs,
            tools=tools,
            output_schema=output_schema,
            model=worker_model or model,
            backend=backend,
            max_concurrent=max_concurrent,
            max_turns=max_turns,
            on_result=on_result,
            checkpoint_dir=stage_checkpoint("map"),
            **kwargs,
        )

        reduce_payload = []
        for result in map_results:
            if result.ok:
                reduce_payload.append({
                    "status": "ok",
                    "index": result.index,
                    "item": items[result.index] if result.index < len(items) else None,
                    "output": to_jsonable(result.output),
                })
            else:
                reduce_payload.append({
                    "status": "error",
                    "index": result.index,
                    "item": items[result.index] if result.index < len(items) else None,
                    "error": {
                        "type": result.error.type if result.error else "Unknown",
                        "message": result.error.message if result.error else "unknown error",
                    },
                })

        reduce_text = reduce.format(n=len(map_results))
        reduce_prompt = (
            f"{reduce_text}\n\n"
            f"Planner output:\n{json.dumps(to_jsonable(planner_output), default=str, ensure_ascii=False)}\n\n"
            f"Results:\n{json.dumps(reduce_payload, default=str, ensure_ascii=False)}"
        )
        reduce_results = await cls.run(
            task="{reduce_prompt}",
            inputs=[{"reduce_prompt": reduce_prompt}],
            tools=tools,
            output_schema=reduce_schema,
            model=reducer_model or model,
            backend=backend,
            max_concurrent=1,
            max_turns=max_turns,
            checkpoint_dir=stage_checkpoint("reduce"),
            no_hoist=True,
            **kwargs,
        )
        if not reduce_results or not reduce_results[0].ok:
            error = reduce_results[0].error if reduce_results else None
            raise RuntimeError(f"Reduce agent failed: {error}")

        return map_results, reduce_results[0].output

    @classmethod
    def _scheduler(
        cls, spec: BatchSpec, backend: BackendAdapter | None = None
    ) -> WaveScheduler:
        plan = TaskCompiler().compile(spec)
        return WaveScheduler(plan, backend or backend_from_url(spec.backend))


def _stage_checkpoint_factory(checkpoint_dir: str | None):
    def stage(name: str) -> str | None:
        if checkpoint_dir is None:
            return None
        return str(Path(checkpoint_dir) / name)

    return stage


def _extract_plan_items(planner_output: Any) -> list[str]:
    raw_items = None
    if hasattr(planner_output, "items") and not callable(getattr(planner_output, "items")):
        raw_items = getattr(planner_output, "items")
    elif isinstance(planner_output, dict):
        raw_items = planner_output.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("plan_output_schema output must contain an items: list[str] field")
    return [str(item) for item in raw_items]


__all__ = ["AgentResult", "BatchAgent", "BatchSpec", "Tool"]
