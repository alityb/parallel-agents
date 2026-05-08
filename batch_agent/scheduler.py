from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .backends import BackendAdapter, BackendResponse, ParsedToolCall
from .repair import parse_and_validate_output
from .spec import AgentError, AgentJob, AgentResult, ExecutionPlan, Message
from .state import AgentStatus, InMemoryStateStore
from .tools import Tool, ToolDefinition
from .tools.pool import ToolPool

logger = logging.getLogger(__name__)


class WaveScheduler:
    def __init__(self, plan: ExecutionPlan, backend: BackendAdapter, tool_pool: ToolPool | None = None) -> None:
        self.plan = plan
        self.backend = backend
        self.states = InMemoryStateStore()
        self.tool_pool = tool_pool or ToolPool()
        self._semaphore = asyncio.Semaphore(plan.spec.max_concurrent)
        # Resolved tool definitions for tools specified in the spec
        self._tools = self._resolve_tools(plan.spec.tools)
        # Anthropic-format tool schemas for the backend
        self._tool_schemas = self._build_tool_schemas(self._tools)

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
                result = await self._run_agent_loop(job, state)
                return AgentResult(job_id=job.job_id, index=job.index, output=result, attempts=attempt)
            except Exception as exc:
                retryable = attempt < attempts
                if retryable:
                    logger.debug("[%s] attempt %d failed (%s), retrying...", job.job_id, attempt, exc)
                    # Reset state for retry
                    state.messages.clear()
                    state.turn = 0
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
                    continue
                error = AgentError(type=exc.__class__.__name__, message=str(exc), retryable=False)
                state.error = error
                state.set_status(AgentStatus.FAILED)
                return AgentResult(job_id=job.job_id, index=job.index, output=None, error=error, attempts=attempt)

        raise AssertionError("unreachable")

    async def _run_agent_loop(self, job: AgentJob, state: AgentState) -> Any:
        """Multi-turn agent loop with semaphore release during tool waits (W5)."""
        max_turns = self.plan.spec.max_turns

        # Initialize messages with the user prompt
        state.messages = [Message(role="user", content=job.prompt)]
        state.set_status(AgentStatus.RUNNING)

        for turn in range(max_turns):
            state.turn = turn + 1

            # === ACQUIRE SEMAPHORE: hold only during inference ===
            logger.info("[%s] turn=%d acquiring semaphore", job.job_id, state.turn)
            t_acquire = time.monotonic()
            await self._semaphore.acquire()
            logger.info("[%s] turn=%d semaphore acquired (waited %.3fs)",
                        job.job_id, state.turn, time.monotonic() - t_acquire)

            try:
                response = await self.backend.generate(
                    shared=self.plan.shared,
                    job=job,
                    messages=state.messages,
                    model=self.plan.spec.model,
                    tools=self._tool_schemas if self._tools else None,
                    timeout=self.plan.spec.timeout_per_agent,
                )
            finally:
                # === RELEASE SEMAPHORE: free GPU slot immediately after inference ===
                self._semaphore.release()
                logger.info("[%s] turn=%d semaphore released", job.job_id, state.turn)

            # Append the assistant's response to conversation history
            # Store raw content blocks so multi-turn tool_use/tool_result works
            if response.raw and "content" in response.raw:
                state.messages.append(Message(role="assistant_raw", content=json.dumps(response.raw["content"])))
            else:
                state.messages.append(Message(role="assistant", content=response.content))

            # Check if model requested tool calls
            if response.tool_calls:
                # === TOOL_WAIT: semaphore is NOT held during tool execution ===
                state.set_status(AgentStatus.TOOL_WAIT)
                state.tool_calls_pending = [tc.to_tool_call() for tc in response.tool_calls if not tc.error]
                logger.info("[%s] turn=%d TOOL_WAIT: %d tool calls (semaphore free)",
                            job.job_id, state.turn, len(response.tool_calls))

                # Execute all tool calls
                tool_result_blocks = await self._execute_tool_calls(response.tool_calls)

                # Append tool results as a user message (Anthropic format)
                state.messages.append(Message(role="tool_result", content=json.dumps(tool_result_blocks)))
                state.tool_calls_pending = []
                state.set_status(AgentStatus.RUNNING)
                logger.info("[%s] turn=%d tool results injected, continuing loop", job.job_id, state.turn)
                continue

            # No tool calls — model produced a final response
            if response.is_final:
                output = parse_and_validate_output(response.content, self.plan.spec.output_schema)
                state.output = output
                state.set_status(AgentStatus.COMPLETE)
                return output

        # Exhausted max_turns — try to parse whatever we have
        output = parse_and_validate_output(response.content, self.plan.spec.output_schema)
        state.output = output
        state.set_status(AgentStatus.COMPLETE)
        return output

    async def _execute_tool_calls(self, tool_calls: list[ParsedToolCall]) -> list[dict[str, Any]]:
        """Execute tool calls via the pool and return Anthropic-format tool_result blocks."""
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            if tc.error:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"[ERROR] Malformed tool call: {tc.error_message}",
                    "is_error": True,
                })
                continue

            try:
                result = await self.tool_pool.call(tc.name, tc.args)
                content = result if isinstance(result, str) else json.dumps(result, default=str)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": content,
                })
            except KeyError:
                logger.warning("[tool_pool] unknown tool: %s", tc.name)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"[ERROR] Unknown tool: {tc.name}",
                    "is_error": True,
                })
            except Exception as exc:
                logger.warning("[tool_pool] tool %s failed: %s", tc.name, exc)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"[ERROR] Tool execution failed: {exc}",
                    "is_error": True,
                })

        return results

    def _resolve_tools(self, tools: Any) -> dict[str, ToolDefinition]:
        """Resolve tool specs into ToolDefinition objects."""
        resolved: dict[str, ToolDefinition] = {}
        if not tools:
            return resolved
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                resolved[tool.name] = tool
            elif isinstance(tool, str):
                if tool in Tool.registry:
                    resolved[tool] = Tool.registry[tool]
                else:
                    logger.warning("Unknown tool name: %s", tool)
            else:
                logger.warning("Unrecognized tool spec: %s", tool)
        return resolved

    def _build_tool_schemas(self, tools: dict[str, ToolDefinition]) -> list[dict[str, Any]]:
        """Build Anthropic-format tool schemas from resolved definitions."""
        schemas: list[dict[str, Any]] = []
        for name, defn in tools.items():
            # Introspect function signature to build input_schema
            import inspect as _inspect
            sig = _inspect.signature(defn.func)
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                prop: dict[str, Any] = {"type": "string"}  # default to string
                if param.annotation is not _inspect.Parameter.empty:
                    prop = _annotation_to_schema(param.annotation)
                if param.default is _inspect.Parameter.empty:
                    required.append(param_name)
                properties[param_name] = prop

            schemas.append({
                "name": name,
                "description": (defn.func.__doc__ or f"Tool: {name}").strip(),
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        return schemas


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a basic JSON Schema type."""
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    return {"type": "string"}


# Re-export for type checking
from .state import AgentState as AgentState  # noqa: E402
