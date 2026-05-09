"""Wave Scheduler — core asyncio orchestration engine.

Implements:
- Priority-ordered inference semaphore (Drift 2): agents with fewer turns_remaining
  jump ahead of fresh agents when competing for GPU slots.
- Staggered dispatch (Drift 10): new PENDING agents are dispatched when an existing
  agent enters TOOL_WAIT or COMPLETE — not all at t=0.
- Adaptive concurrency (Drift 3): polls prefix_cache_hit_rate every 10s, adjusts
  max_concurrent ±10%.
- Per-turn latency recording into AgentState.historical_turn_latencies (Phase 3A).
- Per-TOOL_WAIT duration recording into AgentState.tool_wait_durations (Phase 3A).
- kv_key stored on AgentState after warm_prefix (Phase 3A).
- Metrics written on every turn (Drift 11).
- Checkpoint save_state after every turn (Stub 5).
"""
from __future__ import annotations

import asyncio
import heapq
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .backends import BackendAdapter, BackendResponse, ParsedToolCall
from .backpressure import BackpressureController, calibrate_max_inflight
from .compaction import compact_messages_async, should_compact
from .kvflow import KVFlowAdvisor
from .metrics import SchedulerMetrics
from .priority_semaphore import PrioritySemaphore
from .repair import parse_and_validate_output
from .schema import build_tool_schemas
from .spec import AgentError, AgentJob, AgentResult, ExecutionPlan, Message
from .state import AgentState, AgentStatus, InMemoryStateStore
from .tools import Tool, ToolDefinition
from .tools.pool import ToolPool

logger = logging.getLogger(__name__)

# Adaptive concurrency thresholds (spec §3.2)
ADAPTIVE_POLL_INTERVAL = 10.0    # seconds between metric polls
ADAPTIVE_HIT_RATE_LOW = 0.90     # below this → reduce by 10%
ADAPTIVE_HIT_RATE_HIGH = 0.98    # above this + low GPU util → increase by 10%
ADAPTIVE_GPU_UTIL_HIGH = 0.80    # GPU util threshold for increase
ADAPTIVE_CHANGE_FACTOR = 0.10    # ±10% per adjustment


class WaveScheduler:
    def __init__(
        self,
        plan: ExecutionPlan,
        backend: BackendAdapter,
        tool_pool: ToolPool | None = None,
    ) -> None:
        self.plan = plan
        self.backend = backend
        self.states = InMemoryStateStore()
        self.tool_pool = tool_pool or ToolPool()
        self.metrics = SchedulerMetrics()

        # PrioritySemaphore caps simultaneous in-flight HTTP requests (max_inflight)
        self._semaphore = PrioritySemaphore(plan.spec.effective_max_inflight)
        # Resolved tool definitions and schemas
        self._tools = self._resolve_tools(plan.spec.tools)
        self._tool_schemas = self._build_tool_schemas(self._tools)
        # kv_key for the shared prefix (stored on each AgentState)
        self._shared_kv_key: str | None = None
        self._kvflow_advisor: KVFlowAdvisor | None = None
        # Checkpoint store
        self._checkpoint = None
        if plan.spec.checkpoint_dir:
            from .checkpoint import CheckpointStore
            self._checkpoint = CheckpointStore(plan.spec.checkpoint_dir)
        # Backpressure controller (None = disabled, i.e. for API/mock backends)
        self._backpressure: BackpressureController | None = (
            BackpressureController(queue_depth_ceiling=plan.spec.backpressure_ceiling)
            if plan.spec.backpressure_ceiling > 0
            else None
        )

    # ── public API ─────────────────────────────────────────────────────────────

    async def run(self) -> list[AgentResult]:
        results: list[AgentResult | None] = [None] * len(self.plan.jobs)
        async for result in self.stream():
            results[result.index] = result
        return [result for result in results if result is not None]

    async def stream(self) -> AsyncIterator[AgentResult]:
        # Optional auto-calibration before first wave
        if self.plan.spec.calibrate_backend:
            calibrated = await calibrate_max_inflight(
                self.backend,
                self.plan.shared,
                self.plan.spec.model,
                backend_url=self.plan.spec.backend,
            )
            self._semaphore.set_capacity(calibrated)
            logger.info("Auto-calibrated max_inflight to %d", calibrated)

        # Warm the shared prefix and store the kv_key on all future states
        self._shared_kv_key = await self.backend.warm_prefix(
            self.plan.shared, self.plan.spec.model
        )

        # Check for already-completed jobs (crash recovery)
        completed_ids: set[str] = set()
        if self._checkpoint:
            completed_ids = self._checkpoint.get_completed_job_ids()

        # Build priority queue: (max_turns, tiebreaker, job)
        pq: list[tuple[int, int, AgentJob]] = []
        skipped_results: list[AgentResult] = []
        for i, job in enumerate(self.plan.jobs):
            if job.job_id in completed_ids and self._checkpoint:
                prev = self._checkpoint.load_result(job.job_id)
                if prev:
                    skipped_results.append(prev)
                    continue
            heapq.heappush(pq, (job.max_turns, i, job))

        result_queue: asyncio.Queue[AgentResult | None] = asyncio.Queue()
        tasks: list[asyncio.Task] = []

        # max_dispatched: how many tasks to create before waiting for results.
        # -1 = unlimited (dispatch all N immediately; max_inflight semaphore + backpressure control flow).
        max_dispatched = self.plan.spec.max_dispatched
        if max_dispatched < 0:
            max_dispatched = len(self.plan.jobs)  # unlimited → all jobs

        async def dispatch_loop() -> None:
            dispatched = 0
            in_flight = 0  # tasks created but not yet completed

            while pq:
                # Respect max_dispatched: if we'd exceed it, wait for a completion
                if in_flight >= max_dispatched:
                    await asyncio.sleep(0.005)
                    continue

                # Backpressure: pause if backend queue is full
                if self._backpressure:
                    await self._backpressure.wait_for_capacity(self.backend)

                _, _, job = heapq.heappop(pq)
                in_flight += 1

                async def _wrapped(j: AgentJob, counter: list) -> None:
                    await self._run_job(j, result_queue)
                    counter[0] -= 1

                counter = [in_flight]  # mutable reference so _wrapped can decrement
                task = asyncio.create_task(
                    self._run_job(job, result_queue),
                    name=f"agent-{job.job_id}",
                )
                tasks.append(task)
                dispatched += 1

                # Track completions to unblock max_dispatched gate
                async def _track(t: asyncio.Task) -> None:
                    nonlocal in_flight
                    try:
                        await t
                    except Exception:
                        pass
                    finally:
                        in_flight -= 1

                asyncio.create_task(_track(task))
                await asyncio.sleep(0)  # yield to event loop periodically

        # Start adaptive concurrency controller
        adaptive_task = asyncio.create_task(
            self._adaptive_concurrency_loop(), name="adaptive-concurrency"
        )
        kvflow_task = None
        if self.plan.spec.kvflow:
            self._kvflow_advisor = KVFlowAdvisor(
                state_store=self.states,
                tool_pool=self.tool_pool,
                backend=self.backend,
            )
            kvflow_task = asyncio.create_task(
                self._kvflow_advisor.run(),
                name="kvflow-advisor",
            )

        # Yield previously completed results immediately
        for sr in skipped_results:
            yield sr

        dispatcher = asyncio.create_task(dispatch_loop(), name="dispatcher")
        total_to_receive = len(self.plan.jobs) - len(skipped_results)
        received = 0

        try:
            while received < total_to_receive:
                item = await result_queue.get()
                if item is None:
                    break
                received += 1
                yield item
        finally:
            adaptive_task.cancel()
            if kvflow_task:
                kvflow_task.cancel()
            dispatcher.cancel()
            try:
                cancel_tasks = [adaptive_task, dispatcher]
                if kvflow_task:
                    cancel_tasks.append(kvflow_task)
                await asyncio.gather(*cancel_tasks, return_exceptions=True)
            except Exception:
                pass
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    # ── internal: per-job execution ────────────────────────────────────────────

    async def _run_job(
        self,
        job: AgentJob,
        result_queue: asyncio.Queue[AgentResult | None],
    ) -> None:
        """Execute a job and put the result on the result queue."""
        result = await self._execute(job)
        if self._checkpoint:
            self._checkpoint.save_result(result)
        callback = self.plan.spec.on_result
        if callback:
            maybe_awaitable = callback(result)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        await result_queue.put(result)

    async def _execute(self, job: AgentJob) -> AgentResult:
        state = None
        if self._checkpoint:
            state = self._checkpoint.load_state(job.job_id)
        if state is None:
            state = self.states.create(job.job_id)
        else:
            self.states.save(state)
        state.set_status(AgentStatus.PREFLIGHT)
        # Store the shared prefix kv_key on agent state (Phase 3A)
        state.kv_key = self._shared_kv_key

        if job.oversized:
            error = AgentError(
                "OVERSIZED", "prompt exceeds configured model context budget", retryable=False
            )
            state.error = error
            state.set_status(AgentStatus.FAILED)
            self.metrics.record_failure()
            return AgentResult(job_id=job.job_id, index=job.index, output=None, error=error, attempts=0)

        attempts = self.plan.spec.max_retries + 1
        for attempt in range(1, attempts + 1):
            state.retry_count = attempt - 1
            try:
                result = await self._run_agent_loop(job, state)
                self.metrics.record_completion()
                return AgentResult(job_id=job.job_id, index=job.index, output=result, attempts=attempt)
            except Exception as exc:
                retryable = attempt < attempts
                if retryable:
                    logger.debug("[%s] attempt %d failed (%s), retrying...", job.job_id, attempt, exc)
                    state.messages.clear()
                    state.turn = 0
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
                    continue
                error = AgentError(type=exc.__class__.__name__, message=str(exc), retryable=False)
                state.error = error
                state.set_status(AgentStatus.FAILED)
                self.metrics.record_failure()
                return AgentResult(
                    job_id=job.job_id, index=job.index, output=None, error=error, attempts=attempt
                )

        raise AssertionError("unreachable")

    async def _run_agent_loop(self, job: AgentJob, state: AgentState) -> Any:
        """Multi-turn agent loop.

        Semaphore contract (W5):
        - Acquired ONLY around backend.generate() — never held during tool execution.
        - Priority = turns_remaining so nearly-done agents jump ahead of fresh ones.
        - Releases a dispatch token on first TOOL_WAIT so a new PENDING agent starts.
        """
        max_turns = self.plan.spec.max_turns
        if not state.messages:
            state.messages = [Message(role="user", content=job.prompt)]
            start_turn = 0
        else:
            start_turn = state.turn
        state.set_status(AgentStatus.RUNNING)

        response: BackendResponse | None = None

        for turn in range(start_turn, max_turns):
            state.turn = turn + 1
            # priority = turns_remaining (lower = higher priority)
            priority = float(max_turns - state.turn)

            # === ACQUIRE SEMAPHORE: hold only during inference ===
            t_wait = time.monotonic()
            logger.info("[%s] turn=%d acquiring semaphore", job.job_id, state.turn)
            await self._semaphore.acquire(priority=priority)
            logger.info(
                "[%s] turn=%d semaphore acquired (waited %.3fs)",
                job.job_id, state.turn, time.monotonic() - t_wait,
            )

            t_generate = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self._generate_with_metadata(job, state),
                    timeout=self.plan.spec.timeout_per_turn,
                )
            finally:
                # === RELEASE SEMAPHORE: free GPU slot immediately after inference ===
                self._semaphore.release()
                logger.info("[%s] turn=%d semaphore released", job.job_id, state.turn)

            generate_elapsed = time.monotonic() - t_generate
            state.record_turn_latency(generate_elapsed)
            self.metrics.record_turn(job.job_id, generate_elapsed)
            logger.debug("[%s] turn=%d generate=%.3fs", job.job_id, state.turn, generate_elapsed)

            # Append assistant response to conversation history
            if response.raw and "content" in response.raw:
                state.messages.append(
                    Message(role="assistant_raw", content=json.dumps(response.raw["content"]))
                )
            elif response.raw and "choices" in response.raw:
                choice_msg = response.raw["choices"][0].get("message", {})
                blocks = []
                if choice_msg.get("content"):
                    blocks.append({"type": "text", "text": choice_msg["content"]})
                for tc in choice_msg.get("tool_calls", []):
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except Exception:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": args,
                    })
                state.messages.append(
                    Message(role="assistant_raw", content=json.dumps(blocks))
                )
            else:
                state.messages.append(Message(role="assistant", content=response.content))

            # Handle tool calls
            if response.tool_calls:
                state.set_status(AgentStatus.TOOL_WAIT)
                state.tool_calls_pending = [tc.to_tool_call() for tc in response.tool_calls if not tc.error]

                logger.info(
                    "[%s] turn=%d TOOL_WAIT: %d tool calls (semaphore free)",
                    job.job_id, state.turn, len(response.tool_calls),
                )

                # KVFlow: emit an immediate hint batch on TOOL_WAIT entry so short
                # tool waits (e.g. 300ms) are not missed by the 500ms background tick.
                if self._kvflow_advisor:
                    await self._kvflow_advisor.emit_once()

                t_tool = time.monotonic()
                tool_result_blocks = await self._execute_tool_calls(response.tool_calls)
                tool_elapsed = time.monotonic() - t_tool
                state.record_tool_wait(tool_elapsed)
                # Update KVFlow ETA fields (used by KVFlowAdvisor in Phase 3A)
                state.steps_to_execution = tool_elapsed  # last observed tool wait
                state.estimated_next_activation = time.time() + (state.p75_tool_wait() or tool_elapsed)

                state.messages.append(
                    Message(role="tool_result", content=json.dumps(tool_result_blocks))
                )
                state.tool_calls_pending = []
                state.set_status(AgentStatus.RUNNING)

                # Save only after a complete turn is durable: user + assistant + tool result.
                if self._checkpoint:
                    self._checkpoint.save_state(state)

                # Compact old tool results after every COMPACT_INTERVAL turns
                if should_compact(state.turn):
                    compaction_backend = (
                        self.backend if self.plan.spec.compaction_backend_url else None
                    )
                    state.messages = await compact_messages_async(
                        state.messages,
                        state.turn,
                        backend=compaction_backend,
                        model=self.plan.spec.model,
                    )

                continue

            # No tool calls → model produced a final response
            if response.is_final:
                output = parse_and_validate_output(response.content, self.plan.spec.output_schema)
                state.output = output
                state.set_status(AgentStatus.COMPLETE)
                if self._checkpoint:
                    self._checkpoint.save_state(state)
                return output

        # Exhausted max_turns — parse whatever we have
        assert response is not None
        output = parse_and_validate_output(response.content, self.plan.spec.output_schema)
        state.output = output
        state.set_status(AgentStatus.COMPLETE)
        if self._checkpoint:
            self._checkpoint.save_state(state)
        return output

    # ── tool execution ─────────────────────────────────────────────────────────

    async def _execute_tool_calls(
        self, tool_calls: list[ParsedToolCall]
    ) -> list[dict[str, Any]]:
        """Execute tool calls concurrently and return Anthropic tool_result blocks."""
        async def _one(tc: ParsedToolCall) -> dict[str, Any]:
            if tc.error:
                return {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"[ERROR] Malformed tool call: {tc.error_message}",
                    "is_error": True,
                }
            try:
                timeout = self.plan.spec.timeout_per_tool
                result = await asyncio.wait_for(
                    self.tool_pool.call(tc.name, tc.args), timeout=timeout
                )
                content = result if isinstance(result, str) else json.dumps(result, default=str)
                return {"type": "tool_result", "tool_use_id": tc.id, "content": content}
            except KeyError:
                logger.warning("[tool_pool] unknown tool: %s", tc.name)
                return {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"[ERROR] Unknown tool: {tc.name}",
                    "is_error": True,
                }
            except asyncio.TimeoutError:
                logger.warning("[tool_pool] tool %s timed out", tc.name)
                return {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"[ERROR] Tool timed out after {self.plan.spec.timeout_per_tool}s",
                    "is_error": True,
                }
            except Exception as exc:
                logger.warning("[tool_pool] tool %s failed: %s", tc.name, exc)
                return {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"[ERROR] Tool execution failed: {exc}",
                    "is_error": True,
                }

        # Execute all calls concurrently (they're all in TOOL_WAIT, semaphore is free)
        return list(await asyncio.gather(*[_one(tc) for tc in tool_calls]))

    async def _generate_with_metadata(self, job: AgentJob, state: AgentState) -> BackendResponse:
        """Call backend.generate and pass kv_key metadata when supported."""
        kwargs: dict[str, Any] = {
            "shared": self.plan.shared,
            "job": job,
            "messages": state.messages,
            "model": self.plan.spec.model,
            "tools": self._tool_schemas if self._tools else None,
            "timeout": self.plan.spec.timeout_per_agent,
        }
        if "metadata" in inspect.signature(self.backend.generate).parameters:
            kwargs["metadata"] = {"kv_key": state.kv_key, "job_id": state.job_id}
        return await self.backend.generate(**kwargs)

    # ── adaptive concurrency (Drift 3) ─────────────────────────────────────────

    async def _adaptive_concurrency_loop(self) -> None:
        """Poll backend cache metrics every ADAPTIVE_POLL_INTERVAL seconds.

        Adjusts _semaphore capacity:
        - hit_rate < 90%  → reduce by 10% (memory pressure)
        - hit_rate > 98% and gpu_util < 80% → increase by 10%
        """
        while True:
            try:
                await asyncio.sleep(ADAPTIVE_POLL_INTERVAL)
                metrics = await self.backend.get_cache_metrics()
                if not metrics:
                    continue
                if "recommended_concurrency" in metrics:
                    recommended = max(1, int(metrics["recommended_concurrency"]))
                    if recommended != self._semaphore.capacity:
                        current = self._semaphore.capacity
                        logger.info(
                            "Backend recommended concurrency: %d→%d",
                            current, recommended,
                        )
                        self._semaphore.set_capacity(recommended)
                        self.metrics.record_concurrency_change(current, recommended, "backend-recommendation")
                    continue
                hit_rate = metrics.get("prefix_cache_hit_rate", 1.0)
                gpu_util = metrics.get("gpu_utilization", 0.5)

                current = self._semaphore.capacity
                if hit_rate < ADAPTIVE_HIT_RATE_LOW:
                    new_cap = max(1, int(current * (1 - ADAPTIVE_CHANGE_FACTOR)))
                    if new_cap != current:
                        logger.info(
                            "Adaptive concurrency: hit_rate=%.1f%% < 90%%, reducing %d→%d",
                            hit_rate * 100, current, new_cap,
                        )
                        self._semaphore.set_capacity(new_cap)
                        self.metrics.record_concurrency_change(current, new_cap, "reduce")
                elif hit_rate > ADAPTIVE_HIT_RATE_HIGH and gpu_util < ADAPTIVE_GPU_UTIL_HIGH:
                    new_cap = int(current * (1 + ADAPTIVE_CHANGE_FACTOR))
                    logger.info(
                        "Adaptive concurrency: hit_rate=%.1f%% > 98%%, gpu_util=%.0f%%, increasing %d→%d",
                        hit_rate * 100, gpu_util * 100, current, new_cap,
                    )
                    self._semaphore.set_capacity(new_cap)
                    self.metrics.record_concurrency_change(current, new_cap, "increase")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Adaptive concurrency poll error: %s", e)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _resolve_tools(self, tools: Any) -> dict[str, ToolDefinition]:
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
        return build_tool_schemas(tools)
