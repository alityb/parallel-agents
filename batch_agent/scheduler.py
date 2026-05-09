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
from .compaction import compact_messages_async, should_compact
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

        # PrioritySemaphore: lower turns_remaining = served first (Drift 2)
        self._semaphore = PrioritySemaphore(plan.spec.max_concurrent)
        # Resolved tool definitions for tools specified in the spec
        self._tools = self._resolve_tools(plan.spec.tools)
        # Anthropic-format tool schemas for the backend
        self._tool_schemas = self._build_tool_schemas(self._tools)
        # kv_key for the shared prefix (set after warm_prefix, stored on each state)
        self._shared_kv_key: str | None = None
        # Checkpoint store for crash recovery and per-turn saves
        self._checkpoint = None
        if plan.spec.checkpoint_dir:
            from .checkpoint import CheckpointStore
            self._checkpoint = CheckpointStore(plan.spec.checkpoint_dir)
        # Event set when dispatch_loop should release one more agent
        self._dispatch_token: asyncio.Queue[None] = asyncio.Queue()

    # ── public API ─────────────────────────────────────────────────────────────

    async def run(self) -> list[AgentResult]:
        results: list[AgentResult | None] = [None] * len(self.plan.jobs)
        async for result in self.stream():
            results[result.index] = result
        return [result for result in results if result is not None]

    async def stream(self) -> AsyncIterator[AgentResult]:
        # Warm the shared prefix and store the kv_key on all future states
        self._shared_kv_key = await self.backend.warm_prefix(
            self.plan.shared, self.plan.spec.model
        )

        # Check for already-completed jobs (crash recovery)
        completed_ids: set[str] = set()
        if self._checkpoint:
            completed_ids = self._checkpoint.get_completed_job_ids()

        # Build priority queue: (max_turns, tiebreaker, job)
        # Lower max_turns = dispatched first among fresh agents
        pq: list[tuple[int, int, AgentJob]] = []
        skipped_results: list[AgentResult] = []
        for i, job in enumerate(self.plan.jobs):
            if job.job_id in completed_ids and self._checkpoint:
                prev = self._checkpoint.load_result(job.job_id)
                if prev:
                    skipped_results.append(prev)
                    continue
            heapq.heappush(pq, (job.max_turns, i, job))

        # Seed the dispatch queue with max_concurrent initial tokens
        for _ in range(min(self.plan.spec.max_concurrent, len(pq))):
            await self._dispatch_token.put(None)

        result_queue: asyncio.Queue[AgentResult | None] = asyncio.Queue()
        tasks: list[asyncio.Task] = []
        dispatched = 0

        async def dispatch_loop() -> None:
            nonlocal dispatched
            while pq:
                await self._dispatch_token.get()  # wait for a dispatch token
                if not pq:
                    break
                _, _, job = heapq.heappop(pq)
                task = asyncio.create_task(
                    self._run_job(job, result_queue),
                    name=f"agent-{job.job_id}",
                )
                tasks.append(task)
                dispatched += 1

        # Start adaptive concurrency controller
        adaptive_task = asyncio.create_task(
            self._adaptive_concurrency_loop(), name="adaptive-concurrency"
        )

        # Yield previously completed results immediately
        for sr in skipped_results:
            yield sr

        dispatcher = asyncio.create_task(dispatch_loop(), name="dispatcher")

        remaining = len(pq) + len(pq)  # will be corrected after dispatch
        # Actually count properly: total jobs minus already-skipped
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
            dispatcher.cancel()
            try:
                await asyncio.gather(adaptive_task, dispatcher, return_exceptions=True)
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
        """Execute a job, put result on queue, and release a dispatch token."""
        result = await self._execute(job)
        if self._checkpoint:
            self._checkpoint.save_result(result)
        callback = self.plan.spec.on_result
        if callback:
            maybe_awaitable = callback(result)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        await result_queue.put(result)
        # This agent is fully done → dispatch one new PENDING agent
        await self._dispatch_token.put(None)

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
        dispatch_released = False  # have we already released a dispatch token?

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
                    self.backend.generate(
                        shared=self.plan.shared,
                        job=job,
                        messages=state.messages,
                        model=self.plan.spec.model,
                        tools=self._tool_schemas if self._tools else None,
                        timeout=self.plan.spec.timeout_per_agent,
                    ),
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

                # === STAGGERED DISPATCH: release a dispatch token on TOOL_WAIT (Drift 10) ===
                if not dispatch_released:
                    dispatch_released = True
                    await self._dispatch_token.put(None)

                logger.info(
                    "[%s] turn=%d TOOL_WAIT: %d tool calls (semaphore free)",
                    job.job_id, state.turn, len(response.tool_calls),
                )

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
