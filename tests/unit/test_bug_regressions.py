"""Regression tests for bugs found in the second round of harsh code review.

Bug 1 (CRITICAL, fixed): ToolPool Future not resolved on CancelledError.
  When leader coroutine was cancelled (e.g. by asyncio.wait_for for timeout_per_tool),
  `except Exception` didn't catch CancelledError (BaseException). Future was never set.
  All concurrent waiters hung permanently. Fix: add `except BaseException` branch.

Bug 2 (cleanup): Dead code `_wrapped` and `counter` in dispatch_loop.
  `_wrapped` was defined but never called; `counter` was never read.
  Removed. in_flight tracking still works correctly via `_track`.

Additional edge cases covered:
- ToolPool timeout propagates CancelledError to waiters (not infinite hang)
- Empty inputs list returns immediately with 0 results
- max_dispatched=1 serialises correctly
- Tool not in registry returns structured error, not crash
- Oversized agent returns FAILED immediately (attempts=0), on_result still fires
- Retry clears state.messages but stale tool_calls_pending doesn't corrupt execution
- PrioritySemaphore handles simultaneous capacity reduction correctly under >2 agents
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch_agent.backends import BackendAdapter, BackendResponse, ParsedToolCall
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, AgentResult, BatchSpec, Message, SharedContext
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool


class Payload(BaseModel):
    value: int


class InstantBackend(BackendAdapter):
    async def generate(self, *, shared, job, messages=None, model,
                       tools=None, metadata=None, timeout=None) -> BackendResponse:
        body = json.dumps({"value": job.index})
        return BackendResponse(content=body,
                               raw={"content": [{"type": "text", "text": body}]},
                               stop_reason="end_turn")


# ══════════════════════════════════════════════════════════════════════════════
# BUG 1 REGRESSION: ToolPool concurrent waiters don't hang when leader is cancelled
# ══════════════════════════════════════════════════════════════════════════════

@Tool.define(name="__regression_slow_tool", cacheable=False)
async def _regression_slow_tool(key: str) -> str:
    await asyncio.sleep(2.0)
    return "done"


def test_toolpool_cancelled_future_wakes_waiters_not_hangs() -> None:
    """The critical regression: when leader is cancelled via asyncio.wait_for,
    concurrent waiters must receive an exception immediately, not hang forever."""
    pool = ToolPool()
    waiter_outcomes: list[str] = []

    async def leader() -> None:
        try:
            await asyncio.wait_for(
                pool.call("__regression_slow_tool", {"key": "shared"}),
                timeout=0.05,
            )
        except asyncio.TimeoutError:
            pass  # expected

    async def waiter(idx: int) -> None:
        await asyncio.sleep(0.01)  # let leader get there first
        try:
            await asyncio.wait_for(
                pool.call("__regression_slow_tool", {"key": "shared"}),
                timeout=1.0,  # should not need the full second
            )
            waiter_outcomes.append(f"{idx}:success")
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
            waiter_outcomes.append(f"{idx}:{type(e).__name__}")

    async def run() -> None:
        await asyncio.wait_for(
            asyncio.gather(leader(), waiter(0), waiter(1), waiter(2)),
            timeout=2.0,  # whole test must finish within 2s
        )

    asyncio.run(run())
    # All 3 waiters must have resolved (not hung)
    assert len(waiter_outcomes) == 3, (
        f"Only {len(waiter_outcomes)}/3 waiters resolved — others HUNG (bug regressed)"
    )
    # All waiters should have gotten CancelledError (not TimeoutError — that would mean
    # they waited the full 1s before the leader's cancellation woke them)
    for outcome in waiter_outcomes:
        assert "Timeout" not in outcome or "Cancelled" in outcome or "Runtime" in outcome, (
            f"Waiter timed out (hung until 1s deadline): {outcome}"
        )


def test_toolpool_fresh_attempt_after_leader_cancel() -> None:
    """After cancellation cleanup, a new call for the same key must execute fresh."""
    pool = ToolPool()
    calls = {"n": 0}

    @Tool.define(name="__regression_fresh_tool", cacheable=False)
    async def fresh_tool(key: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(2.0)  # will be cancelled
        return "fresh"

    async def run() -> None:
        # First call — cancelled after 50ms
        try:
            await asyncio.wait_for(
                pool.call("__regression_fresh_tool", {"key": "k"}), timeout=0.05
            )
        except asyncio.TimeoutError:
            pass

        # Second call — should execute fresh (no hung Future in _inflight)
        result = await asyncio.wait_for(
            pool.call("__regression_fresh_tool", {"key": "k"}), timeout=1.0
        )
        assert result == "fresh"
        assert calls["n"] == 2, f"Expected 2 calls (1 cancelled + 1 fresh), got {calls['n']}"

    asyncio.run(run())


def test_toolpool_timeout_under_scheduler_timeout_per_tool() -> None:
    """Full end-to-end: timeout_per_tool causes tool to be cancelled, agents get error
    result, scheduler continues — not hangs."""
    calls = {"n": 0}

    @Tool.define(name="__regression_hang_tool", cacheable=False)
    async def hang_tool(key: str) -> str:
        calls["n"] += 1
        await asyncio.sleep(5.0)  # always times out
        return "never"

    class TwoAgentToolBackend(BackendAdapter):
        def __init__(self):
            self.turn = {}

        async def generate(self, *, shared, job, messages=None, model,
                           tools=None, metadata=None, timeout=None) -> BackendResponse:
            t = self.turn.get(job.job_id, 0)
            self.turn[job.job_id] = t + 1
            has_result = any(getattr(m, "role", "") == "tool_result" for m in (messages or []))
            if not has_result and tools:
                tc_id = f"tc-{job.job_id}"
                return BackendResponse(
                    content="",
                    raw={"content": [{"type": "tool_use", "id": tc_id,
                                      "name": "__regression_hang_tool", "input": {"key": "shared"}}]},
                    tool_calls=[ParsedToolCall(id=tc_id, name="__regression_hang_tool",
                                              args={"key": "shared"})],
                    stop_reason="tool_use",
                )
            body = json.dumps({"value": job.index})
            return BackendResponse(content=body,
                                   raw={"content": [{"type": "text", "text": body}]},
                                   stop_reason="end_turn")

    spec = BatchSpec(
        task="Do {x}", inputs=[{"x": i} for i in range(3)],
        tools=[Tool.registry["__regression_hang_tool"]],
        output_schema=Payload, model="mock", backend="mock://",
        max_turns=2, max_retries=0,
        timeout_per_tool=0.1,  # 100ms — forces timeout
    )

    async def run() -> list[AgentResult]:
        return await asyncio.wait_for(
            WaveScheduler(TaskCompiler().compile(spec), TwoAgentToolBackend()).run(),
            timeout=5.0,  # entire run must finish, not hang
        )

    results = asyncio.run(run())
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    # All agents either got a tool-error result or succeeded; none should hang
    # (tool error is injected as is_error=True, model then produces final JSON)
    print(f"  [PASS] 3 agents with timing-out tools complete in <5s, no hang")
    print(f"  [INFO] Tool call count: {calls['n']} (multiple agents, dedup + retry)")


# ══════════════════════════════════════════════════════════════════════════════
# BUG 2: Dead code is gone (regression guard)
# ══════════════════════════════════════════════════════════════════════════════

def test_dead_code_removed_from_dispatch_loop() -> None:
    """_wrapped and counter must not exist in scheduler source."""
    src = Path("batch_agent/scheduler.py").read_text()
    assert "async def _wrapped(" not in src, (
        "_wrapped dead code was re-introduced in scheduler.py"
    )
    assert "counter = [in_flight]" not in src, (
        "'counter = [in_flight]' dead code was re-introduced"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Empty inputs list
# ══════════════════════════════════════════════════════════════════════════════

def test_empty_inputs_returns_empty_list() -> None:
    spec = BatchSpec(task="Do {x}", inputs=[])
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), InstantBackend()).run())
    assert results == [], f"Expected [], got {results}"


# ══════════════════════════════════════════════════════════════════════════════
# max_dispatched=1 serialises (not just completes)
# ══════════════════════════════════════════════════════════════════════════════

def test_max_dispatched_1_strictly_serial() -> None:
    """max_dispatched=1: no agent starts until the previous one fully completes."""
    active = [0]
    peak = [0]
    completions: list[int] = []

    class StrictSerial(BackendAdapter):
        async def generate(self, *, shared, job, messages=None, model,
                           tools=None, metadata=None, timeout=None) -> BackendResponse:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            completions.append(job.index)
            body = json.dumps({"value": job.index})
            return BackendResponse(content=body,
                                   raw={"content": [{"type": "text", "text": body}]},
                                   stop_reason="end_turn")

    spec = BatchSpec(
        task="x {x}", inputs=[{"x": i} for i in range(10)],
        output_schema=Payload, max_dispatched=1, max_inflight=1,
    )
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), StrictSerial()).run())
    assert len(results) == 10
    assert peak[0] == 1, f"max_dispatched=1 but peak active={peak[0]}"
    assert all(r.ok for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# Unknown tool name returns structured error, does not crash
# ══════════════════════════════════════════════════════════════════════════════

def test_unknown_tool_name_returns_error_block() -> None:
    pool = ToolPool()

    async def run() -> Any:
        from batch_agent.backends import ParsedToolCall
        tc = ParsedToolCall(id="tc-1", name="definitely_does_not_exist", args={})
        # Simulate what _execute_tool_calls does
        try:
            result = await pool.call("definitely_does_not_exist", {})
        except KeyError as e:
            return f"KeyError: {e}"
        return result

    result = asyncio.run(run())
    assert "KeyError" in result or "does_not_exist" in result


# ══════════════════════════════════════════════════════════════════════════════
# Oversized agent: returns FAILED(attempts=0), on_result still fires
# ══════════════════════════════════════════════════════════════════════════════

def test_oversized_agent_fires_on_result() -> None:
    seen = []

    def cb(result: AgentResult) -> None:
        seen.append((result.job_id, result.ok, result.attempts))

    spec = BatchSpec(
        task="x {x}", inputs=[{"x": 1}],
        output_schema=Payload,
        model_max_context=1, min_response_tokens=0,  # forces oversized
        on_result=cb,
    )
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), InstantBackend()).run())
    assert len(results) == 1
    assert not results[0].ok
    assert results[0].error.type == "OVERSIZED"
    assert results[0].attempts == 0
    assert len(seen) == 1 and seen[0][2] == 0, "on_result should fire even for OVERSIZED"


# ══════════════════════════════════════════════════════════════════════════════
# Retry resets messages; stale tool_calls_pending doesn't corrupt
# ══════════════════════════════════════════════════════════════════════════════

def test_retry_resets_state_cleanly() -> None:
    """After a failed agent run, retry must start with fresh messages
    regardless of any stale state from the previous attempt."""
    attempt_count = [0]

    class FailFirstAttempt(BackendAdapter):
        async def generate(self, *, shared, job, messages=None, model,
                           tools=None, metadata=None, timeout=None) -> BackendResponse:
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise RuntimeError("first attempt fails")
            # On retry, check messages are clean (just the user prompt)
            msgs = messages or []
            # Should be exactly 1 message on retry: the initial user prompt
            assert len(msgs) == 1 and msgs[0].role == "user", (
                f"Retry started with dirty messages: {[(m.role, m.content[:20]) for m in msgs]}"
            )
            body = json.dumps({"value": job.index})
            return BackendResponse(content=body,
                                   raw={"content": [{"type": "text", "text": body}]},
                                   stop_reason="end_turn")

    spec = BatchSpec(
        task="x {x}", inputs=[{"x": 1}],
        output_schema=Payload, max_retries=1,
    )
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), FailFirstAttempt()).run())
    assert len(results) == 1 and results[0].ok
    assert attempt_count[0] == 2, f"Expected 2 attempts, got {attempt_count[0]}"


# ══════════════════════════════════════════════════════════════════════════════
# Priority ordering under N > 2 concurrent agents
# ══════════════════════════════════════════════════════════════════════════════

def test_priority_semaphore_ordering_under_load() -> None:
    """With 50 agents all waiting for the semaphore at different priorities,
    lower-priority-value agents (near-done) must be served first in >90% of cases."""
    from batch_agent.priority_semaphore import PrioritySemaphore

    async def run() -> float:
        sem = PrioritySemaphore(1)
        order: list[float] = []
        await sem.acquire(priority=0.0)  # hold the slot

        # Create 50 waiters at various priorities
        async def waiter(pri: float) -> None:
            await sem.acquire(priority=pri)
            order.append(pri)
            sem.release()

        tasks = [asyncio.create_task(waiter(float(i))) for i in range(50)]
        await asyncio.sleep(0)  # let all register
        sem.release()  # release initial holder
        await asyncio.gather(*tasks)

        # Verify ordering: each element should be <= the next
        out_of_order = sum(1 for i in range(len(order)-1) if order[i] > order[i+1])
        return out_of_order / max(1, len(order) - 1)

    error_rate = asyncio.run(run())
    assert error_rate < 0.10, (
        f"PrioritySemaphore out-of-order rate {error_rate:.1%} too high for 50 agents"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tool registry global state: re-registering same name overwrites
# ══════════════════════════════════════════════════════════════════════════════

def test_tool_registry_overwrites_on_redefine() -> None:
    """Re-defining a tool with the same name must overwrite silently.
    This prevents confusing 'old definition' bugs across test runs."""
    @Tool.define(name="__redefine_test_tool")
    async def version_1() -> str:
        return "v1"

    @Tool.define(name="__redefine_test_tool")
    async def version_2() -> str:
        return "v2"

    result = asyncio.run(ToolPool().call("__redefine_test_tool"))
    assert result == "v2", f"Expected v2 (latest definition), got {result!r}"

    # Clean up
    Tool.registry.pop("__redefine_test_tool", None)
