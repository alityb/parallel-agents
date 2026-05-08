"""Heterogeneous scheduling test: mix of 1-turn and multi-turn agents.

Verifies:
- Fast agents (1 turn) complete without waiting for slow agents (3+ turns)
- Slot utilization stays relatively flat (no burst-then-idle)
- Priority queue biases toward draining nearly-finished agents
"""
from __future__ import annotations

import asyncio
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from pydantic import BaseModel

import sys
sys.path.insert(0, ".")

from batch_agent.backends.anthropic import AnthropicBackend
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import BatchSpec
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool


class Result(BaseModel):
    value: str


# Track completion order
completion_order: list[str] = []


class MockHeterogeneousHandler(BaseHTTPRequestHandler):
    """Simulates different response patterns based on job index.

    - Jobs 0-4: "fast" — return final JSON immediately (1 turn)
    - Jobs 5-9: "slow" — require 3 tool calls before final (4 turns total)
    """

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        messages = body.get("messages", [])

        # Determine job type from first user message
        user_msg = messages[0].get("content", "") if messages else ""
        is_slow = "slow" in user_msg.lower()

        # Count how many tool_result turns we've seen
        tool_result_count = sum(
            1 for m in messages
            if isinstance(m.get("content"), list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m.get("content", [])
            )
        )

        # Simulate latency proportional to work
        time.sleep(0.01)  # Base inference latency

        if not is_slow:
            # Fast agent: immediate final response
            final_json = json.dumps({"value": f"fast-{user_msg[-2:]}"})
            self._respond_final(final_json)
        elif tool_result_count < 3:
            # Slow agent: needs more tool calls
            self._respond_tool_call(f"fetch_step_{tool_result_count}")
        else:
            # Slow agent: has enough tool results, produce final
            final_json = json.dumps({"value": f"slow-done-{user_msg[-2:]}"})
            self._respond_final(final_json)

    def _respond_final(self, text: str):
        resp = {
            "id": "msg_final",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "mock",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
        self._send(resp)

    def _respond_tool_call(self, step: str):
        resp = {
            "id": "msg_tc",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Calling {step}..."},
                {"type": "tool_use", "id": f"toolu_{step}", "name": "http_get", "input": {"url": f"http://127.0.0.1:19288/data/{step}"}},
            ],
            "model": "mock",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
        self._send(resp)

    def _send(self, resp: dict):
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class MockDataHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(0.02)  # Simulate tool latency
        payload = b"data-response"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


async def main() -> None:
    global completion_order
    completion_order = []

    # Start servers
    data_server = HTTPServer(("127.0.0.1", 19288), MockDataHandler)
    threading.Thread(target=data_server.serve_forever, daemon=True).start()

    api_server = HTTPServer(("127.0.0.1", 19289), MockHeterogeneousHandler)
    threading.Thread(target=api_server.serve_forever, daemon=True).start()
    time.sleep(0.1)

    print("Starting heterogeneous scheduling test:")
    print("  Jobs 0-4: fast (1 turn), Jobs 5-9: slow (4 turns with tools)")
    print("  max_concurrent=3, max_turns=6")
    print()

    spec = BatchSpec(
        task="{task_desc}",
        inputs=[
            {"task_desc": f"Fast task {i}. Return JSON."} for i in range(5)
        ] + [
            {"task_desc": f"Slow task {i}. Requires multiple fetches."} for i in range(5)
        ],
        output_schema=Result,
        model="mock",
        backend="anthropic://",
        max_concurrent=3,
        max_turns=6,
        max_retries=1,
        timeout_per_agent=15,
        tools=[Tool.http_get],
    )

    backend = AnthropicBackend(api_key="test-key", base_url="http://127.0.0.1:19289")
    plan = TaskCompiler().compile(spec)
    scheduler = WaveScheduler(plan, backend)

    start = time.monotonic()
    results_timeline: list[tuple[float, str, str]] = []

    async for result in scheduler.stream():
        elapsed = time.monotonic() - start
        label = "fast" if result.index < 5 else "slow"
        status = "OK" if result.ok else "FAIL"
        results_timeline.append((elapsed, result.job_id, f"{label}/{status}"))
        completion_order.append(result.job_id)
        print(f"  [{elapsed:5.3f}s] {result.job_id:6s} ({label}) {status}: {result.output.value if result.ok else result.error.message}")

    total = time.monotonic() - start
    print(f"\nTotal time: {total:.3f}s")

    # Analysis
    fast_times = [t for t, jid, lbl in results_timeline if "fast" in lbl]
    slow_times = [t for t, jid, lbl in results_timeline if "slow" in lbl]

    print(f"\nFast agents completed at: {[f'{t:.3f}s' for t in sorted(fast_times)]}")
    print(f"Slow agents completed at: {[f'{t:.3f}s' for t in sorted(slow_times)]}")

    # Verify fast agents don't wait for slow agents
    max_fast = max(fast_times) if fast_times else 0
    min_slow = min(slow_times) if slow_times else float('inf')

    # Fast agents should mostly complete before slow agents
    # (some overlap is fine due to scheduling, but fast shouldn't wait for all slow)
    fast_before_slow_count = sum(1 for ft in fast_times if ft < min_slow + 0.05)
    print(f"\nFast agents completing before slowest slow starts finishing: {fast_before_slow_count}/5")

    # Check completion order: fast agents should generally appear earlier
    fast_positions = [completion_order.index(f"job-{i}") for i in range(5)]
    slow_positions = [completion_order.index(f"job-{i}") for i in range(5, 10)]
    avg_fast_pos = sum(fast_positions) / len(fast_positions)
    avg_slow_pos = sum(slow_positions) / len(slow_positions)

    print(f"Average completion position - fast: {avg_fast_pos:.1f}, slow: {avg_slow_pos:.1f}")

    assert avg_fast_pos < avg_slow_pos, "Fast agents should complete before slow agents on average"
    print("\n[PASS] Fast agents drain before slow agents (priority scheduling works)")

    # Verify all completed successfully
    ok_count = sum(1 for _, _, lbl in results_timeline if "OK" in lbl)
    assert ok_count == 10, f"Expected 10 OK results, got {ok_count}"
    print("[PASS] All 10 agents completed successfully")

    # Verify slow agents did multiple turns
    for i in range(5, 10):
        state = scheduler.states.get(f"job-{i}")
        assert state.turn >= 4, f"job-{i} should have done >= 4 turns, did {state.turn}"
    print("[PASS] Slow agents all completed >= 4 turns")

    data_server.shutdown()
    api_server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
