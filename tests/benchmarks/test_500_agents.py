"""500-agent benchmark — Phase 2 success criteria validation.

Success criteria (from agents.md):
- 500 agents complete with <= 2% failure rate (after retries)
- No OOM crashes
- No unhandled exceptions

This test uses a mock backend to verify orchestration at scale.
Real GPU/API benchmarks require live infrastructure.
"""
from __future__ import annotations

import asyncio
import json
import random
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


class BenchResult(BaseModel):
    index: int
    summary: str


# --- Mock backend: simulates varying response times and occasional tool calls ---

class MockBench500Handler(BaseHTTPRequestHandler):
    """Simulates 500 agents with:
    - ~80% single-turn (just return JSON)
    - ~20% multi-turn (require 1 tool call then final)
    - ~1% transient failure (first attempt fails, retry succeeds)
    """
    request_count = 0
    failure_injection_rate = 0.01

    def do_POST(self):
        MockBench500Handler.request_count += 1
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        messages = body.get("messages", [])

        # Inject transient failures
        if random.random() < self.failure_injection_rate:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            payload = json.dumps({"error": "transient failure"}).encode()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Determine agent index from first user message
        user_msg = messages[0].get("content", "") if messages else ""
        idx = 0
        # Parse "Process item {index}" — find the number after "item"
        parts = user_msg.split()
        for i, word in enumerate(parts):
            if word == "item" and i + 1 < len(parts):
                try:
                    idx = int(parts[i + 1].rstrip("."))
                except ValueError:
                    pass
                break

        # Check if this is a multi-turn agent returning from tool call
        has_tool_result = any(
            isinstance(m.get("content"), list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m.get("content", [])
            )
            for m in messages
        )

        # ~20% of agents need a tool call (deterministic by index)
        needs_tool = (idx % 5 == 0) and not has_tool_result

        if needs_tool:
            resp = {
                "id": f"msg_{idx}_tc",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Fetching data..."},
                    {"type": "tool_use", "id": f"toolu_{idx}", "name": "http_get",
                     "input": {"url": f"http://127.0.0.1:19291/data/{idx}"}},
                ],
                "model": "mock",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 50, "output_tokens": 30},
            }
        else:
            final_json = json.dumps({"index": idx, "summary": f"Agent {idx} completed successfully."})
            resp = {
                "id": f"msg_{idx}_final",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": final_json}],
                "model": "mock",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 20},
            }

        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class MockDataHandler500(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = b"mock tool response data"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


async def main() -> None:
    random.seed(42)  # Reproducible

    # Start servers
    data_server = HTTPServer(("127.0.0.1", 19291), MockDataHandler500)
    threading.Thread(target=data_server.serve_forever, daemon=True).start()

    api_server = HTTPServer(("127.0.0.1", 19292), MockBench500Handler)
    threading.Thread(target=api_server.serve_forever, daemon=True).start()
    time.sleep(0.1)

    N = 500
    print(f"=== 500-Agent Benchmark ===")
    print(f"  Agents: {N}")
    print(f"  max_concurrent: 64")
    print(f"  max_turns: 5")
    print(f"  max_retries: 3")
    print(f"  ~20% require tool calls, ~1% transient failures")
    print()

    spec = BatchSpec(
        task="Process item {index}. Return JSON with index and summary.",
        inputs=[{"index": i} for i in range(N)],
        output_schema=BenchResult,
        model="mock",
        backend="anthropic://",
        max_concurrent=64,
        max_turns=5,
        max_retries=3,
        timeout_per_agent=30,
        timeout_per_turn=10,
        timeout_per_tool=5,
        tools=[Tool.http_get],
    )

    backend = AnthropicBackend(api_key="test-key", base_url="http://127.0.0.1:19292")
    plan = TaskCompiler().compile(spec)
    scheduler = WaveScheduler(plan, backend)

    start = time.monotonic()
    results: list = []
    ok_count = 0
    fail_count = 0

    async for result in scheduler.stream():
        results.append(result)
        if result.ok:
            ok_count += 1
        else:
            fail_count += 1

    total = time.monotonic() - start

    # Report
    print(f"  Results: {len(results)}/{N}")
    print(f"  OK: {ok_count}, Failed: {fail_count}")
    print(f"  Failure rate: {fail_count/N*100:.1f}%")
    print(f"  Wall-clock: {total:.2f}s")
    print(f"  Throughput: {N/total:.0f} agents/sec")
    print(f"  API calls made: {MockBench500Handler.request_count}")
    print()

    # Verify success criteria
    failure_rate = fail_count / N
    assert len(results) == N, f"Expected {N} results, got {len(results)}"
    assert failure_rate <= 0.02, f"Failure rate {failure_rate:.1%} exceeds 2% threshold"

    # Verify output correctness for successful results
    for r in results:
        if r.ok:
            assert r.output.index == r.index, f"Output index mismatch: {r.output.index} != {r.index}"

    print(f"\n  [PASS] All {N} agents completed")
    print(f"  [PASS] Failure rate {failure_rate:.1%} <= 2%")
    print(f"  [PASS] No unhandled exceptions")
    print(f"  [PASS] Output correctness verified")
    print(f"  [PASS] 500-agent benchmark passed")

    data_server.shutdown()
    api_server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
