"""Live integration test with local mock Anthropic server (proves end-to-end path)."""
from __future__ import annotations

import asyncio
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from pydantic import BaseModel

import sys
sys.path.insert(0, ".")
from batch_agent import BatchAgent
from batch_agent.backends.anthropic import AnthropicBackend
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import BatchSpec


class NumberFact(BaseModel):
    n: int
    msg: str


class MockAnthropicHandler(BaseHTTPRequestHandler):
    """Simulates Anthropic Messages API responses."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        user_msg = body.get("messages", [{}])[0].get("content", "")
        # Parse index from "Input: {index}" pattern
        idx = 0
        for word in user_msg.replace(":", " ").split():
            if word.isdigit():
                idx = int(word)
                break

        n = (idx * 7 + 13) % 100 + 1
        response_json = json.dumps({"n": n, "msg": f"The number {n} is interesting because it follows {n-1}."})
        anthropic_response = {
            "id": f"msg_{idx}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": response_json}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
        payload = json.dumps(anthropic_response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


async def main() -> None:
    # Start mock server
    server = HTTPServer(("127.0.0.1", 19283), MockAnthropicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # let server bind

    print(f"[{time.strftime('%H:%M:%S')}] Starting 20-agent integration test (mock Anthropic at localhost:19283)...")
    start = time.monotonic()

    spec = BatchSpec(
        task='Return a JSON object with keys "n" (a random number 1-100) and "msg" (one sentence about that number). Input: {index}',
        inputs=[{"index": i} for i in range(20)],
        output_schema=NumberFact,
        model="claude-sonnet-4-20250514",
        backend="anthropic://",
        max_concurrent=10,
        max_retries=2,
        timeout_per_agent=10,
    )

    backend = AnthropicBackend(api_key="test-key", base_url="http://127.0.0.1:19283")
    plan = TaskCompiler().compile(spec)
    scheduler = WaveScheduler(plan, backend)

    received = 0
    ok_count = 0
    async for result in scheduler.stream():
        received += 1
        elapsed = time.monotonic() - start
        if result.ok:
            ok_count += 1
            print(f"  [{elapsed:5.2f}s] job={result.job_id:6s} n={result.output.n:3d} msg={result.output.msg!r}")
        else:
            print(f"  [{elapsed:5.2f}s] job={result.job_id:6s} FAILED: {result.error.type}: {result.error.message}")

    total = time.monotonic() - start
    print(f"\n[{time.strftime('%H:%M:%S')}] Done. {ok_count}/{received} OK in {total:.2f}s")
    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
