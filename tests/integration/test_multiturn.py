"""Integration test: multi-turn agent loop with tool calls.

Verifies:
1. Model requests tool_use → scheduler executes tools → re-injects results → model produces final output
2. Semaphore is released during TOOL_WAIT (W5 from agents.md)
3. 10 agents, each doing at least 2 turns (1 tool call + 1 final response)

Uses a local mock server that simulates Anthropic's Messages API with tool_use responses.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from typing import Any

from pydantic import BaseModel

import sys
sys.path.insert(0, ".")

from batch_agent.backends.anthropic import AnthropicBackend
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import BatchSpec
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool


# ─── Output schema ───────────────────────────────────────────────────────────

class PageSummary(BaseModel):
    url: str
    title: str
    word_count: int


# ─── Mock HTTP content server (simulates a real web page) ────────────────────

MOCK_PAGES = {
    f"/page/{i}": f"<html><head><title>Page {i} Title</title></head><body>{'Lorem ipsum dolor sit amet. ' * (20 + i * 5)}</body></html>"
    for i in range(10)
}


class MockWebServer(BaseHTTPRequestHandler):
    def do_GET(self):
        content = MOCK_PAGES.get(self.path, "<html><body>Not found</body></html>")
        payload = content.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


# ─── Mock Anthropic server (multi-turn: tool_use then final) ─────────────────

class MockAnthropicMultiTurn(BaseHTTPRequestHandler):
    """Turn 1: return tool_use calling http_get. Turn 2: return final JSON."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        messages = body.get("messages", [])

        # Determine which turn we're on by checking if tool_result is present
        has_tool_result = any(
            isinstance(m.get("content"), list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in m.get("content", [])
            )
            for m in messages
        )

        if not has_tool_result:
            # TURN 1: request tool call
            # Extract the URL from the user's message
            user_msg = messages[0].get("content", "") if messages else ""
            # URL pattern: http://127.0.0.1:{port}/page/{idx}
            url = ""
            for word in user_msg.split():
                if word.startswith("http"):
                    url = word.rstrip(".")
                    break

            response_body = {
                "id": "msg_turn1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"I'll fetch the page at {url} to summarize it."},
                    {
                        "type": "tool_use",
                        "id": f"toolu_{hash(url) % 99999:05d}",
                        "name": "http_get",
                        "input": {"url": url},
                    },
                ],
                "model": body.get("model", "mock"),
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        else:
            # TURN 2: produce final structured output based on tool result
            # Find the tool result content
            tool_content = ""
            for m in messages:
                if isinstance(m.get("content"), list):
                    for block in m["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            break

            # Parse a title from the HTML
            title = "Unknown"
            if "<title>" in tool_content:
                title = tool_content.split("<title>")[1].split("</title>")[0]
            word_count = len(tool_content.split())

            # Find the original URL
            user_msg = messages[0].get("content", "") if messages else ""
            url = ""
            for word in user_msg.split():
                if word.startswith("http"):
                    url = word.rstrip(".")
                    break

            final_json = json.dumps({"url": url, "title": title, "word_count": word_count})
            response_body = {
                "id": "msg_turn2",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": final_json}],
                "model": body.get("model", "mock"),
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 40},
            }

        payload = json.dumps(response_body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


# ─── Main test ───────────────────────────────────────────────────────────────

async def main() -> None:
    # Set up logging to capture semaphore instrumentation
    log_records: list[logging.LogRecord] = []

    class RecordHandler(logging.Handler):
        def emit(self, record: logging.LogRecord):
            log_records.append(record)

    scheduler_logger = logging.getLogger("batch_agent.scheduler")
    scheduler_logger.setLevel(logging.INFO)
    handler = RecordHandler()
    scheduler_logger.addHandler(handler)

    # Also print to stdout for visibility
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("  [LOG] %(message)s"))
    scheduler_logger.addHandler(stream_handler)

    # Start mock web server
    web_server = HTTPServer(("127.0.0.1", 19284), MockWebServer)
    web_thread = threading.Thread(target=web_server.serve_forever, daemon=True)
    web_thread.start()

    # Start mock Anthropic server
    anthropic_server = HTTPServer(("127.0.0.1", 19285), MockAnthropicMultiTurn)
    anthropic_thread = threading.Thread(target=anthropic_server.serve_forever, daemon=True)
    anthropic_thread.start()
    time.sleep(0.1)

    print(f"[{time.strftime('%H:%M:%S')}] Starting 10-agent multi-turn test (max_concurrent=3)...")
    print(f"  Each agent: turn 1 = tool_use(http_get), turn 2 = final JSON")
    print()
    start = time.monotonic()

    spec = BatchSpec(
        task='Fetch and summarize the web page at http://127.0.0.1:19284/page/{index}. Return JSON with url, title, word_count.',
        inputs=[{"index": i} for i in range(10)],
        output_schema=PageSummary,
        model="claude-sonnet-4-20250514",
        backend="anthropic://",
        max_concurrent=3,  # Low concurrency to make semaphore contention visible
        max_turns=5,
        max_retries=1,
        timeout_per_agent=15,
        tools=[Tool.http_get],
    )

    backend = AnthropicBackend(api_key="test-key", base_url="http://127.0.0.1:19285")
    plan = TaskCompiler().compile(spec)
    scheduler = WaveScheduler(plan, backend)

    received = 0
    ok_count = 0
    async for result in scheduler.stream():
        received += 1
        elapsed = time.monotonic() - start
        if result.ok:
            ok_count += 1
            print(f"  [{elapsed:5.2f}s] job={result.job_id:6s} OK: title={result.output.title!r} words={result.output.word_count}")
        else:
            print(f"  [{elapsed:5.2f}s] job={result.job_id:6s} FAILED: {result.error.type}: {result.error.message}")

    total = time.monotonic() - start
    print(f"\n[{time.strftime('%H:%M:%S')}] Done. {ok_count}/{received} OK in {total:.2f}s")

    # ─── Verify semaphore behavior ───────────────────────────────────────────
    print("\n--- Semaphore Instrumentation Analysis ---")
    acquire_msgs = [r for r in log_records if "semaphore acquired" in r.getMessage()]
    release_msgs = [r for r in log_records if "semaphore released" in r.getMessage()]
    tool_wait_msgs = [r for r in log_records if "TOOL_WAIT" in r.getMessage()]

    print(f"  Semaphore acquires: {len(acquire_msgs)}")
    print(f"  Semaphore releases: {len(release_msgs)}")
    print(f"  TOOL_WAIT events:   {len(tool_wait_msgs)}")

    # Each agent should have 2 acquires (turn 1 + turn 2) and 2 releases
    assert len(acquire_msgs) == 20, f"Expected 20 acquires (10 agents x 2 turns), got {len(acquire_msgs)}"
    assert len(release_msgs) == 20, f"Expected 20 releases, got {len(release_msgs)}"
    assert len(tool_wait_msgs) == 10, f"Expected 10 TOOL_WAIT events (1 per agent), got {len(tool_wait_msgs)}"

    # Verify that acquires and releases alternate correctly (release before next acquire for same job)
    # This proves the semaphore is freed during tool wait
    per_job_events: dict[str, list[str]] = {}
    for r in log_records:
        msg = r.getMessage()
        if "semaphore" in msg or "TOOL_WAIT" in msg:
            # Extract job_id from "[job-X]"
            job_id = msg.split("]")[0].split("[")[1]
            if "acquired" in msg:
                per_job_events.setdefault(job_id, []).append("ACQUIRE")
            elif "released" in msg:
                per_job_events.setdefault(job_id, []).append("RELEASE")
            elif "TOOL_WAIT" in msg:
                per_job_events.setdefault(job_id, []).append("TOOL_WAIT")

    print(f"\n  Per-job event sequences:")
    all_correct = True
    for job_id in sorted(per_job_events.keys(), key=lambda x: int(x.split("-")[1])):
        events = per_job_events[job_id]
        expected = ["ACQUIRE", "RELEASE", "TOOL_WAIT", "ACQUIRE", "RELEASE"]
        ok = events == expected
        if not ok:
            all_correct = False
        status = "OK" if ok else "WRONG"
        print(f"    {job_id}: {' -> '.join(events)} [{status}]")

    if all_correct:
        print("\n  [PASS] All jobs show correct ACQUIRE -> RELEASE -> TOOL_WAIT -> ACQUIRE -> RELEASE pattern")
        print("  [PASS] Semaphore is provably released during tool execution (W5 satisfied)")
    else:
        print("\n  [FAIL] Some jobs have incorrect event ordering!")
        sys.exit(1)

    # Verify multi-turn: all agents did 2 turns
    for state in scheduler.states.all():
        assert state.turn >= 2, f"{state.job_id} only did {state.turn} turn(s)"
    print(f"  [PASS] All agents completed >= 2 turns")

    web_server.shutdown()
    anthropic_server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
