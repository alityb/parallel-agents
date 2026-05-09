"""Tests for checkpoint store and reduce topology."""
from __future__ import annotations

import asyncio
import json
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

from pydantic import BaseModel

import sys
sys.path.insert(0, ".")

from batch_agent import BatchAgent
from batch_agent.backends.anthropic import AnthropicBackend
from batch_agent.checkpoint import CheckpointStore
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentResult, AgentError, BatchSpec


class Item(BaseModel):
    value: int


class Summary(BaseModel):
    total: int
    count: int


# --- Checkpoint store tests ---

def test_checkpoint_store_saves_and_loads():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CheckpointStore(tmpdir)

        result = AgentResult(job_id="job-0", index=0, output={"value": 42}, attempts=1)
        store.save_result(result)

        completed = store.get_completed_job_ids()
        assert "job-0" in completed

        loaded = store.load_result("job-0")
        assert loaded is not None
        assert loaded.job_id == "job-0"
        assert loaded.output == {"value": 42}
        assert loaded.attempts == 1
        print("[PASS] Checkpoint store saves and loads results")

        store.close()


def test_checkpoint_store_skips_completed_on_rerun():
    """Simulates crash recovery: completed jobs are skipped on re-run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run: save some results
        store = CheckpointStore(tmpdir)
        store.save_result(AgentResult(job_id="job-0", index=0, output={"v": 1}, attempts=1))
        store.save_result(AgentResult(job_id="job-1", index=1, output={"v": 2}, attempts=1))
        store.close()

        # Second run: check what's completed
        store2 = CheckpointStore(tmpdir)
        completed = store2.get_completed_job_ids()
        assert completed == {"job-0", "job-1"}
        print("[PASS] Checkpoint store identifies previously completed jobs")
        store2.close()


# --- Reduce topology test ---

class MockReduceHandler(BaseHTTPRequestHandler):
    call_count = 0

    def do_POST(self):
        MockReduceHandler.call_count += 1
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        messages = body.get("messages", [])

        # Check if this is the reduce call (has "Results:" in the message)
        user_msg = messages[-1].get("content", "") if messages else ""
        if "Results:" in user_msg:
            # Parse results and compute total — items now have status/output wrapping
            try:
                results_start = user_msg.index("[")
                results_json = user_msg[results_start:]
                items = json.loads(results_json)
                total = sum(
                    item.get("output", {}).get("value", 0) if item.get("status") == "ok"
                    else 0
                    for item in items
                )
                count = len(items)
            except Exception:
                total = 0
                count = 0
            resp_text = json.dumps({"total": total, "count": count})
        else:
            # Regular agent call — just return a value
            idx = MockReduceHandler.call_count
            resp_text = json.dumps({"value": idx * 10})

        resp = {
            "id": f"msg_{MockReduceHandler.call_count}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": resp_text}],
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


def test_reduce_topology():
    MockReduceHandler.call_count = 0
    server = HTTPServer(("127.0.0.1", 19290), MockReduceHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.1)

    async def run():
        # Monkey-patch the backend URL for this test
        import batch_agent
        original_from_url = batch_agent.backend_from_url

        def patched_from_url(url):
            if url == "anthropic://":
                return AnthropicBackend(api_key="test", base_url="http://127.0.0.1:19290")
            return original_from_url(url)

        batch_agent.backend_from_url = patched_from_url
        try:
            results, summary = await BatchAgent.run_with_reduce(
                task="Generate a value for item {idx}",
                inputs=[{"idx": i} for i in range(5)],
                output_schema=Item,
                model="mock",
                backend="anthropic://",
                max_concurrent=5,
                reduce="You have received {n} items. Sum their values and return total and count.",
                reduce_schema=Summary,
            )
        finally:
            batch_agent.backend_from_url = original_from_url

        assert len(results) == 5
        assert all(r.ok for r in results)
        assert summary.count == 5
        assert summary.total == sum(r.output.value for r in results)
        print(f"[PASS] Reduce topology: {len(results)} results -> summary(total={summary.total}, count={summary.count})")

    asyncio.run(run())
    server.shutdown()


if __name__ == "__main__":
    test_checkpoint_store_saves_and_loads()
    test_checkpoint_store_skips_completed_on_rerun()
    test_reduce_topology()
    print("\n[ALL PASS]")
