"""Integration test: vLLM adapter against a mock OpenAI-compatible server.

Verifies:
1. warm_prefix sends POST /v1/completions with max_tokens=0
2. generate sends POST /v1/chat/completions with system + user messages
3. Tool calls in OpenAI format are handled (tested separately in test_openai_tools.py)
"""
from __future__ import annotations

import asyncio
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

import sys
sys.path.insert(0, ".")

from batch_agent.backends.vllm import VLLMBackend
from batch_agent.spec import AgentJob, SharedContext


class MockVLLMHandler(BaseHTTPRequestHandler):
    """Simulates a vLLM OpenAI-compatible server."""
    completions_calls: list = []
    chat_calls: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/v1/completions":
            # Prefix warming
            MockVLLMHandler.completions_calls.append(body)
            resp = {"id": "cmpl-1", "choices": [{"text": "", "finish_reason": "length"}]}
        elif self.path == "/v1/chat/completions":
            MockVLLMHandler.chat_calls.append(body)
            resp = {
                "id": "chatcmpl-1",
                "choices": [{"message": {"role": "assistant", "content": '{"result": "ok"}'}, "finish_reason": "stop"}],
            }
        else:
            self.send_response(404)
            self.end_headers()
            return

        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


async def main() -> None:
    MockVLLMHandler.completions_calls = []
    MockVLLMHandler.chat_calls = []

    server = HTTPServer(("127.0.0.1", 19286), MockVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    backend = VLLMBackend.from_url("vllm://127.0.0.1:19286")

    # Test warm_prefix
    shared = SharedContext(prefix="You are a helpful assistant for scientific analysis.")
    prefix_hash = await backend.warm_prefix(shared, model="meta-llama/Llama-3.1-70B-Instruct")

    assert prefix_hash is not None, "warm_prefix should return a hash"
    assert len(MockVLLMHandler.completions_calls) == 1
    warm_call = MockVLLMHandler.completions_calls[0]
    assert warm_call["prompt"] == shared.prefix
    assert warm_call["max_tokens"] == 0
    assert warm_call["model"] == "meta-llama/Llama-3.1-70B-Instruct"
    print("[PASS] warm_prefix sends correct /v1/completions request")

    # Test generate
    job = AgentJob(job_id="job-0", index=0, input_data={"x": "test"}, prompt="Analyze this.", estimated_prompt_tokens=50)
    response = await backend.generate(shared=shared, job=job, model="meta-llama/Llama-3.1-70B-Instruct", timeout=10)

    assert len(MockVLLMHandler.chat_calls) == 1
    chat_call = MockVLLMHandler.chat_calls[0]
    assert chat_call["model"] == "meta-llama/Llama-3.1-70B-Instruct"
    assert chat_call["messages"][0]["role"] == "system"
    assert chat_call["messages"][0]["content"] == shared.prefix
    assert chat_call["messages"][1]["role"] == "user"
    assert chat_call["messages"][1]["content"] == "Analyze this."
    assert response.content == '{"result": "ok"}'
    print("[PASS] generate sends correct /v1/chat/completions request")
    print("[PASS] vLLM adapter verified against mock server")

    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
