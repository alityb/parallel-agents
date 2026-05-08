"""Test OpenAI/vLLM tool call parsing with a mock server that returns tool_calls."""
from __future__ import annotations

import asyncio
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

import sys
sys.path.insert(0, ".")

from batch_agent.backends.openai import OpenAIBackend
from batch_agent.spec import AgentJob, SharedContext


class MockOpenAIToolHandler(BaseHTTPRequestHandler):
    call_count = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        messages = body.get("messages", [])
        MockOpenAIToolHandler.call_count += 1

        # Check if we have tool results already (turn 2)
        has_tool_msg = any(m.get("role") == "tool" for m in messages)

        if not has_tool_msg:
            # Turn 1: return tool_calls
            resp = {
                "id": "chatcmpl-1",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "http_get",
                                    "arguments": '{"url": "http://example.com"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
            }
        else:
            # Turn 2: final response
            resp = {
                "id": "chatcmpl-2",
                "choices": [{
                    "message": {"role": "assistant", "content": '{"summary": "Example page content"}'},
                    "finish_reason": "stop",
                }],
            }

        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


async def main() -> None:
    MockOpenAIToolHandler.call_count = 0
    server = HTTPServer(("127.0.0.1", 19287), MockOpenAIToolHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    backend = OpenAIBackend(api_key="test-key", base_url="http://127.0.0.1:19287")
    shared = SharedContext(prefix="You are helpful.")
    job = AgentJob(job_id="job-0", index=0, input_data={}, prompt="Summarize example.com", estimated_prompt_tokens=10)

    # Turn 1: should get tool_calls
    resp1 = await backend.generate(shared=shared, job=job, model="gpt-4", timeout=10)
    assert len(resp1.tool_calls) == 1, f"Expected 1 tool call, got {len(resp1.tool_calls)}"
    tc = resp1.tool_calls[0]
    assert tc.name == "http_get"
    assert tc.args == {"url": "http://example.com"}
    assert tc.id == "call_abc123"
    assert tc.error is False
    assert resp1.stop_reason == "tool_use"
    assert resp1.is_final is False
    print("[PASS] Turn 1: parsed tool_calls correctly from OpenAI format")

    # Turn 2: send tool result, get final
    from batch_agent.spec import Message
    msgs = [
        Message(role="user", content="Summarize example.com"),
        Message(role="assistant_raw", content=json.dumps([
            {"type": "text", "text": "Let me fetch that."},
            {"type": "tool_use", "id": "call_abc123", "name": "http_get", "input": {"url": "http://example.com"}},
        ])),
        Message(role="tool_result", content=json.dumps([
            {"type": "tool_result", "tool_use_id": "call_abc123", "content": "<html>Example</html>"},
        ])),
    ]
    resp2 = await backend.generate(shared=shared, job=job, messages=msgs, model="gpt-4", timeout=10)
    assert resp2.is_final is True
    assert resp2.tool_calls == []
    assert '"summary"' in resp2.content
    print("[PASS] Turn 2: final response parsed correctly")
    print(f"[PASS] OpenAI tool call parsing verified ({MockOpenAIToolHandler.call_count} API calls)")

    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
