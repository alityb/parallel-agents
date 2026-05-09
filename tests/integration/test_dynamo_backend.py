from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from batch_agent.backends import StreamingToolCall
from batch_agent.backends.dynamo import DynamoBackend
from batch_agent.spec import AgentJob, SharedContext


class _DynamoSSEHandler(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        type(self).bodies.append(body)
        payload = (
            "event: tool_call_dispatch\n"
            'data: {"type":"tool_call_dispatch","tool_call":{"id":"call_dyn","name":"read_file","args":{"path":"README.md"}}}\n\n'
            'data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return None


def test_dynamo_streaming_tool_call_dispatch_events_are_parsed() -> None:
    async def run() -> None:
        _DynamoSSEHandler.bodies = []
        server = HTTPServer(("127.0.0.1", 0), _DynamoSSEHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = DynamoBackend(base_url=f"http://127.0.0.1:{server.server_port}")
            queue: asyncio.Queue[StreamingToolCall] = asyncio.Queue()
            job = AgentJob(job_id="job-1", index=0, input_data={}, prompt="do", estimated_prompt_tokens=1)

            response = await backend.generate_streaming(
                shared=SharedContext(prefix="system"),
                job=job,
                model="mock",
                metadata={
                    "nvext_agent_hints": True,
                    "steps_to_execution": 0.5,
                    "turn": 1,
                    "max_turns": 3,
                    "kv_key": "abc",
                },
                tool_queue=queue,
            )
            first = await queue.get()

            assert _DynamoSSEHandler.bodies[0]["nvext"]["agent_hints"]["speculative_prefill"] is True
            assert response.tool_calls[0].id == "call_dyn"
            assert response.tool_calls[0].name == "read_file"
            assert response.tool_calls[0].args == {"path": "README.md"}
            assert first.tool_call == response.tool_calls[0]
        finally:
            server.shutdown()

    asyncio.run(run())
