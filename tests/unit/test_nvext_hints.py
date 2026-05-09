from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

from batch_agent.backends.dynamo import DynamoBackend, _build_nvext_hints
from batch_agent.spec import AgentJob, SharedContext


def test_build_nvext_hints_none_without_steps_to_execution() -> None:
    job = SimpleNamespace(state=SimpleNamespace(steps_to_execution=None))

    assert _build_nvext_hints(job) is None


def test_build_nvext_hints_latency_sensitivity_increases_for_lower_eta() -> None:
    slow = SimpleNamespace(state=SimpleNamespace(steps_to_execution=2.0, max_turns=6, turn=2, kv_key=None))
    fast = SimpleNamespace(state=SimpleNamespace(steps_to_execution=0.5, max_turns=6, turn=2, kv_key=None))

    assert _build_nvext_hints(fast)["nvext"]["agent_hints"]["latency_sensitivity"] > (
        _build_nvext_hints(slow)["nvext"]["agent_hints"]["latency_sensitivity"]
    )


def test_build_nvext_hints_speculative_prefill_tracks_kv_key() -> None:
    cold = SimpleNamespace(state=SimpleNamespace(steps_to_execution=1.0, max_turns=3, turn=1, kv_key=None))
    warm = SimpleNamespace(state=SimpleNamespace(steps_to_execution=1.0, max_turns=3, turn=1, kv_key="abc"))

    assert _build_nvext_hints(cold)["nvext"]["agent_hints"]["speculative_prefill"] is False
    assert _build_nvext_hints(warm)["nvext"]["agent_hints"]["speculative_prefill"] is True


class _PayloadHandler(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        type(self).bodies.append(body)
        payload = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return None


def test_hints_attached_to_generate_payload_when_enabled() -> None:
    _PayloadHandler.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _PayloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = DynamoBackend(base_url=f"http://127.0.0.1:{server.server_port}")
        job = AgentJob(job_id="job-1", index=0, input_data={}, prompt="do", estimated_prompt_tokens=1)
        asyncio.run(backend.generate(
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
        ))

        hints = _PayloadHandler.bodies[0]["nvext"]["agent_hints"]
        assert hints["latency_sensitivity"] == 1.0
        assert hints["priority"] == 2
        assert hints["speculative_prefill"] is True
    finally:
        server.shutdown()


def test_hints_absent_when_disabled() -> None:
    _PayloadHandler.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _PayloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = DynamoBackend(base_url=f"http://127.0.0.1:{server.server_port}")
        job = AgentJob(job_id="job-1", index=0, input_data={}, prompt="do", estimated_prompt_tokens=1)
        asyncio.run(backend.generate(
            shared=SharedContext(prefix="system"),
            job=job,
            model="mock",
            metadata={"nvext_agent_hints": False, "steps_to_execution": 0.5},
        ))

        assert "nvext" not in _PayloadHandler.bodies[0]
    finally:
        server.shutdown()
