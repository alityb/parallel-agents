from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from batch_agent.backends.vllm import VLLMBackend
from batch_agent.spec import SharedContext


class _VLLMProbeHandler(BaseHTTPRequestHandler):
    gpu_cache_usage_values: list[float] = []
    chat_calls = 0

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        idx = min(len(type(self).gpu_cache_usage_values) - 1, max(0, type(self).chat_calls - 1))
        value = type(self).gpu_cache_usage_values[idx]
        payload = f"vllm:gpu_cache_usage_perc {value}\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            type(self).chat_calls += 1
            body = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
        elif self.path == "/internal/pin_blocks":
            body = {"ok": True}
        else:
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return None


def test_warm_prefix_warns_when_same_prefix_increases_gpu_cache_usage(caplog) -> None:
    _VLLMProbeHandler.gpu_cache_usage_values = [0.10, 0.25]
    _VLLMProbeHandler.chat_calls = 0
    server = HTTPServer(("127.0.0.1", 0), _VLLMProbeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = VLLMBackend(
            api_key="EMPTY",
            base_url=f"http://127.0.0.1:{server.server_port}",
            block_sharing_probe_agents=3,
            block_sharing_usage_tolerance=0.001,
        )
        with caplog.at_level(logging.WARNING, logger="batch_agent.backends.vllm"):
            asyncio.run(backend.warm_prefix(SharedContext(prefix="shared prefix"), model="mock"))
        assert "prefix block sharing may not be working" in caplog.text
        assert _VLLMProbeHandler.chat_calls == 4
    finally:
        server.shutdown()
