from __future__ import annotations

import asyncio

import httpx
import pytest

from batch_agent.backends.sglang import SGLangBackend
from batch_agent.spec import AgentJob, Message, SharedContext


def test_sglang_url_parsing() -> None:
    backend = SGLangBackend.from_url("sglang://localhost:30000")
    assert backend.base_url == "http://localhost:30000"


def test_sglang_native_generate_uses_backend_response(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"text": '{"value": 1}'}

    class FakeClient:
        def __init__(self, timeout=None) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    backend = SGLangBackend(base_url="http://sglang.test", use_native=True)
    response = asyncio.run(backend.generate(
        shared=SharedContext(prefix="system"),
        job=AgentJob(
            job_id="job-1",
            index=0,
            input_data={},
            prompt="Return JSON",
            estimated_prompt_tokens=10,
        ),
        messages=[Message(role="user", content="Return JSON")],
        model="mock",
    ))
    assert response.content == '{"value": 1}'
    assert response.is_final


def test_sglang_live_health_if_running() -> None:
    try:
        response = httpx.get("http://localhost:30000/health", timeout=0.5)
    except Exception:
        pytest.skip("SGLang server is not running")
    assert response.status_code == 200
