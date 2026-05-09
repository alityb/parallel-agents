from __future__ import annotations

from batch_agent.backends.anthropic import AnthropicBackend
from batch_agent.backends.bedrock import BedrockBackend, BedrockConcurrencyController
from batch_agent.backends.sglang import SGLangBackend
from batch_agent.backends.vllm import VLLMBackend


def test_backend_capabilities_declared() -> None:
    assert BedrockBackend().backend_capabilities() == {
        "prefix_pinning": False,
        "prompt_cache_token_savings": True,
        "prompt_cache_latency_benefit": False,
        "kvflow": False,
        "diff_kv": False,
        "max_safe_concurrent": 1,
    }
    assert AnthropicBackend(api_key="x").backend_capabilities()["kvflow"] is False
    assert VLLMBackend(api_key="EMPTY", base_url="http://localhost:8000").backend_capabilities()["prefix_pinning"] is True
    assert SGLangBackend(api_key="EMPTY", base_url="http://localhost:30000").backend_capabilities()["kvflow"] is True


def test_bedrock_concurrency_controller_aimd_halves_and_recovers() -> None:
    now = {"t": 0.0}
    controller = BedrockConcurrencyController(
        max_concurrent_ceiling=8,
        current_limit=4,
        clock=lambda: now["t"],
    )
    assert controller.record_throttle() == 2
    assert controller.current_limit == 2

    now["t"] = 59.0
    assert controller.maybe_increase() == 2

    now["t"] = 60.0
    assert controller.maybe_increase() == 3
    assert controller.current_limit == 3
