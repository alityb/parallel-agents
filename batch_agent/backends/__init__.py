from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from batch_agent.spec import AgentJob, SharedContext


@dataclass(frozen=True)
class BackendResponse:
    content: str
    raw: Any | None = None
    tool_calls: list[Any] = field(default_factory=list)


class BackendAdapter(ABC):
    @abstractmethod
    async def generate(self, *, shared: SharedContext, job: AgentJob, model: str, timeout: float | None = None) -> BackendResponse:
        raise NotImplementedError

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        return None


def backend_from_url(url: str) -> BackendAdapter:
    if url.startswith("anthropic://"):
        from .anthropic import AnthropicBackend

        return AnthropicBackend()
    if url.startswith("openai://"):
        from .openai import OpenAIBackend

        return OpenAIBackend.from_url(url)
    if url.startswith("vllm://"):
        from .vllm import VLLMBackend

        return VLLMBackend.from_url(url)
    if url.startswith("sglang://"):
        from .sglang import SGLangBackend

        return SGLangBackend.from_url(url)
    raise ValueError(f"unsupported backend URL: {url}")
