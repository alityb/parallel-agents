from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ..spec import AgentJob, Message, SharedContext, ToolCall
from ..utils import NO_API_KEY, PREFIX_WARM_TIMEOUT, prefix_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedToolCall:
    """A tool call parsed from a model response, possibly malformed."""

    id: str
    name: str
    args: dict[str, Any]
    error: bool = False
    error_message: str = ""

    def to_tool_call(self) -> ToolCall:
        return ToolCall(name=self.name, args=self.args)


@dataclass(frozen=True)
class BackendResponse:
    content: str
    raw: Any | None = None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def is_final(self) -> bool:
        """True if the model stopped naturally (not requesting tool use)."""
        return self.stop_reason != "tool_use" and not self.tool_calls


class BackendAdapter(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> BackendResponse:
        raise NotImplementedError

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        return None

    async def get_cache_metrics(self) -> dict[str, float]:
        """Return cache/utilization metrics for adaptive concurrency.

        Keys: ``prefix_cache_hit_rate`` [0.0, 1.0], ``gpu_utilization`` [0.0, 1.0].
        Default: empty dict (no metrics — API-mode adapters).
        """
        return {}

    async def get_queue_metrics(self) -> dict[str, Any]:
        """Return backend queue depth for backpressure dispatch.

        Keys: ``requests_waiting`` int, ``requests_running`` int.
        Default: {} (no queue depth visible — dispatch proceeds freely).
        """
        return {}

    async def send_prefetch_hints(self, hints: list[Any]) -> None:
        """Send KVFlow prefetch hints. API/managed backends no-op by default."""
        return None

    def backend_capabilities(self) -> dict[str, Any]:
        return {
            "prefix_pinning": False,
            "kvflow": False,
            "diff_kv": False,
            "max_safe_concurrent": 1,
        }


# ── URL parsing helper shared by vLLM / SGLang / OpenAI adapters ──────────────

def _http_url_from_scheme(url: str, scheme: str) -> str:
    """Convert ``scheme://host:port`` to ``http://host:port``.

    E.g. ``vllm://localhost:8000`` → ``http://localhost:8000``.
    Falls back to ``https://netloc`` for ``openai://`` style URLs.
    """
    parsed = urlparse(url)
    if parsed.scheme == scheme:
        return f"http://{parsed.netloc}"
    return f"{parsed.scheme}://{parsed.netloc}"


# ── Backend factory ────────────────────────────────────────────────────────────

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
    if url.startswith("bedrock://"):
        from .bedrock import BedrockBackend
        return BedrockBackend.from_url(url)
    raise ValueError(f"unsupported backend URL: {url}")
