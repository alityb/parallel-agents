from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx

from . import BackendAdapter, BackendResponse
from batch_agent.spec import AgentJob, Message, SharedContext


class OpenAIBackend(BackendAdapter):
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openai.com") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_url(cls, url: str) -> "OpenAIBackend":
        parsed = urlparse(url)
        base_url = f"https://{parsed.netloc}" if parsed.netloc else "https://api.openai.com"
        return cls(base_url=base_url)

    async def generate(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> BackendResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai:// backend")

        api_messages: list[dict[str, Any]] = []
        if shared.prefix:
            api_messages.append({"role": "system", "content": shared.prefix})

        if messages is not None:
            for msg in messages:
                api_messages.append({"role": msg.role, "content": msg.content})
        else:
            api_messages.append({"role": "user", "content": job.prompt})

        payload: dict[str, Any] = {"model": model, "messages": api_messages}
        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
        raw = response.json()
        return BackendResponse(
            content=raw["choices"][0]["message"].get("content", ""),
            raw=raw,
            stop_reason=raw["choices"][0].get("finish_reason", ""),
        )
