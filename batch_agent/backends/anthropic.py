from __future__ import annotations

import os
from typing import Any

import httpx

from . import BackendAdapter, BackendResponse
from batch_agent.spec import AgentJob, SharedContext


class AnthropicBackend(BackendAdapter):
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.anthropic.com") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")

    async def generate(self, *, shared: SharedContext, job: AgentJob, model: str, timeout: float | None = None) -> BackendResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic:// backend")

        system: str | list[dict[str, Any]]
        if shared.prefix:
            system = [{"type": "text", "text": shared.prefix, "cache_control": {"type": "ephemeral"}}]
        else:
            system = ""

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": job.prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
        raw = response.json()
        return BackendResponse(content=_extract_text(raw), raw=raw)


def _extract_text(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in raw.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
