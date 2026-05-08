from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import httpx

from .openai import OpenAIBackend
from batch_agent.spec import SharedContext


class VLLMBackend(OpenAIBackend):
    @classmethod
    def from_url(cls, url: str) -> "VLLMBackend":
        parsed = urlparse(url)
        scheme = "http" if parsed.scheme == "vllm" else parsed.scheme
        base_url = f"{scheme}://{parsed.netloc}"
        return cls(api_key="EMPTY", base_url=base_url)

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        if not shared.prefix:
            return None
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/v1/completions",
                json={"model": model, "prompt": shared.prefix, "max_tokens": 0},
                headers={"authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
        return hashlib.sha256(shared.prefix.encode("utf-8")).hexdigest()
