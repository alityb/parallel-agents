from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ..backends import BackendAdapter, BackendResponse, backend_from_url
from ..spec import AgentJob, Message, SharedContext
from ..utils import prefix_hash, strip_preamble_headers


@dataclass
class OpenCodeRuntime(BackendAdapter):
    """Run OpenCode CLI sessions against one shared OpenAI-compatible backend.

    Each BatchAgent worker launches ``opencode run`` with
    ``OPENCODE_CONFIG_CONTENT`` pointing at the same local SGLang/vLLM server.
    The existing BatchAgent scheduler still controls concurrency, prefix
    warming, retry, and result streaming.
    """

    backend: str
    model: str
    opencode_bin: str = "opencode"
    model_alias: str | None = None
    context_limit: int = 32768
    output_limit: int = 4096
    temperature: bool = True
    tool_call: bool = True
    working_dir: str = "."
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = self.backend.rstrip("/")
        self.model_alias = self.model_alias or _default_model_alias(self.model)
        self._provider = _provider_name(self.backend)
        self._base_url = _openai_base_url(self.backend)
        self._inner_backend = backend_from_url(self.backend)

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        """Warm the shared backend prefix once before OpenCode workers launch."""
        try:
            return await self._inner_backend.warm_prefix(shared, self.model)
        except Exception:
            prefix = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            return prefix_hash(prefix) if prefix else None

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
        prompt = self._build_prompt(shared, job, messages)
        env = os.environ.copy()
        env.update(self.extra_env)
        env["OPENCODE_CONFIG_CONTENT"] = self.build_config_content()

        try:
            proc = await asyncio.create_subprocess_exec(
                self.opencode_bin,
                "run",
                "--model",
                f"{self._provider}/{self.model_alias}",
                "--format",
                "json",
                "--dangerously-skip-permissions",
                prompt,
                cwd=self.working_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "opencode CLI not found. Install OpenCode and ensure `opencode` is on PATH."
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")
        events, content = parse_opencode_jsonl(stdout_text)

        if proc.returncode != 0:
            detail = (stderr_text or stdout_text).strip()
            raise RuntimeError(f"opencode run failed with exit code {proc.returncode}: {detail[:1000]}")

        return BackendResponse(
            content=content,
            raw={"events": events, "stdout": stdout_text, "stderr": stderr_text},
            stop_reason="end_turn",
        )

    async def generate_streaming(self, **kwargs: Any) -> BackendResponse:
        # OpenCode emits JSONL, but the CLI subprocess completes as one process.
        # Returning through generate keeps scheduler behavior deterministic.
        return await self.generate(**{k: v for k, v in kwargs.items() if k != "tool_queue"})

    async def get_cache_metrics(self) -> dict[str, float]:
        return await self._inner_backend.get_cache_metrics()

    async def get_queue_metrics(self) -> dict[str, Any]:
        return await self._inner_backend.get_queue_metrics()

    async def send_prefetch_hints(self, hints: list[Any]) -> None:
        await self._inner_backend.send_prefetch_hints(hints)

    def backend_capabilities(self) -> dict[str, Any]:
        capabilities = dict(self._inner_backend.backend_capabilities())
        capabilities["runtime"] = "opencode"
        return capabilities

    def build_config_content(self) -> str:
        provider_display = "SGLang" if self._provider == "sglang" else "vLLM"
        config = {
            "provider": {
                self._provider: {
                    "name": provider_display,
                    "npm": "@ai-sdk/openai-compatible",
                    "env": [],
                    "models": {
                        self.model_alias: {
                            "id": self.model,
                            "name": _display_model_name(self.model),
                            "tool_call": self.tool_call,
                            "temperature": self.temperature,
                            "limit": {
                                "context": self.context_limit,
                                "output": self.output_limit,
                            },
                            "cost": {"input": 0, "output": 0},
                        }
                    },
                    "options": {
                        "apiKey": "fake",
                        "baseURL": self._base_url,
                        "includeUsage": False,
                    },
                }
            }
        }
        return json.dumps(config, separators=(",", ":"))

    def _build_prompt(
        self,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None,
    ) -> str:
        parts: list[str] = []
        if shared.prefix:
            prefix = strip_preamble_headers(shared.prefix) if shared.strip_preamble else shared.prefix
            parts.append(prefix)
        if messages:
            for message in messages:
                if message.role == "user":
                    parts.append(message.content)
                else:
                    parts.append(f"{message.role}: {message.content}")
        else:
            parts.append(job.prompt)
        return "\n\n".join(part for part in parts if part)


def parse_opencode_jsonl(stdout: str) -> tuple[list[dict[str, Any]], str]:
    events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    fallback_parts: list[str] = []

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            fallback_parts.append(stripped)
            continue
        if isinstance(event, dict):
            events.append(event)
            if event.get("type") == "text":
                text = _event_text(event)
                if text:
                    text_parts.append(text)
            else:
                fallback = _fallback_event_text(event)
                if fallback:
                    fallback_parts.append(fallback)

    if text_parts:
        return events, "".join(text_parts)
    if fallback_parts:
        return events, "\n".join(fallback_parts)
    return events, stdout


def _event_text(event: dict[str, Any]) -> str:
    for key in ("text", "content", "message", "delta"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


def _fallback_event_text(event: dict[str, Any]) -> str:
    for key in ("result", "content", "message", "text"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


def _provider_name(backend: str) -> str:
    if backend.startswith("sglang://"):
        return "sglang"
    if backend.startswith("vllm://"):
        return "vllm"
    parsed = urlparse(backend)
    if parsed.scheme in {"sglang", "vllm"}:
        return parsed.scheme
    raise ValueError("OpenCodeRuntime backend must start with sglang:// or vllm://")


def _openai_base_url(backend: str) -> str:
    parsed = urlparse(backend)
    if parsed.scheme in {"sglang", "vllm"}:
        base = f"http://{parsed.netloc}"
    else:
        base = backend.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _default_model_alias(model: str) -> str:
    leaf = model.rsplit("/", 1)[-1].lower()
    match = re.search(r"qwen(?:[.\-_\d]*)?-(\d+)b", leaf)
    if "qwen" in leaf and match:
        return f"qwen{match.group(1)}b"
    cleaned = re.sub(r"[^a-z0-9]+", "-", leaf).strip("-")
    return cleaned[:48] or "model"


def _display_model_name(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace("-", " ")
