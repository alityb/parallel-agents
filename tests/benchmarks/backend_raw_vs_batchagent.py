from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batch_agent import BatchAgent, Tool
from batch_agent.backends import BackendAdapter, BackendResponse, ParsedToolCall
from batch_agent.spec import AgentJob, Message, SharedContext
from batch_agent.tools.pool import ToolPool


DEFAULT_OUTPUT = "tests/benchmarks/results/backend_raw_vs_batchagent/results.json"


class CompanyReport(BaseModel):
    company: str
    summary: str
    source_count: int


def build_shared_prompt(target_tokens: int) -> str:
    base = (
        "You are a concise analyst for a multi-agent inference benchmark. "
        "Every answer must be a compact JSON object with company, summary, "
        "and source_count fields. "
    )
    return base + ("x " * max(1, target_tokens - 64))


def company_names(n: int) -> list[str]:
    base = [
        "NVIDIA",
        "OpenAI",
        "Anthropic",
        "vLLM",
        "SGLang",
        "Meta AI",
        "Google DeepMind",
        "AWS Bedrock",
        "Together AI",
        "LMCache",
    ]
    return [base[i % len(base)] for i in range(n)]


class LiveOpenAICompatDeterministicBackend(BackendAdapter):
    """Real OpenAI-compatible forwards with deterministic tool/output behavior."""

    def __init__(self, base_url: str, model: str, max_tokens: int = 8) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.request_count = 0
        self.request_latencies: list[float] = []
        self.ttft_latencies: list[float] = []
        self.usage_records: list[dict[str, Any]] = []

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        if not shared.prefix:
            return None
        try:
            await self._post_chat(
                [
                    {"role": "system", "content": shared.prefix},
                    {"role": "user", "content": "ping"},
                ],
                timeout=60,
                max_tokens=1,
            )
        except Exception:
            return None
        return None

    async def get_cache_metrics(self) -> dict[str, float]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/metrics")
            if response.status_code != 200:
                return {}
        except Exception:
            return {}
        text = response.text
        return {
            "prefix_cache_hit_rate": _metric_hit_rate(text),
            "gpu_utilization": _metric_gpu_usage(text),
        }

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
        user_text = job.prompt
        if messages:
            user_text = messages[-1].content[:2000] if messages[-1].content else job.prompt
        await self._post_chat(
            [
                {"role": "system", "content": shared.prefix},
                {"role": "user", "content": user_text},
            ],
            timeout=timeout or 120,
            max_tokens=self.max_tokens,
            metadata=metadata,
        )
        has_tool_result = any(m.role == "tool_result" for m in (messages or []))
        company = str(job.input_data.get("company", f"company-{job.index}"))
        query = str(job.input_data.get("query", "shared-query"))
        if not has_tool_result:
            return BackendResponse(
                content="",
                raw={"usage": self.usage_records[-1] if self.usage_records else {}},
                tool_calls=[
                    ParsedToolCall(
                        id=f"tool-{job.job_id}",
                        name="slow_company_lookup",
                        args={"query": query},
                    )
                ],
                stop_reason="tool_use",
            )
        return BackendResponse(
            content=json.dumps({
                "company": company,
                "summary": f"{company} benchmark report using shared prefix and slow tool result.",
                "source_count": 3,
            }),
            raw={"usage": self.usage_records[-1] if self.usage_records else {}},
            stop_reason="end_turn",
        )

    async def _post_chat(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float,
        max_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        if metadata:
            extensions = metadata.get("request_extensions")
            if isinstance(extensions, dict):
                payload.update(extensions)
        started = time.monotonic()
        ttft: float | None = None
        raw: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raw = response.json()
                else:
                    content_parts: list[str] = []
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        event = json.loads(data)
                        choice = (event.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        if ttft is None and delta.get("content"):
                            ttft = time.monotonic() - started
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        if isinstance(event.get("usage"), dict):
                            raw["usage"] = event["usage"]
                    raw.setdefault(
                        "choices",
                        [{"message": {"role": "assistant", "content": "".join(content_parts)}}],
                    )
        elapsed = time.monotonic() - started
        self.request_count += 1
        self.request_latencies.append(elapsed)
        if ttft is not None:
            self.ttft_latencies.append(ttft)
        if isinstance(raw.get("usage"), dict):
            self.usage_records.append(raw["usage"])
        return raw


async def slow_lookup(query: str, latency_seconds: float, counter: dict[str, int]) -> str:
    counter["executed"] += 1
    await asyncio.sleep(latency_seconds)
    return f"lookup result for {query}: source one; source two; source three"


async def run_naive(
    *,
    backend: LiveOpenAICompatDeterministicBackend,
    shared: SharedContext,
    companies: list[str],
    max_inflight: int,
    tool_latency_seconds: float,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(max_inflight)
    tool_counter = {"executed": 0}
    started = time.monotonic()

    async def one(index: int, company: str) -> tuple[bool, str | dict[str, Any]]:
        query = f"kv-cache-shared-query-{index % 10}"
        prompt = f"Company: {company}. Query: {query}."
        job = AgentJob(
            job_id=f"naive-{index}",
            index=index,
            input_data={"company": company, "query": query},
            prompt=prompt,
            estimated_prompt_tokens=len(prompt.split()),
            max_turns=2,
        )
        try:
            # Baseline user code usually holds its concurrency slot across the
            # whole agent loop, including the slow tool wait.
            async with sem:
                messages = [Message(role="user", content=prompt)]
                first = await backend.generate(shared=shared, job=job, messages=messages, model=backend.model)
                if first.tool_calls:
                    result = await slow_lookup(
                        first.tool_calls[0].args["query"],
                        tool_latency_seconds,
                        tool_counter,
                    )
                    messages.append(Message(
                        role="tool_result",
                        content=json.dumps([{
                            "type": "tool_result",
                            "tool_use_id": first.tool_calls[0].id,
                            "content": result,
                        }]),
                    ))
                second = await backend.generate(shared=shared, job=job, messages=messages, model=backend.model)
            return True, CompanyReport.model_validate_json(second.content).model_dump()
        except Exception as exc:
            return False, repr(exc)

    raw = await asyncio.gather(*[one(i, c) for i, c in enumerate(companies)])
    wall = time.monotonic() - started
    ok = sum(1 for success, _ in raw if success)
    failures = [value for success, value in raw if not success]
    return _result_summary("raw_endpoint_loop", len(companies), ok, failures, wall, tool_counter, backend)


async def run_batchagent(
    *,
    backend: LiveOpenAICompatDeterministicBackend,
    shared_prompt: str,
    companies: list[str],
    max_inflight: int,
    tool_latency_seconds: float,
) -> dict[str, Any]:
    counter = {"executed": 0}

    @Tool.define(name="slow_company_lookup", max_tokens=1000, cacheable=True, rate_limit=1000)
    async def slow_company_lookup(query: str) -> str:
        return await slow_lookup(query, tool_latency_seconds, counter)

    original_scheduler = BatchAgent._scheduler

    def scheduler_with_live_backend(spec: Any) -> Any:
        from batch_agent.compiler import TaskCompiler
        from batch_agent.scheduler import WaveScheduler

        return WaveScheduler(TaskCompiler().compile(spec), backend, tool_pool=ToolPool())

    BatchAgent._scheduler = staticmethod(scheduler_with_live_backend)  # type: ignore[method-assign]
    started = time.monotonic()
    try:
        results = await BatchAgent.run(
            task="Company: {company}. Query: {query}.",
            inputs=[
                {"company": company, "query": f"kv-cache-shared-query-{i % 10}"}
                for i, company in enumerate(companies)
            ],
            system_prompt=shared_prompt,
            tools=[Tool.registry["slow_company_lookup"]],
            output_schema=CompanyReport,
            model=backend.model,
            backend="mock://deterministic-openai-compatible",
            max_concurrent=max_inflight,
            max_inflight=max_inflight,
            max_dispatched=-1,
            max_turns=2,
            max_retries=0,
            timeout_per_turn=120,
            timeout_per_tool=max(5, tool_latency_seconds + 2),
            kvflow=False,
            streaming_tool_dispatch=False,
            no_hoist=True,
        )
    finally:
        BatchAgent._scheduler = original_scheduler  # type: ignore[method-assign]
    wall = time.monotonic() - started
    ok = sum(1 for r in results if r.ok)
    failures = [str(r.error) for r in results if not r.ok]
    return _result_summary("batchagent", len(results), ok, failures, wall, counter, backend)


def _result_summary(
    mode: str,
    n: int,
    ok: int,
    failures: list[Any],
    wall: float,
    tool_counter: dict[str, int],
    backend: LiveOpenAICompatDeterministicBackend,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "ok": ok,
        "failed": n - ok,
        "failure_samples": failures[:5],
        "wall_clock_seconds": wall,
        "throughput_agents_per_sec": n / wall,
        "tool_calls_requested": n,
        "tool_calls_executed": tool_counter["executed"],
        "backend_requests": backend.request_count,
        "backend_request_latency_p50": percentile(backend.request_latencies, 0.50),
        "backend_request_latency_p95": percentile(backend.request_latencies, 0.95),
        "ttft_p50": percentile(backend.ttft_latencies, 0.50),
        "ttft_p95": percentile(backend.ttft_latencies, 0.95),
        "ttft_p99": percentile(backend.ttft_latencies, 0.99),
        "ttft_samples": len(backend.ttft_latencies),
        "usage": usage_totals(backend.usage_records),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    def get(record: dict[str, Any], key: str) -> int:
        value = record.get(key)
        return value if isinstance(value, int) else 0

    return {
        "prompt_tokens": sum(get(r, "prompt_tokens") for r in records),
        "completion_tokens": sum(get(r, "completion_tokens") for r in records),
        "total_tokens": sum(get(r, "total_tokens") for r in records),
    }


async def discover_model(base_url: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{base_url.rstrip('/')}/v1/models")
        response.raise_for_status()
    data = response.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models reported by {base_url}/v1/models")
    return models[0]["id"]


def _metric_hit_rate(text: str) -> float | None:
    values: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        name = parts[0].split("{", 1)[0]
        values[name] = value
        if "prefix_cache_hit_rate" in name or name.endswith("cache_hit_rate"):
            return value
    queries = next((v for k, v in values.items() if "prefix_cache_queries_total" in k), None)
    hits = next((v for k, v in values.items() if "prefix_cache_hits_total" in k), None)
    if queries and hits is not None:
        return hits / queries
    return next((v for k, v in values.items() if "cache_hit" in k.lower()), None)


def _metric_gpu_usage(text: str) -> float | None:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        if "gpu_cache_usage" in name or "token_usage" in name:
            try:
                return float(parts[-1])
            except ValueError:
                return None
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--tool-latency", type=float, default=800, help="milliseconds")
    parser.add_argument("--system-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = await discover_model(args.base_url)
    shared_prompt = build_shared_prompt(args.system_prompt_tokens)
    companies = company_names(args.n)
    tool_latency_seconds = args.tool_latency / 1000.0

    metrics_probe = LiveOpenAICompatDeterministicBackend(args.base_url, model)
    before = await metrics_probe.get_cache_metrics()

    naive_backend = LiveOpenAICompatDeterministicBackend(args.base_url, model)
    naive = await run_naive(
        backend=naive_backend,
        shared=SharedContext(prefix=shared_prompt),
        companies=companies,
        max_inflight=args.max_inflight,
        tool_latency_seconds=tool_latency_seconds,
    )
    mid = await metrics_probe.get_cache_metrics()

    batch_backend = LiveOpenAICompatDeterministicBackend(args.base_url, model)
    batchagent = await run_batchagent(
        backend=batch_backend,
        shared_prompt=shared_prompt,
        companies=companies,
        max_inflight=args.max_inflight,
        tool_latency_seconds=tool_latency_seconds,
    )
    after = await metrics_probe.get_cache_metrics()

    result = {
        "benchmark": "backend_raw_vs_batchagent",
        "label": args.label,
        "base_url": args.base_url,
        "model": model,
        "n": args.n,
        "system_prompt_tokens_target": args.system_prompt_tokens,
        "tool_latency_ms": args.tool_latency,
        "max_inflight": args.max_inflight,
        "metrics_before": before,
        "metrics_after_raw": mid,
        "metrics_after_batchagent": after,
        "raw_endpoint_loop": naive,
        "batchagent": batchagent,
        "deltas": {
            "wall_clock_seconds_batchagent_minus_raw": (
                batchagent["wall_clock_seconds"] - naive["wall_clock_seconds"]
            ),
            "tool_calls_saved_by_batchagent": (
                naive["tool_calls_executed"] - batchagent["tool_calls_executed"]
            ),
            "total_tokens_batchagent_minus_raw": (
                batchagent["usage"]["total_tokens"] - naive["usage"]["total_tokens"]
            ),
        },
        "timestamp": time.time(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
