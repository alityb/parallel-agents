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
from batch_agent.backends.vllm import VLLMBackend
from batch_agent.spec import AgentJob, Message, SharedContext
from batch_agent.tools.pool import ToolPool


DEFAULT_OUTPUT = "tests/benchmarks/results/definitive_benchmark/results.json"


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
    # Use a simple single-token filler for common BPE tokenizers. Avoid long
    # hyphenated filler strings: Qwen tokenizes those into many tokens and can
    # accidentally exceed vLLM's max_model_len.
    repeat = max(1, target_tokens - 64)
    return base + ("x " * repeat)


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


class LiveVLLMDeterministicBackend(BackendAdapter):
    """Runs real vLLM forward passes while keeping tool flow deterministic.

    Turn 1 performs a real vLLM request, then returns a synthetic tool call.
    Turn 2 performs a real vLLM request, then returns deterministic JSON.
    This isolates scheduling, prefix caching, and tool waiting from model-specific
    tool-call parser variability.
    """

    def __init__(self, backend_url: str, model: str, max_tokens: int = 8) -> None:
        self.vllm = VLLMBackend.from_url(backend_url)
        self.model = model
        self.max_tokens = max_tokens
        self.request_count = 0
        self.request_latencies: list[float] = []
        self.usage_records: list[dict[str, Any]] = []

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        return await self.vllm.warm_prefix(shared, model)

    async def get_cache_metrics(self) -> dict[str, float]:
        return await self.vllm.get_cache_metrics()

    async def get_queue_metrics(self) -> dict[str, Any]:
        return await self.vllm.get_queue_metrics()

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
        await self._real_forward(shared, job, messages, timeout)
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
        body = json.dumps(
            {
                "company": company,
                "summary": f"{company} benchmark report using shared vLLM prefix and slow tool result.",
                "source_count": 3,
            }
        )
        return BackendResponse(
            content=body,
            raw={"usage": self.usage_records[-1] if self.usage_records else {}},
            stop_reason="end_turn",
        )

    async def _real_forward(
        self,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None,
        timeout: float | None,
    ) -> None:
        user_text = job.prompt
        if messages:
            tail = messages[-1].content
            user_text = tail[:2000] if tail else job.prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": shared.prefix},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout or 120) as client:
            response = await client.post(
                f"{self.vllm.base_url}/v1/chat/completions",
                json=payload,
                headers={"authorization": f"Bearer {self.vllm.api_key}"},
            )
            response.raise_for_status()
        elapsed = time.monotonic() - started
        self.request_count += 1
        self.request_latencies.append(elapsed)
        raw = response.json()
        if isinstance(raw.get("usage"), dict):
            self.usage_records.append(raw["usage"])


async def slow_lookup(query: str, latency_seconds: float, counter: dict[str, int]) -> str:
    counter["executed"] += 1
    await asyncio.sleep(latency_seconds)
    return f"lookup result for {query}: source one; source two; source three"


async def run_naive(
    *,
    backend: LiveVLLMDeterministicBackend,
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
            # Intentional naive baseline: semaphore covers the whole agent loop,
            # so slow tools occupy inference slots.
            async with sem:
                messages = [Message(role="user", content=prompt)]
                first = await backend.generate(
                    shared=shared,
                    job=job,
                    messages=messages,
                    model=backend.model,
                    timeout=120,
                )
                if first.tool_calls:
                    result = await slow_lookup(
                        first.tool_calls[0].args["query"],
                        tool_latency_seconds,
                        tool_counter,
                    )
                    messages.append(
                        Message(
                            role="tool_result",
                            content=json.dumps(
                                [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": first.tool_calls[0].id,
                                        "content": result,
                                    }
                                ]
                            ),
                        )
                    )
                second = await backend.generate(
                    shared=shared,
                    job=job,
                    messages=messages,
                    model=backend.model,
                    timeout=120,
                )
            return True, CompanyReport.model_validate_json(second.content).model_dump()
        except Exception as exc:
            return False, repr(exc)

    raw = await asyncio.gather(*[one(i, c) for i, c in enumerate(companies)])
    wall = time.monotonic() - started
    ok = sum(1 for success, _ in raw if success)
    failures = [value for success, value in raw if not success]
    return {
        "mode": "naive_asyncio_gather",
        "ok": ok,
        "failed": len(companies) - ok,
        "failure_samples": failures[:5],
        "wall_clock_seconds": wall,
        "throughput_agents_per_sec": len(companies) / wall,
        "tool_calls_requested": len(companies),
        "tool_calls_executed": tool_counter["executed"],
        "vllm_requests": backend.request_count,
        "vllm_request_latency_p50": percentile(backend.request_latencies, 0.50),
        "vllm_request_latency_p95": percentile(backend.request_latencies, 0.95),
        "usage": usage_totals(backend.usage_records),
    }


async def run_batchagent(
    *,
    backend: LiveVLLMDeterministicBackend,
    shared_prompt: str,
    companies: list[str],
    max_inflight: int,
    tool_latency_seconds: float,
) -> dict[str, Any]:
    counter = {"executed": 0}

    @Tool.define(name="slow_company_lookup", max_tokens=1000, cacheable=True, rate_limit=1000)
    async def slow_company_lookup(query: str) -> str:
        return await slow_lookup(query, tool_latency_seconds, counter)

    started = time.monotonic()
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
        backend="mock://deterministic-live-vllm",
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
    wall = time.monotonic() - started
    ok = sum(1 for r in results if r.ok)
    failures = [str(r.error) for r in results if not r.ok]
    return {
        "mode": "batchagent",
        "ok": ok,
        "failed": len(results) - ok,
        "failure_samples": failures[:5],
        "wall_clock_seconds": wall,
        "throughput_agents_per_sec": len(results) / wall,
        "tool_calls_requested": len(companies),
        "tool_calls_executed": counter["executed"],
        "vllm_requests": backend.request_count,
        "vllm_request_latency_p50": percentile(backend.request_latencies, 0.50),
        "vllm_request_latency_p95": percentile(backend.request_latencies, 0.95),
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


async def discover_model(backend_url: str) -> str:
    base = VLLMBackend.from_url(backend_url).base_url
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{base}/v1/models")
        response.raise_for_status()
    data = response.json()
    return data["data"][0]["id"]


async def scrape_metrics(backend_url: str) -> dict[str, float]:
    return await VLLMBackend.from_url(backend_url).get_cache_metrics()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="vllm://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--slow-tools", action="store_true")
    parser.add_argument("--tool-latency", type=float, default=800, help="milliseconds")
    parser.add_argument("--system-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = await discover_model(args.backend)
    shared_prompt = build_shared_prompt(args.system_prompt_tokens)
    shared = SharedContext(prefix=shared_prompt)
    companies = company_names(args.n)
    tool_latency_seconds = (args.tool_latency / 1000.0) if args.slow_tools else 0.0

    before = await scrape_metrics(args.backend)
    naive_backend = LiveVLLMDeterministicBackend(args.backend, model)
    naive = await run_naive(
        backend=naive_backend,
        shared=shared,
        companies=companies,
        max_inflight=args.max_inflight,
        tool_latency_seconds=tool_latency_seconds,
    )
    mid = await scrape_metrics(args.backend)
    batch_backend = LiveVLLMDeterministicBackend(args.backend, model)

    original_scheduler = BatchAgent._scheduler

    def scheduler_with_live_backend(spec: Any) -> Any:
        from batch_agent.compiler import TaskCompiler
        from batch_agent.scheduler import WaveScheduler

        return WaveScheduler(TaskCompiler().compile(spec), batch_backend, tool_pool=ToolPool())

    BatchAgent._scheduler = staticmethod(scheduler_with_live_backend)  # type: ignore[method-assign]
    try:
        batchagent = await run_batchagent(
            backend=batch_backend,
            shared_prompt=shared_prompt,
            companies=companies,
            max_inflight=args.max_inflight,
            tool_latency_seconds=tool_latency_seconds,
        )
    finally:
        BatchAgent._scheduler = original_scheduler  # type: ignore[method-assign]

    after = await scrape_metrics(args.backend)
    result = {
        "benchmark": "definitive_slow_tool_vllm",
        "backend": args.backend,
        "model": model,
        "n": args.n,
        "system_prompt_tokens_target": args.system_prompt_tokens,
        "tool_latency_ms": args.tool_latency if args.slow_tools else 0,
        "max_inflight": args.max_inflight,
        "metrics_before": before,
        "metrics_after_naive": mid,
        "metrics_after_batchagent": after,
        "naive": naive,
        "batchagent": batchagent,
        "deltas": {
            "wall_clock_seconds_batchagent_minus_naive": (
                batchagent["wall_clock_seconds"] - naive["wall_clock_seconds"]
            ),
            "tool_calls_saved_by_batchagent": (
                naive["tool_calls_executed"] - batchagent["tool_calls_executed"]
            ),
            "total_tokens_batchagent_minus_naive": (
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
