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
from batch_agent.backends import BackendResponse, ParsedToolCall, StreamingToolCall
from batch_agent.backends.dynamo import DynamoBackend
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, Message, SharedContext
from batch_agent.tools.pool import ToolPool
from batch_agent.utils import prefix_hash


DEFAULT_OUTPUT = "tests/benchmarks/results/research_summary/dynamo_h100_results.json"


PAPERS: list[dict[str, str]] = [
    {"id": "1706.03762", "title": "Attention Is All You Need", "topic": "transformer attention"},
    {"id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "topic": "transformer attention"},
    {"id": "2005.14165", "title": "Language Models are Few-Shot Learners", "topic": "foundation models"},
    {"id": "2203.02155", "title": "Training language models to follow instructions", "topic": "foundation models"},
    {"id": "2307.08691", "title": "FlashAttention-2", "topic": "attention kernels"},
    {"id": "2205.14135", "title": "FlashAttention", "topic": "attention kernels"},
    {"id": "2309.06180", "title": "Efficient Memory Management for LLM Serving with PagedAttention", "topic": "kv cache serving"},
    {"id": "2402.14837", "title": "SGLang: Efficient Execution of Structured Language Model Programs", "topic": "kv cache serving"},
    {"id": "2507.07400", "title": "KVFlow", "topic": "workflow kv cache"},
    {"id": "2510.09665", "title": "LMCache", "topic": "workflow kv cache"},
    {"id": "2604.03143", "title": "TokenDance", "topic": "multi-agent kv cache"},
    {"id": "2401.08671", "title": "S-LoRA", "topic": "serving systems"},
    {"id": "2308.16369", "title": "Sarathi-Serve", "topic": "serving systems"},
    {"id": "2405.04434", "title": "DistServe", "topic": "serving systems"},
    {"id": "2312.07104", "title": "DeepSpeed-FastGen", "topic": "serving systems"},
    {"id": "2406.14086", "title": "RouteLLM", "topic": "inference routing"},
    {"id": "2406.04692", "title": "Speculative Streaming", "topic": "inference routing"},
    {"id": "2302.01318", "title": "Toolformer", "topic": "tool-using agents"},
    {"id": "2210.03629", "title": "ReAct", "topic": "tool-using agents"},
    {"id": "2303.11366", "title": "Reflexion", "topic": "tool-using agents"},
]


class ResearchSummary(BaseModel):
    paper_id: str
    title: str
    topic: str
    contribution: str
    relevance_to_batchagent: str
    turns: int
    sources_used: int


def build_shared_prompt(target_tokens: int) -> str:
    base = (
        "You are a systems research analyst benchmarking multi-turn agents. "
        "Each agent summarizes one arXiv paper for an LLM inference and "
        "multi-agent orchestration survey. Use concise, concrete language. "
        "Final answers must be JSON matching the requested schema. "
    )
    return base + ("shared-prefix-token " * max(1, target_tokens - 80))


def task_template() -> str:
    return (
        "Paper ID: {paper_id}\n"
        "Title: {title}\n"
        "Topic: {topic}\n\n"
        "Task: use web_search for recent context on the paper's topic, then "
        "produce a compact research summary with contribution and relevance "
        "to BatchAgent."
    )


def paper_inputs() -> list[dict[str, str]]:
    return [{"paper_id": p["id"], "title": p["title"], "topic": p["topic"]} for p in PAPERS]


class MeasuredResearchDynamoBackend(DynamoBackend):
    """OpenAI-compatible benchmark adapter with deterministic tool flow.

    Each turn still performs a real backend forward pass so TTFT and provider
    cache accounting come from the live backend. The model output is not trusted
    for tool-call formatting or JSON validity; this keeps A and C comparable.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_tokens: int = 8,
        label: str = "",
        enable_nvext: bool = False,
    ) -> None:
        super().__init__(base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.label = label
        self.enable_nvext = enable_nvext
        self.request_count = 0
        self.request_latencies: list[float] = []
        self.ttft_latencies: list[float] = []
        self.ttft_by_turn: dict[int, list[float]] = {}
        self.usage_records: list[dict[str, Any]] = []

    async def warm_prefix(self, shared: SharedContext, model: str) -> str | None:
        if not shared.prefix:
            return None
        metadata = None
        if self.enable_nvext:
            metadata = {
                "request_extensions": {
                    "nvext": {
                        "agent_hints": {
                            "latency_sensitivity": 0.2,
                            "priority": 0,
                            "speculative_prefill": False,
                            "osl": 1,
                        }
                    }
                }
            }
        await self._post_chat(
            [
                {"role": "system", "content": shared.prefix},
                {"role": "user", "content": "warm the shared research-summary prefix"},
            ],
            timeout=120,
            max_tokens=1,
            metadata=metadata,
            turn=0,
        )
        return prefix_hash(shared.prefix)

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
        return await self._generate_research(
            shared=shared,
            job=job,
            messages=messages,
            metadata=metadata,
            timeout=timeout,
        )

    async def generate_streaming(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        tool_queue: Any | None = None,
    ) -> BackendResponse:
        response = await self._generate_research(
            shared=shared,
            job=job,
            messages=messages,
            metadata=metadata,
            timeout=timeout,
        )
        if tool_queue is not None:
            for tool_call in response.tool_calls:
                await tool_queue.put(StreamingToolCall(tool_call=tool_call))
            await tool_queue.put(StreamingToolCall(is_final=True))
        return response

    async def _generate_research(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None,
        metadata: dict[str, Any] | None,
        timeout: float | None,
    ) -> BackendResponse:
        user_text = job.prompt
        if messages:
            tail = messages[-1].content
            user_text = tail[:3000] if tail else job.prompt

        turn = int((metadata or {}).get("turn") or (2 if _has_tool_result(messages) else 1))
        request_extensions = self._request_extensions(metadata)
        await self._post_chat(
            [
                {"role": "system", "content": shared.prefix},
                {"role": "user", "content": user_text},
            ],
            timeout=timeout or 180,
            max_tokens=self.max_tokens,
            metadata={"request_extensions": request_extensions} if request_extensions else None,
            turn=turn,
        )

        paper_id = str(job.input_data["paper_id"])
        title = str(job.input_data["title"])
        topic = str(job.input_data["topic"])
        if not _has_tool_result(messages):
            query = _search_query(topic)
            return BackendResponse(
                content="",
                raw={"usage": self.usage_records[-1] if self.usage_records else {}},
                tool_calls=[
                    ParsedToolCall(
                        id=f"search-{job.job_id}",
                        name="web_search",
                        args={"query": query, "num_results": 5},
                    )
                ],
                stop_reason="tool_use",
            )

        content = ResearchSummary(
            paper_id=paper_id,
            title=title,
            topic=topic,
            contribution=f"{title} is summarized as a {topic} paper with systems relevance.",
            relevance_to_batchagent=(
                f"It informs BatchAgent's {topic} path by clarifying where batching, "
                "tool scheduling, and cache reuse change throughput."
            ),
            turns=2,
            sources_used=3,
        ).model_dump_json()
        return BackendResponse(
            content=content,
            raw={"usage": self.usage_records[-1] if self.usage_records else {}},
            stop_reason="end_turn",
        )

    def _request_extensions(self, metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        if not self.enable_nvext:
            return None
        if not metadata or not metadata.get("nvext_agent_hints"):
            return None
        steps = metadata.get("steps_to_execution")
        if steps is None:
            steps = 0.5
        return {
            "nvext": {
                "agent_hints": {
                    "latency_sensitivity": min(1.0, 1.0 / max(0.1, float(steps))),
                    "priority": int(metadata.get("max_turns", 2)) - int(metadata.get("turn", 1)),
                    "speculative_prefill": metadata.get("kv_key") is not None,
                    "osl": 256,
                }
            }
        }

    async def _post_chat(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float,
        max_tokens: int,
        metadata: dict[str, Any] | None = None,
        turn: int,
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
                    raw = json.loads((await response.aread()).decode())
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
            self.ttft_by_turn.setdefault(turn, []).append(ttft)
        if isinstance(raw.get("usage"), dict):
            self.usage_records.append(raw["usage"])
        return raw


def _has_tool_result(messages: list[Message] | None) -> bool:
    return any(m.role == "tool_result" for m in (messages or []))


def _search_query(topic: str) -> str:
    return f"systems research context for {topic}"


def _result_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    def get(record: dict[str, Any], key: str) -> int:
        value = record.get(key)
        return value if isinstance(value, int) else 0

    cached = 0
    for record in records:
        details = record.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            cached += details["cached_tokens"]
    prompt = sum(get(r, "prompt_tokens") for r in records)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": sum(get(r, "completion_tokens") for r in records),
        "total_tokens": sum(get(r, "total_tokens") for r in records),
        "cached_prompt_tokens": cached,
        "prefix_cache_hit_rate": (cached / prompt) if prompt else None,
    }


async def _discover_model(base_url: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{base_url.rstrip('/')}/v1/models")
        response.raise_for_status()
    models = (response.json().get("data") or [])
    if not models:
        raise RuntimeError(f"No models reported by {base_url}/v1/models")
    return models[0]["id"]


async def _get_prefix_cache_hit_rate(base_url: str) -> float | None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{base_url.rstrip('/')}/metrics")
        if response.status_code != 200:
            return None
    except Exception:
        return None

    values: dict[str, float] = {}
    for raw in response.text.splitlines():
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
    return None


async def _web_search_impl(query: str, num_results: int, latency_seconds: float, counter: dict[str, int]) -> str:
    counter["executed"] += 1
    await asyncio.sleep(latency_seconds)
    return (
        f"Search query: {query}\n"
        f"Result 1: Recent systems papers discuss {query} in relation to batching and KV reuse.\n"
        f"Result 2: Benchmarks emphasize TTFT, prefix-cache hit rate, and tool wait overlap.\n"
        f"Result 3: Production deployments compare raw serving endpoints with orchestrated multi-turn agents.\n"
    )


async def run_condition_a(
    *,
    backend: MeasuredResearchDynamoBackend,
    shared: SharedContext,
    inputs: list[dict[str, str]],
    max_inflight: int,
    tool_latency_seconds: float,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(max_inflight)
    tool_counter = {"executed": 0}
    started = time.monotonic()

    async def one(index: int, data: dict[str, str]) -> tuple[bool, Any]:
        prompt = task_template().format(**data)
        job = AgentJob(
            job_id=f"condition-a-{index}",
            index=index,
            input_data=data,
            prompt=prompt,
            estimated_prompt_tokens=len(prompt.split()),
            max_turns=2,
        )
        try:
            async with sem:
                messages = [Message(role="user", content=prompt)]
                first = await backend.generate(
                    shared=shared,
                    job=job,
                    messages=messages,
                    model=backend.model,
                    metadata={"turn": 1},
                    timeout=180,
                )
                if first.tool_calls:
                    tc = first.tool_calls[0]
                    result = await _web_search_impl(
                        tc.args["query"],
                        int(tc.args.get("num_results", 5)),
                        tool_latency_seconds,
                        tool_counter,
                    )
                    messages.append(Message(
                        role="tool_result",
                        content=json.dumps([{
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result,
                        }]),
                    ))
                second = await backend.generate(
                    shared=shared,
                    job=job,
                    messages=messages,
                    model=backend.model,
                    metadata={"turn": 2},
                    timeout=180,
                )
            return True, ResearchSummary.model_validate_json(second.content).model_dump()
        except Exception as exc:
            return False, repr(exc)

    raw = await asyncio.gather(*[one(i, data) for i, data in enumerate(inputs)])
    wall = time.monotonic() - started
    ok = sum(1 for success, _ in raw if success)
    failures = [value for success, value in raw if not success]
    return _summary(
        condition="A",
        mode="raw_endpoint_loop",
        n=len(inputs),
        ok=ok,
        failures=failures,
        wall=wall,
        tool_calls_executed=tool_counter["executed"],
        backend=backend,
    )


async def run_condition_c(
    *,
    backend: MeasuredResearchDynamoBackend,
    backend_url: str,
    enable_nvext: bool,
    shared_prompt: str,
    inputs: list[dict[str, str]],
    max_inflight: int,
    tool_latency_seconds: float,
) -> dict[str, Any]:
    tool_counter = {"executed": 0}

    @Tool.define(name="web_search", max_tokens=1200, cacheable=True, rate_limit=1000)
    async def benchmark_web_search(query: str, num_results: int = 5) -> str:
        return await _web_search_impl(query, num_results, tool_latency_seconds, tool_counter)

    original_scheduler = BatchAgent._scheduler

    def scheduler_with_live_backend(spec: Any, _backend: Any = None) -> WaveScheduler:
        return WaveScheduler(TaskCompiler().compile(spec), backend, tool_pool=ToolPool(cache_ttl=300))

    BatchAgent._scheduler = staticmethod(scheduler_with_live_backend)  # type: ignore[method-assign]
    started = time.monotonic()
    try:
        results = await BatchAgent.run(
            task=task_template(),
            inputs=inputs,
            system_prompt=shared_prompt,
            tools=[Tool.registry["web_search"]],
            output_schema=ResearchSummary,
            model=backend.model,
            backend=backend_url,
            max_concurrent=max_inflight,
            max_inflight=max_inflight,
            max_dispatched=-1,
            max_turns=2,
            max_retries=0,
            timeout_per_turn=180,
            timeout_per_tool=max(10, tool_latency_seconds + 5),
            kvflow=False,
            streaming_tool_dispatch=False,
            nvext_agent_hints=enable_nvext,
            no_hoist=True,
        )
    finally:
        BatchAgent._scheduler = original_scheduler  # type: ignore[method-assign]
    wall = time.monotonic() - started
    ok = sum(1 for r in results if r.ok)
    failures = [str(r.error) for r in results if not r.ok]
    return _summary(
        condition="C",
        mode="batchagent",
        n=len(inputs),
        ok=ok,
        failures=failures,
        wall=wall,
        tool_calls_executed=tool_counter["executed"],
        backend=backend,
    )


def _summary(
    *,
    condition: str,
    mode: str,
    n: int,
    ok: int,
    failures: list[Any],
    wall: float,
    tool_calls_executed: int,
    backend: MeasuredResearchDynamoBackend,
) -> dict[str, Any]:
    usage = _usage_totals(backend.usage_records)
    turns_per_agent = 2.0 if ok else None
    return {
        "condition": condition,
        "mode": mode,
        "ok": ok,
        "failed": n - ok,
        "failure_samples": failures[:5],
        "wall_clock_seconds": wall,
        "throughput_agents_per_sec": n / wall,
        "turns_per_agent_mean": turns_per_agent,
        "tool_calls_requested": n,
        "web_search_calls_executed": tool_calls_executed,
        "web_search_dedup_ratio": (n / tool_calls_executed) if tool_calls_executed else None,
        "backend_requests": backend.request_count,
        "request_latency_p50": _result_percentile(backend.request_latencies, 0.50),
        "request_latency_p95": _result_percentile(backend.request_latencies, 0.95),
        "ttft_all_p50": _result_percentile(backend.ttft_latencies, 0.50),
        "ttft_all_p95": _result_percentile(backend.ttft_latencies, 0.95),
        "ttft_turn1_p50": _result_percentile(backend.ttft_by_turn.get(1, []), 0.50),
        "ttft_turn1_p95": _result_percentile(backend.ttft_by_turn.get(1, []), 0.95),
        "ttft_turn2_p50": _result_percentile(backend.ttft_by_turn.get(2, []), 0.50),
        "ttft_turn2_p95": _result_percentile(backend.ttft_by_turn.get(2, []), 0.95),
        "usage": usage,
        "prefix_cache_hit_rate": usage["prefix_cache_hit_rate"],
    }


async def main() -> None:
    parser = argparse.ArgumentParser("bench_research_summary")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--backend", default="dynamo://localhost:8001")
    parser.add_argument("--label", default="")
    parser.add_argument("--nvext-agent-hints", action="store_true")
    parser.add_argument("--system-prompt-tokens", type=int, default=2048)
    parser.add_argument("--tool-latency", type=float, default=400, help="milliseconds")
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = await _discover_model(args.base_url)
    shared_prompt = build_shared_prompt(args.system_prompt_tokens)
    inputs = paper_inputs()
    tool_latency_seconds = args.tool_latency / 1000.0

    metrics_before = await _get_prefix_cache_hit_rate(args.base_url)

    cold_backend = MeasuredResearchDynamoBackend(
        args.base_url,
        model,
        label="condition-a",
        enable_nvext=args.nvext_agent_hints,
    )
    condition_a = await run_condition_a(
        backend=cold_backend,
        shared=SharedContext(prefix=shared_prompt),
        inputs=inputs,
        max_inflight=args.max_inflight,
        tool_latency_seconds=tool_latency_seconds,
    )
    metrics_after_a = await _get_prefix_cache_hit_rate(args.base_url)

    warm_backend = MeasuredResearchDynamoBackend(
        args.base_url,
        model,
        label="condition-c",
        enable_nvext=args.nvext_agent_hints,
    )
    condition_c = await run_condition_c(
        backend=warm_backend,
        backend_url=args.backend,
        enable_nvext=args.nvext_agent_hints,
        shared_prompt=shared_prompt,
        inputs=inputs,
        max_inflight=args.max_inflight,
        tool_latency_seconds=tool_latency_seconds,
    )
    metrics_after_c = await _get_prefix_cache_hit_rate(args.base_url)

    if condition_a["prefix_cache_hit_rate"] is None:
        condition_a["prefix_cache_hit_rate"] = metrics_after_a
        condition_a["usage"]["prefix_cache_hit_rate"] = metrics_after_a
    if condition_c["prefix_cache_hit_rate"] is None:
        condition_c["prefix_cache_hit_rate"] = metrics_after_c
        condition_c["usage"]["prefix_cache_hit_rate"] = metrics_after_c

    result = {
        "benchmark": "research_summary",
        "label": args.label,
        "backend": args.backend,
        "base_url": args.base_url,
        "model": model,
        "paper_ids": [p["id"] for p in PAPERS],
        "task": task_template(),
        "n": len(inputs),
        "system_prompt_tokens_target": args.system_prompt_tokens,
        "tool_latency_ms": args.tool_latency,
        "max_inflight": args.max_inflight,
        "conditions": {
            "A": condition_a,
            "C": condition_c,
        },
        "metrics": {
            "prefix_cache_hit_rate_before": metrics_before,
            "prefix_cache_hit_rate_after_a": metrics_after_a,
            "prefix_cache_hit_rate_after_c": metrics_after_c,
        },
        "comparison": {
            "wall_clock_seconds_c_minus_a": (
                condition_c["wall_clock_seconds"] - condition_a["wall_clock_seconds"]
            ),
            "speedup_a_over_c": (
                condition_a["wall_clock_seconds"] / condition_c["wall_clock_seconds"]
                if condition_c["wall_clock_seconds"]
                else None
            ),
            "web_search_calls_saved": (
                condition_a["web_search_calls_executed"] - condition_c["web_search_calls_executed"]
            ),
            "ttft_cold_turn1_p50": condition_a["ttft_turn1_p50"],
            "ttft_warm_turn1_p50": condition_c["ttft_turn1_p50"],
            "prefix_cache_hit_rate_a": condition_a["prefix_cache_hit_rate"],
            "prefix_cache_hit_rate_c": condition_c["prefix_cache_hit_rate"],
        },
        "notes": [
            "A is a raw endpoint loop that holds its concurrency slot during web_search.",
            "C is BatchAgent using the same OpenAI-compatible backend forwards and ToolPool coalescing.",
            "web_search is a deterministic 400ms research-context lookup grouped by topic; no external search key is required.",
            "TTFT and prefix-cache accounting come from live backend forwards on every turn.",
        ],
        "timestamp": time.time(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
