from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

from pydantic import BaseModel

from batch_agent.backends.bedrock import BedrockBackend
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import BatchSpec


MODEL = os.getenv("BEDROCK_LIVE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


class NumberMessage(BaseModel):
    n: int
    msg: str


def _usage_cost_usd(usage: dict) -> float | None:
    # Bedrock returns token usage, not dollar cost. Avoid hardcoding model pricing here;
    # exact USD should come from AWS Pricing/CUR for the model+region actually used.
    return None


async def main() -> None:
    backend = BedrockBackend(region=REGION)
    spec = BatchSpec(
        task="Return JSON with keys n (random 1-100) and msg (one sentence about that number). Input: {index}",
        inputs=[{"index": i} for i in range(20)],
        output_schema=NumberMessage,
        model=MODEL,
        backend=f"bedrock://{REGION}/{MODEL}",
        max_concurrent=5,
        max_turns=1,
        max_retries=1,
        timeout_per_agent=60,
        timeout_per_turn=60,
    )
    scheduler = WaveScheduler(TaskCompiler().compile(spec), backend)
    started = time.monotonic()
    results = []
    async for result in scheduler.stream():
        elapsed = time.monotonic() - started
        print(f"[{elapsed:6.2f}s] {result.job_id} ok={result.ok} output={result.output}")
        results.append(result)

    metrics = backend.request_metrics
    ttfts = [m.get("ttft_seconds") for m in metrics if m.get("ttft_seconds") is not None]
    usages = [m.get("usage", {}) for m in metrics]
    costs = [_usage_cost_usd(u) for u in usages]
    costs = [c for c in costs if c is not None]
    cache_usage_keys = sorted({k for u in usages for k in u.keys() if "cache" in k.lower()})
    cache_point_requested = any(m.get("cachePointRequested") for m in metrics)
    payload = {
        "model": MODEL,
        "region": REGION,
        "n": 20,
        "ok": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok),
        "wall_clock_seconds": time.monotonic() - started,
        "ttft_seconds": ttfts,
        "ttft_p50_seconds": statistics.median(ttfts) if ttfts else None,
        "ttft_p95_seconds": sorted(ttfts)[int(len(ttfts) * 0.95) - 1] if ttfts else None,
        "usage": usages,
        "total_input_tokens": sum(u.get("inputTokens", 0) for u in usages),
        "total_output_tokens": sum(u.get("outputTokens", 0) for u in usages),
        "total_cache_read_input_tokens": sum(u.get("cacheReadInputTokens", 0) for u in usages),
        "total_cache_write_input_tokens": sum(u.get("cacheWriteInputTokens", 0) for u in usages),
        "cache_usage_keys": cache_usage_keys,
        "cachePoint_requested": cache_point_requested,
        "cachePoint_tokens_visible": bool(cache_usage_keys),
        "estimated_cost_usd": sum(costs) if costs else None,
        "cost_source": "Bedrock response usage fields include tokens but no dollar cost; exact USD requires AWS Pricing/CUR.",
    }
    out = Path("tests/benchmarks/results/bedrock_live_batch/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
