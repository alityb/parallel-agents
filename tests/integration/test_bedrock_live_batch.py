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
from batch_agent.repair import parse_and_validate_output
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentResult, BatchSpec


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
    backend = BedrockBackend(
        region=REGION,
        max_concurrent_ceiling=int(os.getenv("BEDROCK_LIVE_MAX_CONCURRENT_CEILING", "5")),
    )
    pad_tokens = int(os.getenv("BEDROCK_CACHE_PAD_TOKENS", "0"))
    system_prompt = ""
    if pad_tokens:
        # Approx 1 token per short word for this diagnostic prompt.
        system_prompt = " ".join(["cacheable-prefix-token"] * pad_tokens)
    spec = BatchSpec(
        system_prompt=system_prompt,
        task="Return JSON with keys n (random 1-100) and msg (one sentence about that number). Input: {index}",
        inputs=[{"index": i} for i in range(20)],
        output_schema=NumberMessage,
        model=MODEL,
        backend=f"bedrock://{REGION}/{MODEL}",
        # Bedrock concurrency is controlled by BedrockConcurrencyController (AIMD):
        # starts at 1, increases after quiet windows, halves on throttling.
        max_concurrent=backend.concurrency_controller.current_limit,
        max_turns=1,
        max_retries=int(os.getenv("BEDROCK_LIVE_MAX_RETRIES", "3")),
        timeout_per_agent=float(os.getenv("BEDROCK_LIVE_TIMEOUT", "180")),
        timeout_per_turn=float(os.getenv("BEDROCK_LIVE_TIMEOUT", "180")),
    )
    scheduler = WaveScheduler(TaskCompiler().compile(spec), backend)
    started = time.monotonic()
    results = []
    async for result in scheduler.stream():
        elapsed = time.monotonic() - started
        if result.ok:
            print(f"[{elapsed:6.2f}s] {result.job_id} ok=True output={result.output}")
        else:
            print(f"[{elapsed:6.2f}s] {result.job_id} ok=False error={result.error}")
        results.append(result)

    # Optional diagnostic: direct raw calls for failed inputs, so we can print raw
    # model responses for parse/schema failures rather than only scheduler errors.
    raw_failure_samples = []
    failed_indices = [r.index for r in results if not r.ok][:3]
    for idx in failed_indices:
        job = TaskCompiler().compile(BatchSpec(
            task=spec.task,
            inputs=[{"index": idx}],
            output_schema=NumberMessage,
            model=MODEL,
            backend=spec.backend,
            max_concurrent=1,
            max_turns=1,
        )).jobs[0]
        try:
            response = await backend.generate(
                shared=TaskCompiler().compile(spec).shared,
                job=job,
                model=MODEL,
                timeout=spec.timeout_per_turn,
            )
            parsed = parse_and_validate_output(response.content, NumberMessage)
            raw_failure_samples.append({
                "index": idx,
                "raw_response": response.content,
                "parse_status": "ok_on_diagnostic_rerun",
                "parsed": parsed.model_dump(),
            })
        except Exception as exc:
            raw_failure_samples.append({
                "index": idx,
                "raw_response": locals().get("response").content if "response" in locals() else None,
                "parse_status": "failed_on_diagnostic_rerun",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            })

    metrics = backend.request_metrics
    first_payload = backend.request_payloads[0] if backend.request_payloads else {}
    ttfts = [m.get("ttft_seconds") for m in metrics if m.get("ttft_seconds") is not None]
    usages = [m.get("usage", {}) for m in metrics]
    cache_miss_ttfts = [
        m.get("ttft_seconds") for m in metrics
        if m.get("ttft_seconds") is not None and m.get("usage", {}).get("cacheWriteInputTokens", 0) > 0
    ]
    cache_hit_ttfts = [
        m.get("ttft_seconds") for m in metrics
        if m.get("ttft_seconds") is not None and m.get("usage", {}).get("cacheReadInputTokens", 0) > 0
    ]
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
        "ttft_cache_miss_seconds": cache_miss_ttfts,
        "ttft_cache_hit_seconds": cache_hit_ttfts,
        "ttft_cache_miss_p50_seconds": statistics.median(cache_miss_ttfts) if cache_miss_ttfts else None,
        "ttft_cache_hit_p50_seconds": statistics.median(cache_hit_ttfts) if cache_hit_ttfts else None,
        "ttft_cache_miss_to_hit_ratio": (
            statistics.median(cache_miss_ttfts) / statistics.median(cache_hit_ttfts)
            if cache_miss_ttfts and cache_hit_ttfts else None
        ),
        "ttft_cache_hit_to_miss_ratio": (
            statistics.median(cache_hit_ttfts) / statistics.median(cache_miss_ttfts)
            if cache_miss_ttfts and cache_hit_ttfts else None
        ),
        "usage": usages,
        "failures": [
            {
                "job_id": r.job_id,
                "index": r.index,
                "error_type": None if r.error is None else r.error.type,
                "error": None if r.error is None else r.error.message,
            }
            for r in results if not r.ok
        ],
        "raw_failure_samples": raw_failure_samples,
        "total_input_tokens": sum(u.get("inputTokens", 0) for u in usages),
        "total_output_tokens": sum(u.get("outputTokens", 0) for u in usages),
        "total_cache_read_input_tokens": sum(u.get("cacheReadInputTokens", 0) for u in usages),
        "total_cache_write_input_tokens": sum(u.get("cacheWriteInputTokens", 0) for u in usages),
        "cache_usage_keys": cache_usage_keys,
        "cachePoint_requested": cache_point_requested,
        "cachePoint_tokens_visible": bool(cache_usage_keys),
        "system_prompt_char_count": len(system_prompt),
        "system_prompt_estimated_tokens": max(0, len(system_prompt.split())),
        "request_body_sample": first_payload,
        "estimated_cost_usd": sum(costs) if costs else None,
        "cost_source": "Bedrock response usage fields include tokens but no dollar cost; exact USD requires AWS Pricing/CUR.",
    }
    out = Path("tests/benchmarks/results/bedrock_live_batch/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
