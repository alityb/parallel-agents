from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
import math
from pathlib import Path

from batch_agent.backends.bedrock import BedrockBackend
from batch_agent.spec import AgentJob, SharedContext


DEFAULT_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"


def _p50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[math.ceil(len(values) * 0.95) - 1]


def _request_cache_mode(usage: dict) -> str:
    if usage.get("cacheWriteInputTokens", 0) > 0:
        return "write"
    if usage.get("cacheReadInputTokens", 0) > 0:
        return "read"
    return "none"


async def main() -> None:
    parser = argparse.ArgumentParser("bedrock_cache_isolation")
    parser.add_argument("--model", default=os.getenv("BEDROCK_LIVE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--pad-tokens", type=int, default=1200)
    parser.add_argument("--parallel-cache-hits", action="store_true", help="Send request 1 first, then requests 2-N concurrently")
    parser.add_argument("--output", default="tests/benchmarks/results/bedrock_cache_isolation/results.json")
    args = parser.parse_args()

    # Unique prefix avoids accidentally reading a cache written by earlier runs.
    run_marker = f"cache-isolation-run-{uuid.uuid4()}"
    system_prompt = run_marker + " " + " ".join(["cacheable-prefix-token"] * args.pad_tokens)
    user_prompt = "Return exactly this JSON object and nothing else: {\"ok\": true}"
    shared = SharedContext(prefix=system_prompt)
    job = AgentJob(
        job_id="cache-isolation",
        index=0,
        input_data={},
        prompt=user_prompt,
        estimated_prompt_tokens=(len(system_prompt) + len(user_prompt)) // 4,
    )

    started = time.monotonic()
    records: list[dict] = []

    async def run_one(request_index: int, backend: BedrockBackend) -> dict:
        before_count = len(backend.request_metrics)
        response = await backend.generate(
            shared=shared,
            job=job,
            model=args.model,
            timeout=180,
        )
        metric = backend.request_metrics[-1] if len(backend.request_metrics) > before_count else {}
        usage = metric.get("usage", {})
        return {
            "request_index": request_index,
            "ttft_seconds": metric.get("ttft_seconds"),
            "cache_mode": _request_cache_mode(usage),
            "usage": usage,
            "response_preview": response.content[:120],
        }

    if args.parallel_cache_hits:
        writer_backend = BedrockBackend(region=args.region, max_concurrent_ceiling=10)
        first = await run_one(1, writer_backend)
        records.append(first)
        print(json.dumps(first, sort_keys=True))
        hit_records = await asyncio.gather(*[
            run_one(i, BedrockBackend(region=args.region, max_concurrent_ceiling=10))
            for i in range(2, args.requests + 1)
        ])
        records.extend(hit_records)
        for record in hit_records:
            print(json.dumps(record, sort_keys=True))
    else:
        backend = BedrockBackend(region=args.region)
        for i in range(args.requests):
            record = await run_one(i + 1, backend)
            records.append(record)
            print(json.dumps(record, sort_keys=True))

    miss_ttfts = [r["ttft_seconds"] for r in records if r["request_index"] == 1 and r["ttft_seconds"] is not None]
    hit_ttfts = [r["ttft_seconds"] for r in records if r["request_index"] > 1 and r["ttft_seconds"] is not None]
    first_usage = records[0]["usage"] if records else {}
    hit_usages = [r["usage"] for r in records[1:]]

    payload = {
        "benchmark": "bedrock_cache_isolation",
        "model": args.model,
        "region": args.region,
        "requests": args.requests,
        "system_prompt_estimated_tokens": len(system_prompt.split()),
        "wall_clock_seconds": time.monotonic() - started,
        "records": records,
        "parallel_cache_hits": args.parallel_cache_hits,
        "request_1_cache_mode": records[0]["cache_mode"] if records else None,
        "requests_2_10_cache_modes": [r["cache_mode"] for r in records[1:]],
        "ttft_cache_miss_request_1_p50_seconds": _p50(miss_ttfts),
        "ttft_cache_hit_requests_2_10_p50_seconds": _p50(hit_ttfts),
        "ttft_cache_hit_requests_2_10_p95_seconds": _p95(hit_ttfts),
        "hit_to_miss_ttft_ratio": (_p50(hit_ttfts) / _p50(miss_ttfts)) if hit_ttfts and miss_ttfts else None,
        "cache_write_input_tokens_request_1": first_usage.get("cacheWriteInputTokens", 0),
        "cache_read_input_tokens_requests_2_10": sum(u.get("cacheReadInputTokens", 0) for u in hit_usages),
        "finding": None,
    }
    miss_p50 = payload["ttft_cache_miss_request_1_p50_seconds"]
    hit_p50 = payload["ttft_cache_hit_requests_2_10_p50_seconds"]
    if miss_p50 is not None and hit_p50 is not None:
        if hit_p50 < miss_p50:
            payload["finding"] = "cache_hit_ttft_lower_than_cache_miss"
        else:
            payload["finding"] = (
                "bedrock_queue_latency_dominates_prefill_savings_for_1200_token_prefix; "
                "prompt_caching_saves_tokens_but_not_latency_at_this_scale"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
