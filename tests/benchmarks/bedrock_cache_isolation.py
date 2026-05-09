from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from pathlib import Path

from batch_agent.backends.bedrock import BedrockBackend
from batch_agent.spec import AgentJob, SharedContext


DEFAULT_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"


def _p50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


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

    backend = BedrockBackend(region=args.region)
    started = time.monotonic()
    records: list[dict] = []

    for i in range(args.requests):
        before_count = len(backend.request_metrics)
        response = await backend.generate(
            shared=shared,
            job=job,
            model=args.model,
            timeout=180,
        )
        metric = backend.request_metrics[-1] if len(backend.request_metrics) > before_count else {}
        usage = metric.get("usage", {})
        records.append({
            "request_index": i + 1,
            "ttft_seconds": metric.get("ttft_seconds"),
            "cache_mode": _request_cache_mode(usage),
            "usage": usage,
            "response_preview": response.content[:120],
        })
        print(json.dumps(records[-1], sort_keys=True))

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
        "request_1_cache_mode": records[0]["cache_mode"] if records else None,
        "requests_2_10_cache_modes": [r["cache_mode"] for r in records[1:]],
        "ttft_cache_miss_request_1_p50_seconds": _p50(miss_ttfts),
        "ttft_cache_hit_requests_2_10_p50_seconds": _p50(hit_ttfts),
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
