from __future__ import annotations

import argparse
import time

from _common import base_result, write_results


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _mock_row(config: str, n: int) -> dict:
    # Deterministic mock model:
    # D is one generation pass. E/F are two-turn BatchAgent runs with 100ms of
    # streaming-overlapped tool latency removed by W16. F emits KVFlow hints but
    # has no prefetch-specific benefit until vLLM scheduler integration lands.
    if config == "D":
        wall = 0.06 + (0.0025 * n)
        cache_hit = 0.78 if n == 50 else 0.86 if n == 100 else 0.91
        note = "naive one-turn mock"
    elif config == "E":
        baseline_without_streaming = 0.36 + (0.018 * n)
        wall = baseline_without_streaming - 0.101
        cache_hit = 0.968
        note = "BatchAgent native mock with W15/W16"
    elif config == "F":
        baseline_without_streaming = 0.36 + (0.018 * n)
        wall = baseline_without_streaming - 0.101
        cache_hit = 0.968
        note = "KVFlow hints emitted; prefetch benefit blocked pending vLLM scheduler integration"
    else:
        raise ValueError(f"unsupported config {config!r}")
    return {
        "config": config,
        "n": n,
        "ok": n,
        "failed": 0,
        "wall_clock_seconds": round(wall, 4),
        "throughput_agents_per_sec": round(n / wall, 4),
        "prefix_cache_hit_rate": cache_hit,
        "streaming_tool_dispatch_savings_seconds": 0.101 if config in {"E", "F"} else 0.0,
        "kvflow_prefetch_specific_improvement_seconds": 0.0 if config == "F" else None,
        "note": note,
    }


def main() -> None:
    p = argparse.ArgumentParser("paper_summarization")
    p.add_argument("--live", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--configs", default="D,E,F")
    p.add_argument("--n", default="50,100,200")
    p.add_argument("--output", default="tests/benchmarks/results/paper_summarization/results.json")
    args = p.parse_args()
    started = time.monotonic()
    if args.live:
        status = "blocked_without_live_backend_and_dataset"
        rows = []
    else:
        status = "ok"
        configs = _parse_csv(args.configs)
        ns = [int(value) for value in _parse_csv(args.n)]
        rows = [_mock_row(config, n) for config in configs for n in ns]
    result = base_result("paper_summarization", args.live, started) | {
        "status": status,
        "mode": "live" if args.live else "mock",
        "rows": rows,
        "summary": {
            "streaming_tool_dispatch_savings_seconds": 0.101 if not args.live else None,
            "cache_hit_rate_with_billing_header_fix": 0.968 if not args.live else None,
            "kvflow_prefetch_confirmed": False,
            "kvflow_note": "F emits hints, but prefetch-specific speedup remains blocked pending vLLM scheduler integration.",
        },
    }
    write_results(args.output, result)


if __name__ == "__main__":
    main()
