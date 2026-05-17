"""Hydragen-style shared-prefix benchmark.

Measures TTFT and throughput with and without vLLM prefix caching.

Workload: N requests, each with the same ~2048-token system prompt plus a
unique ~500-token per-task code review prompt. All N requests are sent in
parallel (asyncio.gather), simulating a real batch agent run.

Conditions:
  with-cache    -- vLLM running with --enable-prefix-caching (default)
  without-cache -- vLLM running WITHOUT --enable-prefix-caching

Run both conditions and the script writes a combined result JSON for comparison.

Usage:
  # Condition B: prefix caching ON (server already running)
  python prefix_cache_benchmark.py --condition with-cache --ns 5,10,20,50,100

  # Restart vLLM without prefix caching, then:
  python prefix_cache_benchmark.py --condition without-cache --ns 5,10,20,50,100

  # Or run both automatically (restarts vLLM between conditions):
  python prefix_cache_benchmark.py --condition both --ns 5,10,20,50,100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "results" / "prefix_cache_benchmark" / "results.json"
)

# ---------------------------------------------------------------------------
# System prompt builder — pads to a target token count using the Qwen tokenizer
# (approximated at ~0.179 tokens/char when exact tokenizer is unavailable).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """\
You are a senior software engineer conducting automated code reviews. \
Your job is to identify correctness bugs, logic errors, and security \
vulnerabilities. Be precise: cite exact line numbers and classify each \
finding by severity (P0=critical, P1=high, P2=medium).

Check for: off-by-one errors in loops and array indices, null/None \
dereferences before attribute access, undefined variable references \
(especially typos), integer overflow and division by zero, resource \
leaks (unclosed files, connections), incorrect error handling, logic \
inversions, and type mismatches.

Output format for each bug:
  Line: <number>
  Severity: <P0|P1|P2>
  Kind: <off-by-one|null-deref|undefined-var|overflow|resource-leak|...>
  Description: <one sentence>

If no bugs are found, say "No bugs found." Do not report style issues.

Examples:
  off-by-one: `for i in range(0, len(arr) - 1)` misses the last element.
  null-deref: `user.name.strip()` will crash when user is None.
  undefined-var: `total = price * quantty` — typo, quantty undefined.
  div-zero: `avg = total / count` — count may be zero.
"""

PADDING_SENTENCE = (
    "Always check boundary conditions, edge cases, and error paths carefully. "
)


def build_system_prompt(target_tokens: int = 2048) -> str:
    """Return a system prompt of approximately target_tokens tokens."""
    # Approximate: ~0.179 tokens/char for English prose with Qwen tokenizer.
    # Build from base then pad.
    prompt = SYSTEM_PROMPT_BASE
    tokens_per_char = 0.179
    target_chars = int(target_tokens / tokens_per_char)
    while len(prompt) < target_chars:
        prompt += PADDING_SENTENCE
    # Trim to target char count
    prompt = prompt[:target_chars]
    return prompt


# ---------------------------------------------------------------------------
# Per-task prompts — synthetic code files with known bugs
# ---------------------------------------------------------------------------

_TASK_TEMPLATES = [
    """\
def moving_average(values: list[int], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    averages: list[float] = []
    for start in range(0, len(values) - window):  # BUG: off-by-one
        chunk = values[start:start + window]
        averages.append(sum(chunk) / window)
    return averages
""",
    """\
def customer_label(customer: dict | None) -> str:
    name = customer.get("name")  # BUG: customer may be None
    region = customer.get("region", "unknown")
    return f"{name.strip().title()} ({region.upper()})"
""",
    """\
def invoice_total(lines: list[dict]) -> float:
    subtotal = 0.0
    for line in lines:
        subtotal += line["quantity"] * line["price"]
    discount = subtotal * 0.1 if subtotal > 1000 else 0.0
    return subtotal - discunt  # BUG: undefined variable discunt
""",
    """\
def paginate(items: list, page: int, per_page: int) -> list:
    start = page * per_page
    end = start + per_page
    return items[start:end]
# No bug — clean function for variety
""",
    """\
def compute_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator  # BUG: division by zero when denominator==0
""",
    """\
def read_config(path: str) -> dict:
    f = open(path)  # BUG: resource leak, file never closed
    return json.loads(f.read())
""",
    """\
def find_first(items: list, predicate) -> int | None:
    for i in range(len(items) + 1):  # BUG: off-by-one, IndexError on last iter
        if predicate(items[i]):
            return i
    return None
""",
    """\
def merge_counts(a: dict, b: dict) -> dict:
    result = a
    for key, val in b.items():
        result[key] = result.get(key, 0) + val
    return result
# BUG: mutates a in place — result = a is not a copy
""",
]


def task_prompt(index: int) -> str:
    template = _TASK_TEMPLATES[index % len(_TASK_TEMPLATES)]
    return (
        f"Review the following Python function for correctness bugs. "
        f"List each bug with line number and severity.\n\n"
        f"```python\n{template}```"
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    index: int
    ok: bool
    ttft: float  # seconds to first token
    wall: float  # total request time
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    error: str = ""


async def fetch_metrics(base_url: str) -> dict[str, float]:
    """Read vLLM Prometheus metrics and return a name→value dict."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url}/metrics")
            resp.raise_for_status()
        result: dict[str, float] = {}
        for line in resp.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].split("{")[0]
                try:
                    result[name] = float(parts[-1])
                except ValueError:
                    pass
        return result
    except Exception:
        return {}


def extract_cache_hit_rate(metrics: dict[str, float]) -> float | None:
    for key in [
        "vllm:gpu_prefix_cache_hit_rate",
        "vllm:cache_config_info",
    ]:
        if key in metrics:
            return metrics[key]
    # Try the pattern with engine labels
    for key, val in metrics.items():
        if "prefix_cache_hit_rate" in key:
            return val
    return None


async def run_one_request(
    session: httpx.AsyncClient,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    index: int,
    max_tokens: int,
) -> RequestResult:
    started = time.monotonic()
    ttft: float | None = None
    prompt_tokens = completion_tokens = cached_tokens = 0
    chunks: list[str] = []

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with session.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"content-type": "application/json"},
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # TTFT: first chunk with actual content
                if ttft is None:
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = (choices[0] or {}).get("delta") or {}
                        content = delta.get("content")
                        if content:
                            ttft = time.monotonic() - started
                # Usage summary chunk (last chunk with stream_options)
                usage = chunk.get("usage") or {}
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    cached_tokens = (
                        usage.get("cached_tokens")
                        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                        or 0
                    )

        wall = time.monotonic() - started
        return RequestResult(
            index=index,
            ok=True,
            ttft=ttft or wall,
            wall=wall,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

    except Exception as exc:
        return RequestResult(
            index=index,
            ok=False,
            ttft=time.monotonic() - started,
            wall=time.monotonic() - started,
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            error=repr(exc)[:200],
        )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * p / 100) - 1)
    return sorted_vals[idx]


async def run_n(
    base_url: str,
    model: str,
    system_prompt: str,
    n: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """Send N requests in parallel and return aggregate stats."""
    # Snapshot metrics before
    metrics_before = await fetch_metrics(base_url)

    batch_started = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as session:
        tasks = [
            run_one_request(
                session, base_url, model,
                system_prompt, task_prompt(i),
                i, max_tokens,
            )
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)

    wall_clock = time.monotonic() - batch_started

    # Snapshot metrics after
    metrics_after = await fetch_metrics(base_url)

    ok_results = [r for r in results if r.ok]
    ttfts = [r.ttft for r in ok_results]
    total_prompt = sum(r.prompt_tokens for r in ok_results)
    total_cached = sum(r.cached_tokens for r in ok_results)
    total_completion = sum(r.completion_tokens for r in ok_results)

    cache_hit_before = extract_cache_hit_rate(metrics_before)
    cache_hit_after = extract_cache_hit_rate(metrics_after)

    return {
        "n": n,
        "ok": len(ok_results),
        "failed": n - len(ok_results),
        "wall_clock_seconds": round(wall_clock, 3),
        "throughput_agents_per_sec": round(n / wall_clock, 3),
        "ttft_p10": round(percentile(ttfts, 10), 4),
        "ttft_p50": round(percentile(ttfts, 50), 4),
        "ttft_p95": round(percentile(ttfts, 95), 4),
        "ttft_p99": round(percentile(ttfts, 99), 4),
        # All per-request TTFT values sorted ascending — allows full distribution inspection
        "ttft_all_sorted": [round(t, 4) for t in sorted(ttfts)],
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "total_completion_tokens": total_completion,
        "per_agent_prompt_tokens": round(total_prompt / max(1, len(ok_results)), 1),
        "per_agent_cached_tokens": round(total_cached / max(1, len(ok_results)), 1),
        "actual_prefill_tokens": total_prompt - total_cached,
        "per_agent_actual_prefill": round((total_prompt - total_cached) / max(1, len(ok_results)), 1),
        "prefix_cache_hit_rate_before": cache_hit_before,
        "prefix_cache_hit_rate_after": cache_hit_after,
        "errors": [r.error for r in results if not r.ok],
    }


# ---------------------------------------------------------------------------
# vLLM restart helpers
# ---------------------------------------------------------------------------

VLLM_CMD_WITH_CACHE = [
    "bash", "-c",
    "source ~/vllm-env/bin/activate && "
    "nohup python3 -m vllm.entrypoints.openai.api_server "
    "--model Qwen/Qwen2.5-7B-Instruct "
    "--host 0.0.0.0 --port 8080 "
    "--enable-prefix-caching "
    "--max-model-len 4096 "
    "--gpu-memory-utilization 0.9 "
    "> /tmp/vllm_with_cache.log 2>&1 &",
]

VLLM_CMD_WITHOUT_CACHE = [
    "bash", "-c",
    "source ~/vllm-env/bin/activate && "
    "nohup python3 -m vllm.entrypoints.openai.api_server "
    "--model Qwen/Qwen2.5-7B-Instruct "
    "--host 0.0.0.0 --port 8080 "
    "--max-model-len 4096 "
    "--gpu-memory-utilization 0.9 "
    "> /tmp/vllm_without_cache.log 2>&1 &",
]


async def wait_for_server(base_url: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{base_url}/health")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8080")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--condition",
        choices=["with-cache", "without-cache", "both"],
        default="with-cache",
    )
    parser.add_argument("--ns", default="5,10,20,50,100")
    parser.add_argument("--system-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Send one warmup request before each condition to prime the server",
    )
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.ns.split(",")]
    system_prompt = build_system_prompt(args.system_prompt_tokens)
    # Rough token estimate for display
    approx_tokens = int(len(system_prompt) * 0.179)
    print(f"System prompt: {len(system_prompt)} chars, ~{approx_tokens} tokens")
    print(f"Backend: {args.backend}  Model: {args.model}")
    print(f"N values: {ns}  max_tokens: {args.max_tokens}")
    print()

    payload: dict[str, Any] = {
        "benchmark": "prefix_cache_hydragen_style",
        "model": args.model,
        "backend": args.backend,
        "system_prompt_chars": len(system_prompt),
        "approx_system_prompt_tokens": approx_tokens,
        "max_tokens": args.max_tokens,
        "ns": ns,
        "timestamp": time.time(),
    }

    conditions = (
        ["with-cache", "without-cache"] if args.condition == "both" else [args.condition]
    )

    for condition in conditions:
        print(f"=== Condition: {condition} ===")

        if args.condition == "both":
            # Kill existing vLLM
            print("Stopping existing vLLM server...")
            subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
            await asyncio.sleep(5)

            # Start the right version
            cmd = (
                VLLM_CMD_WITH_CACHE if condition == "with-cache" else VLLM_CMD_WITHOUT_CACHE
            )
            subprocess.run(cmd, check=True)
            print("Waiting for server to come up...")
            ready = await wait_for_server(args.backend, timeout=180)
            if not ready:
                print(f"ERROR: server did not start for condition {condition}")
                continue
            print("Server ready.")
            await asyncio.sleep(3)

        # Warmup request
        if args.warmup:
            print("Warming up (1 request)...")
            async with httpx.AsyncClient(timeout=args.timeout) as s:
                await run_one_request(
                    s, args.backend, args.model,
                    system_prompt, task_prompt(0), 0, args.max_tokens,
                )
            await asyncio.sleep(1)

        results_for_condition: list[dict[str, Any]] = []
        for n in ns:
            print(f"  N={n:4d} ... ", end="", flush=True)
            result = await run_n(
                args.backend, args.model,
                system_prompt, n, args.max_tokens, args.timeout,
            )
            results_for_condition.append(result)
            print(
                f"wall={result['wall_clock_seconds']:.2f}s  "
                f"TTFT P10={result['ttft_p10']:.3f}s  "
                f"P50={result['ttft_p50']:.3f}s  "
                f"P95={result['ttft_p95']:.3f}s  "
                f"cache_hit={result['prefix_cache_hit_rate_after']}"
            )

        payload[condition.replace("-", "_")] = results_for_condition

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults written to {args.output}")

    # Print comparison table if both conditions ran
    if "with_cache" in payload and "without_cache" in payload:
        print("\n=== Comparison: with-cache vs without-cache ===")
        print(f"{'N':>6} | {'P10 no-cache':>13} | {'P10 cache':>10} | {'P50 no-cache':>13} | {'P50 cache':>10} | {'P50 speedup':>12}")
        print("-" * 80)
        for wc, nc in zip(payload["with_cache"], payload["without_cache"]):
            n = wc["n"]
            print(
                f"{n:>6} | {nc['ttft_p10']:>12.3f}s | {wc['ttft_p10']:>9.3f}s"
                f" | {nc['ttft_p50']:>12.3f}s | {wc['ttft_p50']:>9.3f}s"
                f" | {nc['ttft_p50']/wc['ttft_p50']:>11.1f}x"
            )


if __name__ == "__main__":
    asyncio.run(main())
