from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Pricing constants, USD per MTok. Update these when pricing changes.
MODEL_PRICING = {
    "haiku-4.5": {"input": 1.00, "output": 5.00},
    "sonnet-4.6": {"input": 3.00, "output": 15.00},
    "opus-4.6": {"input": 15.00, "output": 75.00},
}
CACHE_READ_MULTIPLIER = 0.10
BATCH_DISCOUNT = 0.50
VLLM_GPU_COST_PER_HOUR = 0.805  # L4 on RunPod
VLLM_AGENTS_PER_HOUR = 5.5 * 3600  # from real benchmark

RESULTS_PATH = Path(__file__).parent / "results" / "cost_comparison" / "results.json"


def usd(value: float) -> str:
    return f"${value:.4f}"


def compute_rows(
    *,
    n: int,
    input_tokens: int,
    output_tokens: int,
    cache_hit_rate: float,
    model: str,
) -> dict[str, Any]:
    if model not in MODEL_PRICING:
        raise ValueError(f"unknown model {model!r}; choose one of {', '.join(sorted(MODEL_PRICING))}")
    input_price = MODEL_PRICING[model]["input"]
    output_price = MODEL_PRICING[model]["output"]
    total_input_mtok = (n * input_tokens) / 1_000_000
    total_output_mtok = (n * output_tokens) / 1_000_000

    naive_cost = total_input_mtok * input_price + total_output_mtok * output_price
    batch_cost = naive_cost * BATCH_DISCOUNT

    cached_input_mtok = total_input_mtok * cache_hit_rate
    uncached_input_mtok = total_input_mtok - cached_input_mtok
    cached_api_cost = (
        uncached_input_mtok * input_price
        + cached_input_mtok * input_price * CACHE_READ_MULTIPLIER
        + total_output_mtok * output_price
    )

    vllm_hours = n / VLLM_AGENTS_PER_HOUR
    vllm_cost = vllm_hours * VLLM_GPU_COST_PER_HOUR
    dynamo_cost = vllm_cost

    rows = [
        {
            "name": "Naive API",
            "cost_usd": naive_cost,
            "relative_to_naive": 1.0,
            "note": "Parallel API calls, no cache discount.",
        },
        {
            "name": "Anthropic Batch API",
            "cost_usd": batch_cost,
            "relative_to_naive": batch_cost / naive_cost,
            "note": "No tool calls, single turn only.",
        },
        {
            "name": "BatchAgent + API caching",
            "cost_usd": cached_api_cost,
            "relative_to_naive": cached_api_cost / naive_cost,
            "note": f"Uses measured {cache_hit_rate:.1%} cache hit rate.",
        },
        {
            "name": "BatchAgent + self-hosted vLLM",
            "cost_usd": vllm_cost,
            "relative_to_naive": vllm_cost / naive_cost,
            "note": f"L4 at ${VLLM_GPU_COST_PER_HOUR:.3f}/hr, {VLLM_AGENTS_PER_HOUR:.0f} agents/hr.",
        },
        {
            "name": "BatchAgent + NVIDIA Dynamo",
            "cost_usd": dynamo_cost,
            "relative_to_naive": dynamo_cost / naive_cost,
            "note": "Same L4 cost model as vLLM; nvext_agent_hints benefit is scheduling, not pricing.",
        },
    ]

    return {
        "parameters": {
            "n": n,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit_rate": cache_hit_rate,
            "model": model,
        },
        "pricing": {
            "models": MODEL_PRICING,
            "selected_input_per_mtok": input_price,
            "selected_output_per_mtok": output_price,
            "cache_read_multiplier": CACHE_READ_MULTIPLIER,
            "batch_discount": BATCH_DISCOUNT,
            "vllm_gpu_cost_per_hour": VLLM_GPU_COST_PER_HOUR,
            "vllm_agents_per_hour": VLLM_AGENTS_PER_HOUR,
        },
        "autoresearch_example": {
            "status": "blocked",
            "reason": "Step 5 live AutoResearch run did not execute because ANTHROPIC_API_KEY and search API keys were unavailable.",
            "planned_run": {
                "n_questions": 20,
                "planner": "opus-4.6",
                "workers": "sonnet-4.6",
                "reducer": "opus-4.6",
            },
            "real_cost_usd": None,
        },
        "rows": rows,
    }


def format_table(result: dict[str, Any]) -> str:
    params = result["parameters"]
    lines = [
        f"Cost comparison for N={params['n']}, input={params['input_tokens']} tokens, "
        f"output={params['output_tokens']} tokens, model={params['model']}",
        "",
        "| Mode | Cost / batch | Relative | Note |",
        "|---|---:|---:|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['name']} | {usd(row['cost_usd'])} | "
            f"{row['relative_to_naive']:.3f}x | {row['note']} |"
        )
    return "\n".join(lines)


def format_latex(result: dict[str, Any]) -> str:
    rows = [
        r"\begin{tabular}{lrrl}",
        r"\hline",
        r"Mode & Cost / batch & Relative & Note \\",
        r"\hline",
    ]
    for row in result["rows"]:
        rows.append(
            f"{row['name']} & {usd(row['cost_usd'])} & "
            f"{row['relative_to_naive']:.3f}x & {row['note']} \\\\"
        )
    rows.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--input-tokens", type=int, default=3000)
    parser.add_argument("--output-tokens", type=int, default=500)
    parser.add_argument("--cache-hit-rate", type=float, default=0.968)
    parser.add_argument("--model", default="sonnet-4.6", choices=sorted(MODEL_PRICING))
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    result = compute_rows(
        n=args.N,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        cache_hit_rate=args.cache_hit_rate,
        model=args.model,
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(format_latex(result) if args.latex else format_table(result))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
