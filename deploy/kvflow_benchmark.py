"""
KVFlow benchmark: TTFT-after-TOOL_WAIT with explicit measurement-integrity modes.

The important integrity check is whether kv_key-only prefetch hints produce a
measurable effect after the vLLM patch was fixed to reject unsafe resident block
IDs. If condition B does not beat condition A, the old 72ms result was not
evidence of KVFlow prefetch specifically.

Conditions:
  A: kvflow=False. No prefetch hints.
  B: kvflow=True. Hints are sent, but the corrected patch rejects kv_key-only
     hints as missing because no CPU->GPU block mappings are available.
  C: Same as B, plus an explicit warm_prefix()/metrics verification pass.

All three use the same A10G/Qwen workload and report turn-2 TTFT after TOOL_WAIT.
"""
from __future__ import annotations
import argparse
import asyncio, json, math, sys, time
from pathlib import Path
import httpx

sys.path.insert(0, "/home/ubuntu/parallel-agents")
from batch_agent.backends.vllm import VLLMBackend
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import BatchSpec, SharedContext
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool
from batch_agent.utils import INTERNAL_HTTP_TIMEOUT, NO_API_KEY

VLLM = "vllm://localhost:8000"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You are a concise assistant. " + "Analysis requires structured evaluation. " * 100
SYSTEM = SYSTEM[:4096]
SHARED = "/tmp/kvflow_bench_doc.txt"
Path(SHARED).write_text("KVFlow benchmark document. The quick brown fox. GPU memory is precious.")
OUT = Path("/tmp/session_results"); OUT.mkdir(exist_ok=True)

async def _do_kvflow_read(path: str = SHARED) -> str:
    await asyncio.sleep(0.3)   # 300ms tool latency → gives advisor time to emit hints
    return Path(path).read_text()


@Tool.define(name="kvflow_read", cacheable=True)
async def kvflow_read(path: str = SHARED) -> str:
    """Read a file — simulates a real tool call."""
    return await _do_kvflow_read(path)


@Tool.define(name="kvflow_analysis", cacheable=True)
async def kvflow_analysis(path: str = SHARED) -> str:
    """Alias for models that name the benchmark tool semantically."""
    return await _do_kvflow_read(path)

# Track per-turn latencies to isolate TTFT-after-TOOL_WAIT
import logging
turn_latencies: dict[str, list[tuple[int, float]]] = {}  # job_id -> [(turn, latency)]

class InstrumentedBackend(VLLMBackend):
    """Wraps VLLMBackend to record per-turn TTFT and prefetch responses."""
    def __init__(self, *args, disable_warm_prefix: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_warm_prefix = disable_warm_prefix
        self.prefetch_statuses: list[int] = []
        self.prefetch_responses: list[dict] = []

    async def warm_prefix(self, shared, model):
        if self.disable_warm_prefix:
            return None
        return await super().warm_prefix(shared, model)

    async def generate(self, *, shared, job, messages=None, model, tools=None, metadata=None, timeout=None):
        t0 = time.monotonic()
        result = await super().generate(shared=shared, job=job, messages=messages,
                                        model=model, tools=tools, metadata=metadata, timeout=timeout)
        elapsed = time.monotonic() - t0
        # Figure out which turn based on message count
        n_msgs = len(messages or [])
        has_tool_result = any(getattr(m, 'role', '') == 'tool_result' for m in (messages or []))
        turn = 2 if has_tool_result else 1
        lst = turn_latencies.setdefault(job.job_id, [])
        lst.append((turn, elapsed))
        return result

    async def send_prefetch_hints(self, hints):
        if not hints:
            return
        payload = [h.to_dict() if hasattr(h, "to_dict") else h for h in hints]
        try:
            async with httpx.AsyncClient(timeout=INTERNAL_HTTP_TIMEOUT) as client:
                response = await client.post(
                    f"{self.base_url}/internal/prefetch",
                    json={"hints": payload},
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
            self.prefetch_statuses.append(response.status_code)
            try:
                self.prefetch_responses.append(response.json())
            except Exception:
                self.prefetch_responses.append({"raw": response.text[:500]})
        except Exception as exc:
            self.prefetch_statuses.append(0)
            self.prefetch_responses.append({"error": str(exc)})


def _cache_delta(before: dict, after: dict) -> dict:
    q0 = before.get("prefix_cache_query_tokens", 0.0)
    h0 = before.get("prefix_cache_hit_tokens", 0.0)
    q1 = after.get("prefix_cache_query_tokens", 0.0)
    h1 = after.get("prefix_cache_hit_tokens", 0.0)
    dq = q1 - q0
    dh = h1 - h0
    return {
        "prefix_cache_query_tokens_delta": dq,
        "prefix_cache_hit_tokens_delta": dh,
        "prefix_cache_hit_rate_delta_window": (dh / dq) if dq > 0 else None,
        "prefix_cache_hit_rate_before": before.get("prefix_cache_hit_rate"),
        "prefix_cache_hit_rate_after": after.get("prefix_cache_hit_rate"),
    }


async def _verify_warm_prefix_active(backend: InstrumentedBackend) -> dict:
    before = await backend.get_cache_metrics()
    kv_key = await backend.warm_prefix(SharedContext(prefix=SYSTEM), MODEL)
    after_warm = await backend.get_cache_metrics()
    # Hit the same prefix once more so metrics can show whether cached prefix
    # counters move after the warm request.
    await backend.generate(
        shared=SharedContext(prefix=SYSTEM),
        job=type("Job", (), {
            "job_id": "warm-verify",
            "index": 0,
            "prompt": "ping",
            "input_data": {},
            "estimated_prompt_tokens": 1,
        })(),
        messages=None,
        model=MODEL,
        tools=None,
        timeout=60,
    )
    after_probe = await backend.get_cache_metrics()
    return {
        "kv_key": kv_key,
        "before": before,
        "after_warm": after_warm,
        "after_probe": after_probe,
        "warm_delta": _cache_delta(before, after_warm),
        "probe_delta": _cache_delta(after_warm, after_probe),
    }


async def run_kvflow_benchmark(
    *,
    condition: str,
    n_agents: int = 20,
    kvflow: bool = True,
    verify_warm_prefix: bool = False,
) -> dict:
    turn_latencies.clear()
    pool = ToolPool()
    spec = BatchSpec(
        system_prompt=SYSTEM,
        task="Call kvflow_read with path={file} then return JSON: {{\"agent\": {idx}, \"word\": \"first word\"}}",
        inputs=[{"file": SHARED, "idx": i} for i in range(n_agents)],
        tools=[Tool.registry["kvflow_read"], Tool.registry["kvflow_analysis"]],
        model=MODEL, backend=VLLM,
        max_inflight=min(n_agents, 16), max_dispatched=-1,
        max_turns=2, max_retries=1, timeout_per_turn=60, timeout_per_tool=5,
        kvflow=kvflow,
    )
    backend = InstrumentedBackend(
        api_key=NO_API_KEY,
        base_url="http://localhost:8000",
        block_sharing_probe_agents=0,
    )
    warm_verification = None
    if verify_warm_prefix:
        warm_verification = await _verify_warm_prefix_active(backend)

    cache_before = await backend.get_cache_metrics()
    t0 = time.monotonic()
    results = await WaveScheduler(TaskCompiler().compile(spec), backend, pool).run()
    wall = time.monotonic() - t0
    cache_after = await backend.get_cache_metrics()

    ok = sum(1 for r in results if r.ok)
    # Collect turn-1 and turn-2 latencies separately
    t1_lats = [lat for lats in turn_latencies.values() for turn, lat in lats if turn == 1]
    t2_lats = [lat for lats in turn_latencies.values() for turn, lat in lats if turn == 2]

    def pct(vals, p):
        if not vals: return None
        return sorted(vals)[math.ceil(len(vals)*p)-1]

    return {
        "condition": condition,
        "n": n_agents, "ok": ok, "failed": n_agents - ok, "kvflow": kvflow,
        "wall_seconds": wall, "throughput": n_agents / wall,
        "turn1_ttft_p50": pct(t1_lats, .5), "turn1_ttft_p95": pct(t1_lats, .95),
        "turn2_ttft_after_tool_wait_p50": pct(t2_lats, .5),
        "turn2_ttft_after_tool_wait_p95": pct(t2_lats, .95),
        "prefetch_statuses": backend.prefetch_statuses,
        "prefetch_response_samples": backend.prefetch_responses[:5],
        "prefetch_missing_total": sum(len(r.get("missing", [])) for r in backend.prefetch_responses if isinstance(r, dict)),
        "prefetch_block_pairs_total": sum(len(r.get("prefetched", {}).get("block_pairs", [])) for r in backend.prefetch_responses if isinstance(r, dict)),
        "cache_before": cache_before,
        "cache_after": cache_after,
        "cache_delta": _cache_delta(cache_before, cache_after),
        "warm_prefix_verification": warm_verification,
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--output", default=str(OUT / "kvflow_measurement_integrity.json"))
    args = parser.parse_args()

    print("Running KVFlow measurement-integrity benchmark...")
    print("  Condition A: kvflow=False, no prefetch hints")
    r_a = await run_kvflow_benchmark(condition="A_no_hints", n_agents=args.n_agents, kvflow=False)
    print(f"    A: turn2 P50={r_a['turn2_ttft_after_tool_wait_p50']:.6f}s wall={r_a['wall_seconds']:.3f}s ok={r_a['ok']}/{args.n_agents}")

    print("  Condition B: kvflow=True, kv_key-only hints sent to corrected patch")
    r_b = await run_kvflow_benchmark(condition="B_hints_rejected", n_agents=args.n_agents, kvflow=True)
    print(f"    B: turn2 P50={r_b['turn2_ttft_after_tool_wait_p50']:.6f}s wall={r_b['wall_seconds']:.3f}s ok={r_b['ok']}/{args.n_agents} missing={r_b['prefetch_missing_total']} block_pairs={r_b['prefetch_block_pairs_total']}")

    print("  Condition C: same as B, with explicit warm_prefix metrics verification")
    r_c = await run_kvflow_benchmark(condition="C_hints_rejected_warm_verified", n_agents=args.n_agents, kvflow=True, verify_warm_prefix=True)
    print(f"    C: turn2 P50={r_c['turn2_ttft_after_tool_wait_p50']:.6f}s wall={r_c['wall_seconds']:.3f}s ok={r_c['ok']}/{args.n_agents} missing={r_c['prefetch_missing_total']} block_pairs={r_c['prefetch_block_pairs_total']}")

    a = r_a["turn2_ttft_after_tool_wait_p50"] or 0.0
    b = r_b["turn2_ttft_after_tool_wait_p50"] or 0.0
    c = r_c["turn2_ttft_after_tool_wait_p50"] or 0.0
    b_minus_a = b - a
    c_minus_a = c - a
    tolerance = 0.020
    if b < a - tolerance:
        finding = "kvflow_prefetch_candidate_b_beats_a"
    elif c < a - tolerance:
        finding = "prefix_cache_or_warm_state_candidate_c_beats_a"
    else:
        finding = "no_measurable_prefetch_effect_a_b_c_similar"
    result = {
        "condition_a_no_hints": r_a,
        "condition_b_hints_rejected": r_b,
        "condition_c_hints_rejected_warm_verified": r_c,
        "b_minus_a_turn2_p50_seconds": b_minus_a,
        "c_minus_a_turn2_p50_seconds": c_minus_a,
        "significance_threshold_seconds": tolerance,
        "finding": finding,
        "model": MODEL, "tool_latency_ms": 300, "n_agents": args.n_agents,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n  → {out}")
    print(json.dumps({
        "finding": finding,
        "A_turn2_p50": a,
        "B_turn2_p50": b,
        "C_turn2_p50": c,
        "B_minus_A": b_minus_a,
        "C_minus_A": c_minus_a,
    }, indent=2))

asyncio.run(main())
