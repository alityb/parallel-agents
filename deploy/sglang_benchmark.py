"""
SGLang live backend test + cache hit rate comparison vs vLLM.
Assumes SGLang is running on localhost:30000.
"""
from __future__ import annotations
import asyncio, json, math, re, sys, time
from pathlib import Path
import httpx

sys.path.insert(0, "/home/ubuntu/parallel-agents")
from batch_agent.backends.sglang import SGLangBackend
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import BatchSpec

SGLANG = "sglang://localhost:30000"
VLLM   = "vllm://localhost:8000"   # may not be running; SGLang session is sequential
MODEL  = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You are a concise assistant. " + "Analysis requires structured evaluation. " * 100
SYSTEM = SYSTEM[:4096]
OUT    = Path("/tmp/session_results"); OUT.mkdir(exist_ok=True)
from pydantic import BaseModel

class Out(BaseModel):
    value: int

async def run_sglang_batch(n: int) -> dict:
    backend = SGLangBackend.from_url(SGLANG)

    # Warm prefix
    from batch_agent.spec import SharedContext
    kv_key = await backend.warm_prefix(SharedContext(prefix=SYSTEM), model=MODEL)
    print(f"  warm_prefix hash={kv_key[:12] if kv_key else 'None'}")

    spec = BatchSpec(
        system_prompt=SYSTEM,
        task='Return JSON: {{"value": {idx}}}',
        inputs=[{"idx": i} for i in range(n)],
        output_schema=Out, model=MODEL, backend=SGLANG,
        max_inflight=min(n, 16), max_dispatched=-1,
        max_turns=1, max_retries=2, timeout_per_turn=45,
    )
    t0 = time.monotonic()
    results = await WaveScheduler(TaskCompiler().compile(spec), backend).run()
    wall = time.monotonic() - t0
    ok = sum(1 for r in results if r.ok)

    # Get SGLang cache metrics
    m = await backend.get_cache_metrics()
    return {"n": n, "ok": ok, "wall": wall, "tput": n/wall, "cache_metrics": m}

async def main():
    # First verify SGLang is alive
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("http://localhost:30000/health")
            assert r.status_code == 200
        print("SGLang health: OK")
    except Exception as e:
        print(f"SGLang not ready: {e}")
        # Save placeholder result
        OUT_f = OUT / "sglang_benchmark.json"
        OUT_f.write_text(json.dumps({"status": "sglang_not_running", "error": str(e)}, indent=2))
        return

    print("\nRunning SGLang benchmark...")
    results = {}
    for n in [10, 50]:
        print(f"\n  N={n}...")
        r = await run_sglang_batch(n)
        print(f"    {r['ok']}/{n} OK  wall={r['wall']:.1f}s  tput={r['tput']:.1f}/s")
        print(f"    cache metrics: {r['cache_metrics']}")
        results[f"N{n}"] = r

    # Run the existing integration tests
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration/test_sglang_backend.py",
         "-v", "--tb=short", "--no-header"],
        cwd="/home/ubuntu/parallel-agents",
        env={**__import__("os").environ, "PYTHONPATH": "/home/ubuntu/parallel-agents"},
        capture_output=True, text=True
    )
    test_output = proc.stdout[-3000:]  # last 3000 chars
    print("\nIntegration test output:")
    print(test_output)

    final = {
        "benchmarks": results,
        "integration_tests_passed": proc.returncode == 0,
        "integration_test_output": test_output,
        "model": MODEL,
    }
    out = OUT / "sglang_benchmark.json"
    out.write_text(json.dumps(final, indent=2, sort_keys=True))
    print(f"\n  → {out}")

asyncio.run(main())
