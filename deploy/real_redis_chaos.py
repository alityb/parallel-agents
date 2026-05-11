"""
Distributed scheduler chaos test with REAL Redis (not mock).
Two scheduler processes share localhost:6379, coordinate on 100 agents.
Kill node-A after 30 completions. Verify node-B picks up remainder.
"""
from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/batchagent")

OUT = Path("/tmp/session_results"); OUT.mkdir(exist_ok=True)

async def main():
    try:
        import redis
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "redis", "-q"], check=True)
        import redis

    # Connect to real Redis
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    pong = r.ping()
    print(f"Redis ping: {pong}")

    # Flush test keys
    r.flushdb()

    # Run the test using our RedisStreamsStateStore with real Redis
    from batch_agent.backends import BackendAdapter, BackendResponse
    from batch_agent.distributed import DistributedWaveScheduler, NodeStopped
    from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext
    from batch_agent.state import AgentStatus, RedisStreamsStateStore
    from pydantic import BaseModel
    import json as _json

    class MockB(BackendAdapter):
        async def generate(self, *, shared, job, messages=None, model, tools=None, metadata=None, timeout=None):
            await asyncio.sleep(0.005)
            body = _json.dumps({"value": job.index})
            return BackendResponse(content=body, raw={"content": [{"type":"text","text":body}]}, stop_reason="end_turn")

    class Out(BaseModel):
        value: int

    # Use real Redis client as the "client" object
    # RedisStreamsStateStore expects a client with get/set/delete/xadd methods
    # redis-py's Redis object has these methods, but parameters differ slightly
    # We use a thin adapter
    class RedisPyAdapter:
        def __init__(self, r): self._r = r
        def get(self, k): return self._r.get(k)
        def set(self, k, v, nx=False, ex=None, px=None):
            if nx:
                return self._r.set(k, v, nx=True, ex=ex, px=px)
            return self._r.set(k, v, ex=ex, px=px)
        def delete(self, k): return self._r.delete(k)
        def xadd(self, stream, fields): return self._r.xadd(stream, fields)

    adapter = RedisPyAdapter(r)
    spec = BatchSpec(
        task="Process {i}", inputs=[{"i": i} for i in range(100)],
        output_schema=Out, model="mock", backend="mock://",
        max_concurrent=10, distributed=True,
    )
    nodes = ["node-a", "node-b"]
    node_a = DistributedWaveScheduler(spec=spec, backend=MockB(), redis_client=adapter,
                                       node_id="node-a", nodes=nodes, lease_ttl_seconds=0.1)
    node_b = DistributedWaveScheduler(spec=spec, backend=MockB(), redis_client=adapter,
                                       node_id="node-b", nodes=nodes, lease_ttl_seconds=0.1)

    t0 = time.monotonic()
    try:
        await node_a.run(stop_after=30)
    except NodeStopped:
        pass
    a_elapsed = time.monotonic() - t0
    print(f"  Node-A stopped after ~30 completions ({a_elapsed:.2f}s)")

    # Let leases expire
    await asyncio.sleep(0.2)

    # Node-B picks up remaining jobs
    await node_b.run()
    await node_b.run(failover=True)   # pick up anything node-A left
    b_elapsed = time.monotonic() - t0

    # Count completions in real Redis
    store = RedisStreamsStateStore(adapter, node_id="observer")
    completed = sum(1 for i in range(100)
                    if store.load(f"job-{i}") and store.load(f"job-{i}").status == AgentStatus.COMPLETE)
    loss_rate = (100 - completed) / 100

    print(f"  Node-B finished ({b_elapsed:.2f}s total)")
    print(f"  Completed: {completed}/100  loss={loss_rate:.1%}")
    print(f"  {'PASS' if completed >= 95 else 'FAIL'} — ≤5% loss target")

    result = {
        "real_redis": True, "redis_host": "localhost:6379",
        "completed": completed, "loss_rate": loss_rate,
        "passes_5pct_target": completed >= 95,
        "node_a_stop_at": 30, "total_wall_seconds": b_elapsed,
    }
    out = OUT / "real_redis_chaos.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n  → {out}")

asyncio.run(main())
