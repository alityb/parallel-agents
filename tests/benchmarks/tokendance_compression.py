from __future__ import annotations

import asyncio
import time

from _common import base_result, parser, write_results
from batch_agent.backends.vllm_patch.diff_cache_engine import AgentKVSnapshot, DiffCacheEngine


def _tokens(agent_id: int) -> tuple[int, ...]:
    return tuple(range(2048)) + tuple(range(10_000, 10_400)) + tuple(range(20_000 + agent_id * 100, 20_000 + (agent_id + 1) * 100))


def main() -> None:
    args = parser("tokendance_compression").parse_args()
    started = time.monotonic()
    snapshots = [AgentKVSnapshot(job_id=f"job-{i}", tokens=_tokens(i), turn=4) for i in range(100)]
    engine = DiffCacheEngine(block_size_tokens=16)
    asyncio.run(engine.all_gather(snapshots, soft_timeout_seconds=10.0, completion_fraction=1.0))
    stats = engine.stats(snapshots)
    result = base_result("tokendance_compression", args.live, started) | {
        "status": "ok",
        "n": 100,
        "shared_prefix_tokens": 2048,
        "per_agent_context_tokens": 500,
        "full_blocks": stats.full_blocks,
        "stored_unique_blocks": stats.stored_unique_blocks,
        "compression_ratio": stats.compression_ratio,
        "target_ratio": 10.0,
    }
    write_results(args.output, result)


if __name__ == "__main__":
    main()
