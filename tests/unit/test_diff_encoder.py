from __future__ import annotations

import asyncio

from batch_agent.backends.vllm_patch.diff_cache_engine import (
    AgentKVSnapshot,
    DiffCacheEngine,
    maybe_create_diff_cache_engine,
)


def _make_tokens(agent_id: int) -> tuple[int, ...]:
    shared_prefix = tuple(range(2048))
    # 500-token per-agent context: 400 tokens common across agents, 100 unique.
    common_context = tuple(range(10_000, 10_400))
    unique_context = tuple(range(20_000 + agent_id * 100, 20_000 + (agent_id + 1) * 100))
    return shared_prefix + common_context + unique_context


def test_diff_kv_false_returns_none() -> None:
    assert maybe_create_diff_cache_engine(diff_kv=False) is None


def test_tokendance_diff_encoder_compression_ratio_at_100_agents() -> None:
    snapshots = [
        AgentKVSnapshot(job_id=f"job-{i}", tokens=_make_tokens(i), turn=4)
        for i in range(100)
    ]
    engine = DiffCacheEngine(block_size_tokens=16)
    encoded = asyncio.run(engine.all_gather(
        snapshots,
        soft_timeout_seconds=10.0,
        completion_fraction=1.0,
    ))
    stats = engine.stats(snapshots)
    assert len(encoded) == 100
    assert stats.compression_ratio >= 10.0
    assert stats.agents_encoded == 100
