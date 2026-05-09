from __future__ import annotations

import asyncio

from batch_agent.backends.vllm_patch.prefetch_route import (
    handle_pin_blocks_request,
    handle_prefetch_request,
)


class MockCacheEngine:
    def __init__(self) -> None:
        self.calls = []

    def prefetch(self, block_pairs) -> None:
        self.calls.append([list(pair) for pair in block_pairs])


class MockBlockManager:
    def __init__(self) -> None:
        self.pinned = []

    def pin_blocks(self, block_ids) -> None:
        self.pinned.append(list(block_ids))


def test_prefetch_route_maps_kv_keys_to_swap_pairs_and_calls_cache_engine() -> None:
    cache = MockCacheEngine()
    registry = {"kv-a": [[1, 10], [2, 11], [3, 12]], "kv-b": [[4, 13]]}
    payload = {
        "hints": [
            {"job_id": "b", "kv_key": "kv-b", "priority": 0.5, "eta_seconds": 0.1},
            {"job_id": "a", "kv_key": "kv-a", "priority": 2.0, "eta_seconds": 1.0},
            {"job_id": "missing", "kv_key": "kv-missing", "priority": 3.0},
        ]
    }
    result = asyncio.run(handle_prefetch_request(payload, cache_engine=cache, kv_registry=registry))
    assert result["ok"] is True
    assert result["prefetched"] == {"kv-a": [[1, 10], [2, 11], [3, 12]], "kv-b": [[4, 13]]}
    assert result["missing"] == ["kv-missing"]
    assert cache.calls == [[[1, 10], [2, 11], [3, 12]], [[4, 13]]]


def test_prefetch_route_accepts_explicit_block_pairs() -> None:
    cache = MockCacheEngine()
    result = asyncio.run(handle_prefetch_request(
        {"block_ids": [[10, 20], [11, 21]]},
        cache_engine=cache,
        kv_registry={},
    ))
    assert result["ok"] is True
    assert result["prefetched"] == {"__direct__": [[10, 20], [11, 21]]}
    assert cache.calls == [[[10, 20], [11, 21]]]


def test_prefetch_route_rejects_resident_gpu_block_ids() -> None:
    cache = MockCacheEngine()
    result = asyncio.run(handle_prefetch_request(
        {"block_ids": [10, 11]},
        cache_engine=cache,
        kv_registry={},
    ))
    assert result["ok"] is False
    assert "explicit [cpu_block_id, gpu_block_id] pairs" in result["error"]
    assert cache.calls == []


def test_pin_blocks_route_maps_kv_keys_to_block_ids_and_calls_block_manager() -> None:
    manager = MockBlockManager()
    registry = {"kv-prefix": [[1, 10], [2, 11]]}
    result = asyncio.run(handle_pin_blocks_request(
        {"kv_keys": ["kv-prefix", "kv-missing"]},
        block_manager=manager,
        kv_registry=registry,
    ))
    assert result["ok"] is True
    assert result["pinned"] == {"kv-prefix": [10, 11]}
    assert result["missing"] == ["kv-missing"]
    assert manager.pinned == [[10, 11]]
