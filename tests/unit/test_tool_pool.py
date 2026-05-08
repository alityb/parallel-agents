from __future__ import annotations

import asyncio

from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool


def test_tool_pool_coalesces_inflight_calls() -> None:
    async def run() -> None:
        pool = ToolPool()
        results = await asyncio.gather(pool.call("slow_test_tool", {"value": "x"}), pool.call("slow_test_tool", {"value": "x"}))

        assert results == ["X", "X"]

    calls = 0

    @Tool.define(name="slow_test_tool", cacheable=False)
    async def slow(value: str) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return value.upper()

    asyncio.run(run())
    assert calls == 1


def test_tool_pool_caches_completed_calls() -> None:
    async def run() -> None:
        pool = ToolPool(cache_ttl=60)
        assert await pool.call("cached_test_tool", {"value": "x"}) == "x"
        assert await pool.call("cached_test_tool", {"value": "x"}) == "x"

    calls = 0

    @Tool.define(name="cached_test_tool", cacheable=True)
    async def cached(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    asyncio.run(run())
    assert calls == 1
