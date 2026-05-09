from __future__ import annotations

import asyncio
import time

from _common import base_result, parser, write_results
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool


def main() -> None:
    args = parser("tool_dedup_efficiency").parse_args()
    started = time.monotonic()
    calls = {"count": 0}

    @Tool.define(name="bench_shared_read", cacheable=False)
    async def bench_shared_read(doc_id: int) -> str:
        calls["count"] += 1
        await asyncio.sleep(0.001)
        return f"doc-{doc_id}"

    async def run() -> None:
        pool = ToolPool()
        await asyncio.gather(*[
            pool.call("bench_shared_read", {"doc_id": doc_id})
            for _agent in range(100)
            for doc_id in range(10)
        ])

    if not args.live:
        asyncio.run(run())
    result = base_result("tool_dedup_efficiency", args.live, started) | {
        "status": "ok" if not args.live else "blocked_without_live_toolset",
        "requested_reads": 1000 if not args.live else None,
        "actual_reads": calls["count"] if not args.live else None,
        "dedup_ratio": (1000 / calls["count"]) if calls["count"] else None,
    }
    write_results(args.output, result)


if __name__ == "__main__":
    main()
