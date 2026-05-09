from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from batch_agent.backends.bedrock import BedrockBackend
from batch_agent.compaction import compact_messages_async
from batch_agent.spec import Message


MODEL = os.getenv("BEDROCK_COMPACTION_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def _cost_usd(usage: dict) -> float | None:
    # Bedrock returns token usage, not dollar cost. Avoid hardcoding model pricing here;
    # exact USD should come from AWS Pricing/CUR for the model+region actually used.
    return None


async def main() -> None:
    tool_text = " ".join([
        "Paper Alpha reports benchmark FooBench with primary metric accuracy 91 percent.",
        "Model A beats Model B by three points on the validation split.",
        "The ablation says retrieval improves factuality but increases latency."
    ] * 35)
    messages = [
        Message("user", "analyze"),
        Message("assistant_raw", json.dumps([{"type": "text", "text": "turn1"}])),
        Message("tool_result", json.dumps([{"type": "tool_result", "tool_use_id": "t1", "content": tool_text}])),
        Message("assistant_raw", json.dumps([{"type": "text", "text": "turn2"}])),
        Message("tool_result", json.dumps([{"type": "tool_result", "tool_use_id": "t2", "content": tool_text}])),
        Message("assistant_raw", json.dumps([{"type": "text", "text": "turn3"}])),
        Message("tool_result", json.dumps([{"type": "tool_result", "tool_use_id": "t3", "content": tool_text}])),
    ]
    before_chars = sum(len(m.content) for m in messages)
    backend = BedrockBackend(region=REGION)
    started = time.monotonic()
    compacted = await compact_messages_async(messages, current_turn=3, backend=backend, model=MODEL)
    latency = time.monotonic() - started
    after_chars = sum(len(m.content) for m in compacted)
    usage = backend.request_metrics[-1].get("usage", {}) if backend.request_metrics else {}
    payload = {
        "model": MODEL,
        "region": REGION,
        "latency_seconds": latency,
        "chars_before": before_chars,
        "chars_after": after_chars,
        "usage": usage,
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "total_tokens": usage.get("totalTokens"),
        "estimated_cost_usd": _cost_usd(usage),
        "cost_source": "Bedrock response usage fields include tokens but no dollar cost; exact USD requires AWS Pricing/CUR.",
    }
    out = Path("tests/benchmarks/results/bedrock_live_compaction/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
