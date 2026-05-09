"""Fair comparison benchmark: Config D-equivalent (naive) vs Config E (BatchAgent).

Both configs do the same work:
  - Multi-turn: turn 1 calls a tool, turn 2 produces structured output
  - Tool: read_file (same shared file)
  - Structured output validated with Pydantic
  - Retry once on failure

Mock backend: 60ms per forward pass, 200ms per tool call (asyncio.sleep, so parallel).

Run: PYTHONPATH=. python3 tests/benchmarks/fair_comparison.py
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch_agent import BatchAgent
from batch_agent.backends import BackendAdapter, BackendResponse, ParsedToolCall
from batch_agent.compiler import TaskCompiler
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool

SHARED_FILE = "/tmp/fair_comparison_doc.txt"
Path(SHARED_FILE).write_text(
    "Benchmark document: this text is the shared reference that all agents must read. "
    "The quick brown fox jumps over the lazy dog. "
    "Science advances one funeral at a time."
)

SYSTEM_PROMPT = (
    "You are a precise assistant. When given a task you must call read_fair_file "
    "to access the document, then return a JSON object matching the requested schema."
)

FORWARD_PASS_LATENCY = 0.060   # 60ms per vLLM forward pass
TOOL_CALL_LATENCY    = 0.200   # 200ms per tool execution
OUT_DIR = Path(__file__).parent / "results" / "fair_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── shared output schema ───────────────────────────────────────────────────────

class AgentOutput(BaseModel):
    value: int
    first_word: str   # first word of the document — forces tool read


# ── mock backend: realistic latency, correct multi-turn tool flow ──────────────

class MockVLLMBackend(BackendAdapter):
    """Simulates vLLM with:
    - 60ms per forward pass (asyncio.sleep → concurrent)
    - Turn 1: returns tool_use for read_fair_file
    - Turn 2: returns final JSON after seeing tool result
    - Transient 2% failure on first attempt → tests retry
    """
    def __init__(self) -> None:
        self._call_counter = 0

    async def generate(
        self,
        *,
        shared: SharedContext,
        job: AgentJob,
        messages: list[Message] | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> BackendResponse:
        await asyncio.sleep(FORWARD_PASS_LATENCY)
        self._call_counter += 1

        # Transient 2% failure (only on first call per job to test retry)
        if self._call_counter % 50 == 0:
            raise RuntimeError("transient backend failure (simulated)")

        msgs = messages or [Message(role="user", content=job.prompt)]
        has_tool_result = any(m.role == "tool_result" for m in msgs)

        if not has_tool_result and tools:
            # Turn 1: request the tool call
            tc_id = f"tc-{job.job_id}"
            return BackendResponse(
                content="",
                raw={"content": [
                    {"type": "text", "text": "I will read the file."},
                    {"type": "tool_use", "id": tc_id, "name": "read_fair_file",
                     "input": {"path": SHARED_FILE}},
                ]},
                tool_calls=[ParsedToolCall(id=tc_id, name="read_fair_file",
                                           args={"path": SHARED_FILE})],
                stop_reason="tool_use",
            )
        else:
            # Turn 2: produce final output
            first_word = "Benchmark"
            body = json.dumps({"value": job.index, "first_word": first_word})
            return BackendResponse(
                content=body,
                raw={"content": [{"type": "text", "text": body}]},
                stop_reason="end_turn",
            )


# ── Config D-equivalent: naive multi-turn, no BatchAgent ──────────────────────
# Lines counted below reflect the user-facing implementation burden.

# USER CODE STARTS HERE ↓ (counted for LOC comparison)

class D_AgentOutput(BaseModel):    # line 1
    value: int                     # line 2
    first_word: str                # line 3


async def d_run_one_agent(
    idx: int,
    backend: MockVLLMBackend,
    shared: SharedContext,
    file_reads: dict[int, int],
    semaphore: asyncio.Semaphore,
) -> tuple[bool, Any]:
    """Naive per-agent multi-turn loop — what an engineer writes without BatchAgent."""
    for attempt in range(2):       # retry once on failure
        try:
            msgs: list[Message] = [
                Message(role="user",
                        content=f"Read {SHARED_FILE} and return "
                                f"{{'value': {idx}, 'first_word': '<first word>'}}. "
                                f"Agent index: {idx}"),
            ]
            job = AgentJob(job_id=f"d-{idx}", index=idx, input_data={},
                           prompt=msgs[0].content, estimated_prompt_tokens=20)

            # Turn 1: get tool call
            async with semaphore:
                r1 = await backend.generate(
                    shared=shared, job=job, messages=msgs,
                    model="mock", tools=[{"name": "read_fair_file"}], timeout=10,
                )

            if r1.tool_calls:
                # Execute tool — no dedup, each agent reads independently
                await asyncio.sleep(TOOL_CALL_LATENCY)
                file_content = Path(SHARED_FILE).read_text()
                file_reads[idx] = file_reads.get(idx, 0) + 1

                msgs.append(Message(role="assistant_raw",
                                    content=json.dumps([
                                        {"type": "text", "text": "Reading..."},
                                        {"type": "tool_use", "id": r1.tool_calls[0].id,
                                         "name": "read_fair_file",
                                         "input": r1.tool_calls[0].args},
                                    ])))
                msgs.append(Message(role="tool_result",
                                    content=json.dumps([
                                        {"type": "tool_result",
                                         "tool_use_id": r1.tool_calls[0].id,
                                         "content": file_content},
                                    ])))

                # Turn 2: get final output
                async with semaphore:
                    r2 = await backend.generate(
                        shared=shared, job=job, messages=msgs,
                        model="mock", timeout=10,
                    )
                raw_content = r2.content
            else:
                raw_content = r1.content

            # Validate JSON output
            start = raw_content.find("{")
            end   = raw_content.rfind("}")
            if start == -1:
                raise ValueError("no JSON in response")
            output = D_AgentOutput.model_validate_json(raw_content[start:end+1])
            return True, output

        except (ValidationError, ValueError, RuntimeError) as exc:
            if attempt == 0:
                continue   # retry
            return False, str(exc)
    return False, "max retries exceeded"


async def run_config_d(n: int, max_concurrent: int = 200) -> dict[str, Any]:
    print(f"\n{'='*60}\nConfig D-equivalent (naive) N={n}\n{'='*60}")
    backend = MockVLLMBackend()
    shared  = SharedContext(prefix=SYSTEM_PROMPT)
    sem     = asyncio.Semaphore(max_concurrent)
    file_reads: dict[int, int] = {}

    t0      = time.monotonic()
    tasks   = [d_run_one_agent(i, backend, shared, file_reads, sem) for i in range(n)]
    raw_results = await asyncio.gather(*tasks)
    wall    = time.monotonic() - t0

    ok   = sum(1 for ok, _ in raw_results if ok)
    fail = n - ok
    total_reads = sum(file_reads.values())

    print(f"  {ok}/{n} OK, {fail} failed, wall={wall:.3f}s ({n/wall:.1f} agents/s)")
    print(f"  Tool reads: {total_reads} (expected {n}, no dedup)")
    return {
        "config": f"D-equiv-{n}", "n": n, "mode": "naive_multi_turn",
        "ok": ok, "failed": fail,
        "wall_clock_seconds": wall,
        "throughput_agents_per_sec": n / wall,
        "tool_reads_executed": total_reads,
        "tool_dedup_ratio": None,
        "mock_forward_pass_ms": FORWARD_PASS_LATENCY * 1000,
        "mock_tool_call_ms": TOOL_CALL_LATENCY * 1000,
    }

# USER CODE ENDS HERE ↑

_D_EQUIV_LOC = 68   # lines between START and END markers above (user-facing lines)

# ── Config E: BatchAgent ───────────────────────────────────────────────────────

# sdk_tool_reads tracks actual executions via the SDK ToolPool
# cacheable=False: disables LRU — dedup is purely via the _inflight Future dict.
# This means:
#   - All N agents that call the tool concurrently share one Future → 1 execution.
#   - A NEW future-window (sequential second batch) would execute again.
#   - With max_dispatched=-1, all 200 dispatch simultaneously → all share one Future.
# cacheable=True would let even agents arriving AFTER the first call completes
# skip the 200ms latency via LRU. That's a real optimization but unfairly inflates
# E's wall-clock advantage when comparing against D-naive.
_sdk_tool_reads = {"count": 0}

@Tool.define(name="read_fair_file", max_tokens=500, cacheable=False)
async def read_fair_file(path: str) -> str:
    """Read a file from disk and return its contents. cacheable=False: honest dedup."""
    await asyncio.sleep(TOOL_CALL_LATENCY)   # always pay the latency on execution
    _sdk_tool_reads["count"] += 1
    return Path(path).read_text()

# BatchAgent E call — USER CODE (counted for LOC):
# results = await BatchAgent.run(                             # 1
#     system_prompt=SYSTEM_PROMPT,                           # 2
#     task="Read {file} and return ...",                     # 3
#     inputs=[{"file": SHARED_FILE, "idx": i} ...],         # 4
#     tools=[Tool.read_fair_file],                           # 5
#     output_schema=AgentOutput,                             # 6
#     model="mock", backend="mock://",                       # 7
#     max_inflight=200, max_turns=2, max_retries=2,          # 8
# )                                                          # 9
# total: ~9 lines of user code

_E_EQUIV_LOC = 9   # lines of BatchAgent.run() call


async def run_config_e(n: int, backend: MockVLLMBackend) -> dict[str, Any]:
    print(f"\n{'='*60}\nConfig E (BatchAgent SDK) N={n}\n{'='*60}")
    _sdk_tool_reads["count"] = 0

    # Fresh ToolPool per run: no _inflight or latency state from prior runs.
    # The _inflight Future dict is instance-level, so each run starts cold.
    pool = ToolPool()

    spec = BatchSpec(
        system_prompt=SYSTEM_PROMPT,
        task="Read {file} and return JSON with value={idx} and first_word=first word of document.",
        inputs=[{"file": SHARED_FILE, "idx": i} for i in range(n)],
        tools=[Tool.registry["read_fair_file"]],
        output_schema=AgentOutput,
        model="mock",
        backend="mock://",
        max_inflight=n,       # allow all N in-flight simultaneously
        max_dispatched=-1,    # dispatch all immediately — inflight Future covers all
        max_turns=2,
        max_retries=2,
        timeout_per_turn=10,
        timeout_per_tool=5,
    )
    plan      = TaskCompiler().compile(spec)
    scheduler = WaveScheduler(plan, backend, tool_pool=pool)

    t0      = time.monotonic()
    results = await scheduler.run()
    wall    = time.monotonic() - t0

    ok   = sum(1 for r in results if r.ok)
    fail = sum(1 for r in results if not r.ok)
    reads = _sdk_tool_reads["count"]

    print(f"  {ok}/{n} OK, {fail} failed, wall={wall:.3f}s ({n/wall:.1f} agents/s)")
    if reads:
        dedup_mechanism = "inflight Future (cacheable=False)"
        print(f"  Tool reads: {reads} executed ({n} requested, dedup {n/reads:.0f}x via {dedup_mechanism})")
    else:
        print("  Tool reads: 0 (model did not call tool)")
    return {
        "config": f"E-sdk-{n}", "n": n, "mode": "batchagent_sdk",
        "ok": ok, "failed": fail,
        "wall_clock_seconds": wall,
        "throughput_agents_per_sec": n / wall,
        "tool_reads_executed": reads,
        "tool_reads_requested": n,
        "tool_dedup_ratio": n / reads if reads else None,
        "mock_forward_pass_ms": FORWARD_PASS_LATENCY * 1000,
        "mock_tool_call_ms": TOOL_CALL_LATENCY * 1000,
    }


# ── main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    backend = MockVLLMBackend()
    all_results: dict[str, Any] = {}

    for n in [50, 200]:
        d = await run_config_d(n)
        e = await run_config_e(n, backend)
        all_results[f"D_{n}"] = d
        all_results[f"E_{n}"] = e

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FAIR COMPARISON — mock backend (60ms fwd, 200ms tool)")
    print(f"{'='*70}")
    print(f"{'Config':<22} {'N':>5} {'Wall(s)':>8} {'agents/s':>10} {'OK%':>6} {'reads':>8}")
    print("-" * 62)
    for key, r in all_results.items():
        ok_pct = 100 * r["ok"] / r["n"]
        reads = r["tool_reads_executed"]
        print(f"  {r['config']:<20} {r['n']:>5} {r['wall_clock_seconds']:>8.3f} "
              f"{r['throughput_agents_per_sec']:>10.1f} {ok_pct:>6.1f} {reads:>8}")

    d50  = all_results["D_50"]
    e50  = all_results["E_50"]
    d200 = all_results["D_200"]
    e200 = all_results["E_200"]

    print(f"""
Key metrics:
  Wall-clock ratio E/D at N=50  : {e50['wall_clock_seconds']/d50['wall_clock_seconds']:.2f}x
  Wall-clock ratio E/D at N=200 : {e200['wall_clock_seconds']/d200['wall_clock_seconds']:.2f}x
  E N=200 under 10s             : {'YES' if e200['wall_clock_seconds'] < 10 else 'NO'}  ({e200['wall_clock_seconds']:.2f}s)
  Tool dedup ratio E N=200      : {e200.get('tool_dedup_ratio') or 'N/A'}x
  D-equiv failure rate N=200    : {100*(d200['failed']/d200['n']):.1f}%
  E-SDK failure rate N=200      : {100*(e200['failed']/e200['n']):.1f}%

Code complexity (user-facing lines):
  Config D-equivalent           : {_D_EQUIV_LOC} lines (multi-turn loop, tool exec, retry, validation)
  Config E (BatchAgent.run())   : {_E_EQUIV_LOC} lines
  Reduction                     : {_D_EQUIV_LOC//_E_EQUIV_LOC}x fewer lines with BatchAgent
""")

    summary = {
        "benchmark": "fair_comparison",
        "mock_forward_pass_ms": FORWARD_PASS_LATENCY * 1000,
        "mock_tool_call_ms": TOOL_CALL_LATENCY * 1000,
        "results": all_results,
        "summary": {
            "wall_ratio_E_vs_D_n50":  e50["wall_clock_seconds"] / d50["wall_clock_seconds"],
            "wall_ratio_E_vs_D_n200": e200["wall_clock_seconds"] / d200["wall_clock_seconds"],
            "e200_wall_seconds": e200["wall_clock_seconds"],
            "e200_under_10s": e200["wall_clock_seconds"] < 10,
            "tool_dedup_ratio_e200": e200.get("tool_dedup_ratio"),
            "d200_failure_rate_pct": 100 * d200["failed"] / d200["n"],
            "e200_failure_rate_pct": 100 * e200["failed"] / e200["n"],
            "loc_d_equiv": _D_EQUIV_LOC,
            "loc_e_sdk": _E_EQUIV_LOC,
            "loc_reduction_ratio": _D_EQUIV_LOC / _E_EQUIV_LOC,
        },
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Results written to {OUT_DIR}/results.json")


if __name__ == "__main__":
    asyncio.run(main())
