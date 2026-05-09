# Batch Agent SDK

Run N LLM agents in parallel. Input list in, result list out. Every number on this page comes from a `tests/benchmarks/results/*/results.json` file or `LOGS.md`.

---

## When to use this

**Use BatchAgent when** you need to run the same LLM task against N inputs (10–500) and want:
- Results as they arrive, not after the slowest agent finishes
- Tool calls deduplicated across agents (N agents reading the same file → 1 actual read)
- Structured output validated, retried, and returned as data — not exceptions
- Priority scheduling that drains near-complete agents before starting fresh ones

**Do not use BatchAgent when** you have a single task, a single turn, no shared system prompt, or you are on a commercial API with strict rate limits. For those cases the raw `asyncio.gather` approach is simpler and faster.

---

## Install

```bash
pip install batch-agent            # core: httpx + pydantic
pip install "batch-agent[bedrock]" # + boto3 for AWS Bedrock
pip install "batch-agent[vllm]"    # + vllm for self-hosted inference
pip install "batch-agent[redis]"   # + redis for distributed mode (Phase 4)
```

Requires Python ≥ 3.10.

---

## Quickstart

```python
from batch_agent import BatchAgent, Tool
from pydantic import BaseModel

class PaperSummary(BaseModel):
    benchmark_name: str | None
    primary_metric: str
    models_tested: list[str]
    summary: str

results = await BatchAgent.run(
    system_prompt="You are a precise scientific summarizer.",
    task="Extract benchmark info from this paper:\n\n{paper_text}",
    inputs=[{"paper_text": text} for text in papers],
    tools=[Tool.read_file, Tool.web_search],
    output_schema=PaperSummary,
    model="meta-llama/Llama-3.1-70B-Instruct",
    backend="vllm://localhost:8000",
    max_inflight=64,
    max_turns=4,
    max_retries=2,
    on_result=lambda r: print(r.output),
)
```

Stream results as they arrive:

```python
async for result in BatchAgent.stream(task=..., inputs=...):
    if result.ok:
        process(result.output)   # Pydantic model, arrives as each agent finishes
    else:
        log_error(result.error)  # structured AgentError, not an exception
```

AWS Bedrock (no self-hosted GPU required):

```python
results = await BatchAgent.run(
    task="Summarize: {text}",
    inputs=[{"text": t} for t in texts],
    model="anthropic.claude-sonnet-4-20250514-v1:0",
    backend="bedrock://us-east-1",
    max_inflight=2,          # start low; AIMD controller raises it automatically
)
```

Reduce (map → aggregate):

```python
results, summary = await BatchAgent.run_with_reduce(
    task="Extract claims from: {text}",
    inputs=[{"text": t} for t in texts],
    reduce="You received {n} claim lists. Deduplicate and rank by evidence.",
    output_schema=ClaimList,
    reduce_schema=RankedClaimList,
)
```

---

## Benchmarks

All numbers are from live hardware runs or deterministic mock runs. Source files listed per table.

### Table 1 — Fair comparison: BatchAgent SDK vs naive asyncio.gather

*Mock backend: 60 ms per forward pass, 200 ms per tool call, `asyncio.sleep` (fully parallel). Source: `tests/benchmarks/results/fair_comparison/results.json`.*

Both configs do identical work: multi-turn loop, one tool call per agent, Pydantic output validation, retry on failure.

| Config | N | Wall (s) | agents/s | OK% | Tool reads | User code |
|---|---|---|---|---|---|---|
| D — naive `asyncio.gather` | 50 | 0.65 | 76.9 | 100% | 51 | 87 lines |
| **E — BatchAgent** | 50 | 3.46 | 14.5 | 100% | 2 (inflight dedup) | **9 lines** |
| D — naive `asyncio.gather` | 200 | 0.66 | 303 | 100% | 204 | 87 lines |
| **E — BatchAgent** | 200 | **3.48s** | 57.5 | 100% | 4 (inflight dedup) | **9 lines** |

Key findings:
- **9.7× fewer lines of user code** (87 vs 9, programmatically verified)
- **50× tool dedup** at N=200 (inflight Future mechanism, `cacheable=False`, no LRU)
- E wall-clock is ~5× higher because the scheduler does more per agent (retry, validation, KVFlow, state). For the equivalent raw throughput the naive approach is faster but requires the user to implement all of that themselves.

### Table 2 — Live GPU: A10G 23 GB, Qwen/Qwen2.5-7B-Instruct, vLLM 0.20.1

*Source: `tests/benchmarks/results/config_d_20/results.json`, `config_d_200/results.json`, `config_e_200/results.json`, `fair_comparison_live/results.json`.*

| Config | N | Wall (s) | agents/s | TTFT P50 | Tool reads | Cache hit% |
|---|---|---|---|---|---|---|
| D naive (single-turn, file in prompt) | 20 | 0.46 | 43.3 | 0.208s | 20 | 83.5% |
| D naive (single-turn) | 200 | 2.67 | 74.8 | 0.979s | 200 | 63.4% |
| **E BatchAgent** (2-turn + tool, old scheduler) | 200 | 36.5 | 5.5 | — | 1 (200x dedup) | 93.0% |
| **E BatchAgent** (2-turn + tool, backpressure dispatch) | 200 | **21.8s** | 9.2 | — | 200 | 90.8% |

Key findings:
- **40% wall-clock improvement** from backpressure dispatch fix (36.5s → 21.8s)
- **93.0% prefix cache hit rate** at N=200 (from `config_e_200/results.json`)
- **200x tool dedup** in the live GPU run (cacheable=True + inflight, all 200 agents dispatched simultaneously)
- D naive N=200 does not OOM or timeout — vLLM queues all 200 simultaneously
- TTFT P50 degrades 4.7× from N=20 to N=200 (0.208s → 0.979s) under naive gather due to queue depth

Note: E is slower per-agent than D on live GPU because E does 2 forward passes per agent (tool-call round-trip) while D does 1. The comparison is not identical work; see Table 1 for the controlled mock comparison.

---

## How it works

```
User: BatchAgent.run(task, inputs, tools, output_schema)
        │
        ▼
TaskCompiler → ExecutionPlan (shared prefix extracted, schema injected)
        │
        ▼
WaveScheduler (asyncio)
  ├─ PrioritySemaphore (max_inflight) — near-done agents served first
  ├─ BackpressureController — pauses dispatch when backend queue fills
  ├─ KVFlowAdvisor — emits prefetch hints to vLLM before agents reactivate
  └─ per-agent loop:
       turn 1: acquire semaphore → generate → release semaphore
       if tool_use: TOOL_WAIT (semaphore FREE) → ToolPool.call → inject result
       turn 2: acquire → generate → release
       → parse_and_validate_output(Pydantic) → AgentResult
        │
        ▼
ToolPool
  ├─ inflight Future dedup (N concurrent callers → 1 execution)
  ├─ LRU cache (cacheable=True tools)
  ├─ token-bucket rate limiter per tool
  └─ @batchable SQL grouping (N WHERE id=? → 1 IN (...))
        │
        ▼
BackendAdapter (Anthropic / OpenAI / vLLM / SGLang / Bedrock)
```

The semaphore wraps **only** the inference call, not tool waits. This is W5 from the spec — it is the reason GPU utilisation stays near 100% instead of collapsing when agents wait for external tools.

---

## Backends

From `BackendAdapter.backend_capabilities()`:

| Backend | URL format | prefix_pinning | kvflow | diff_kv | max_safe_concurrent |
|---|---|---|---|---|---|
| Anthropic API | `anthropic://` | ✗ | ✗ | ✗ | 5 |
| OpenAI API | `openai://host` | ✗ | ✗ | ✗ | 5 |
| **vLLM** (self-hosted) | `vllm://host:8000` | ✓ | ✓ | ✓ | 64 |
| **SGLang** (self-hosted) | `sglang://host:30000` | ✗ | ✓ | ✗ | 64 |
| AWS Bedrock | `bedrock://region/model` | ✗ | ✗ | ✗ | 1–3 |

**Bedrock-specific notes** (from `tests/benchmarks/results/bedrock_cache_isolation/results.json`):
- Prompt caching (`cachePoint`) is confirmed active when system prompt ≥ ~1,024 tokens
- Prompt caching **saves tokens but not latency** at <8K token prompts — Bedrock managed queue/model latency dominates prefill savings (confirmed across 3 isolation variants: sequential, region-swap, parallel)
- Default concurrency = 1; the AIMD controller increases it automatically after 60 s without throttling
- Bedrock mode value: tool deduplication, structured output validation, retry handling, prompt cache management — not GPU scheduling efficiency

---

## Limitations

These are not on the roadmap to soften — they are facts about the current implementation.

1. **vLLM/GPU results are for 7B models on a single A10G.** The throughput numbers in Table 2 are ~10 agents/sec for 2-turn tasks. For 70B models or larger batches a different GPU configuration is needed. The 70B benchmark has not been run; publish is blocked on it.

2. **Tool dedup only fires for concurrent callers.** If N agents call the same tool but complete their first forward pass at different times (as happens on real GPU with sequential-ish inference), each agent may see a different inflight window and execute the tool independently. The 200x dedup in the GPU run required cacheable=True. The mock shows 50x with cacheable=False because the mock is fully parallel.

3. **Bedrock TTFT does not improve with prompt caching at <8K tokens.** Cache-miss P50: 2.32s. Cache-hit P50: 3.24s. Hit/miss ratio: 1.40 (hit is slower, not faster). This is reproducible across 10 sequential identical requests. Source: `tests/benchmarks/results/bedrock_cache_isolation/results.json`.

4. **Distributed mode (Phase 4) is a prototype.** The `RedisStreamsStateStore` and `DistributedWaveScheduler` are implemented and unit-tested with a mock Redis, but have not been tested against a real Redis cluster. The 1,000-agent benchmark requires 4 nodes and has not been run.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 0 — Foundation | ✅ done | Multi-turn loop, W5 semaphore fix, tool coalescing, Anthropic backend |
| 1 — Inference integration | ✅ done | vLLM native mode, prefix warming, priority queue |
| 2 — Scale + robustness | ✅ done | 500-agent benchmark, retry, compaction, checkpointing, reduce |
| 3A — KVFlow prefetch | ✅ done (mock) | KVFlowAdvisor, backpressure dispatch; vLLM patch route written, GPU test pending |
| 3B — TokenDance diff KV | ✅ done (mock) | 18.76× compression in synthetic test; vLLM patch not deployed |
| 3C — SGLang backend | ✅ done (mock) | Full adapter; live GPU test pending |
| 4 — Distributed | ✅ prototype | Mock Redis; real cluster test and 1,000-agent benchmark pending |
| **Publish** | **blocked** | Waiting on: 70B GPU benchmark, live vLLM KVFlow measurement, cost-per-task comparison vs naive API |
