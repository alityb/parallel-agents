# Batch Agent SDK

Run N LLM agents in parallel. Input list in, structured result list out. Results stream as each agent finishes — you don't wait for the slowest one.

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
    task="Summarize this paper:\n\n{paper_text}",
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

---

## When to use this

**Use BatchAgent when** you need to run the same task against N inputs and want:
- Results as they arrive, not after the slowest agent
- Tool calls deduplicated — N agents reading the same file trigger 1 actual read
- Structured output validated and retried, returned as data not exceptions
- Priority scheduling that drains near-complete agents first

**Don't use it when** you have a single task, no shared system prompt, or an API-rate-limited backend. For those cases `asyncio.gather` is simpler and faster.

---

## Install

```bash
pip install batch-agent                # core: httpx + pydantic
pip install "batch-agent[bedrock]"     # + boto3 for AWS Bedrock
pip install "batch-agent[vllm]"        # + vllm for self-hosted inference
pip install "batch-agent[dashboard]"   # + rich for live terminal dashboard
pip install "batch-agent[redis]"       # + redis for distributed mode (Phase 4)
```

Requires Python ≥ 3.10.

---

## Quickstart

```python
# Stream results as they arrive
async for result in BatchAgent.stream(task=..., inputs=...):
    if result.ok:
        process(result.output)   # Pydantic model
    else:
        log_error(result.error)  # AgentError, not an exception

# AWS Bedrock
results = await BatchAgent.run(
    task="Summarize: {text}",
    inputs=[{"text": t} for t in texts],
    model="anthropic.claude-sonnet-4-20250514-v1:0",
    backend="bedrock://us-east-1",
    max_inflight=2,   # AIMD controller raises this automatically
)

# Reduce (map → aggregate)
results, summary = await BatchAgent.run_with_reduce(
    task="Extract claims from: {text}",
    inputs=[{"text": t} for t in texts],
    reduce="You received {n} claim lists. Deduplicate and rank by evidence.",
    output_schema=ClaimList,
    reduce_schema=RankedClaimList,
)
```

Live dashboard:

```bash
batch-agent run --spec my_task.json --dashboard
```

---

## Benchmarks

All numbers are from live hardware runs or deterministic mock runs with confirmed source files.

### Mock benchmark — fair comparison (controlled)

Both configs do the same work: 2-turn loop, one tool call per agent, Pydantic validation, retry on failure. Mock backend: 60ms forward pass, 200ms tool call.
Source: `tests/benchmarks/results/fair_comparison/results.json`

| Config | N | Wall | agents/s | Tool reads | User code |
|---|---|---|---|---|---|
| D — naive `asyncio.gather` | 200 | 0.66s | 303 | 204 | **87 lines** |
| **E — BatchAgent** | 200 | **3.48s** | 57.5 | 4 (50× dedup) | **9 lines** |

E is ~5× slower per-agent because the scheduler does more work per agent. The point is the 9.7× code reduction and the dedup — not wall-clock.

### Live GPU — A10G 23 GB, Qwen/Qwen2.5-7B, vLLM 0.20.1

Sources: `config_d_20`, `config_d_200`, `config_e_200`, `fair_comparison_live` result files.

| Config | N | Wall | agents/s | TTFT P50 | Cache hit% |
|---|---|---|---|---|---|
| D naive (1-turn) | 20 | 0.46s | 43.3 | 0.208s | 83.5% |
| D naive (1-turn) | 200 | 2.67s | 74.8 | 0.979s | 63.4% |
| **E BatchAgent** (2-turn + tool, old scheduler) | 200 | 36.5s | 5.5 | — | 93.0% |
| **E BatchAgent** (2-turn + tool, backpressure) | 200 | **21.8s** | 9.2 | — | 90.8% |

- Backpressure dispatch: **40% improvement** over the old wave-gated scheduler (36.5s → 21.8s)
- Prefix cache at N=200: **93.0%** hit rate
- D naive N=200: did not OOM — vLLM queued all 200 simultaneously
- TTFT P50 degradation N=20→200: **4.7×** (0.208s → 0.979s) from queue depth

---

## How it works

```
BatchAgent.run(task, inputs, tools, output_schema)
    │
    ├─ TaskCompiler   extracts shared prefix, injects schema, hoists constants
    │
    ├─ WaveScheduler  asyncio event loop
    │    ├─ PrioritySemaphore   near-done agents served before fresh ones
    │    ├─ BackpressureController   pauses dispatch when backend queue fills
    │    ├─ KVFlowAdvisor   emits prefetch hints 500ms before agents reactivate
    │    └─ per-agent loop:
    │         acquire semaphore → generate → release semaphore
    │         if tool_use: TOOL_WAIT (semaphore FREE) → ToolPool → inject result
    │         → parse_and_validate_output → AgentResult
    │
    └─ ToolPool
         ├─ inflight Future dedup (N concurrent callers → 1 execution)
         ├─ LRU cache per cacheable tool
         ├─ token-bucket rate limiter per tool
         └─ @batchable SQL grouping
```

The semaphore wraps **only** inference calls, not tool waits. This is the W5 invariant — it keeps GPU utilisation near 100% regardless of tool latency.

---

## Backends

Capabilities come from `BackendAdapter.backend_capabilities()`:

| Backend | URL | prefix_pinning | kvflow | diff_kv | max_safe_concurrent |
|---|---|---|---|---|---|
| Anthropic API | `anthropic://` | — | — | — | 5 |
| OpenAI API | `openai://host` | — | — | — | 5 |
| **vLLM** | `vllm://host:8000` | ✓ | ✓ | ✓ | 64 |
| **SGLang** | `sglang://host:30000` | — | ✓ | — | 64 |
| AWS Bedrock | `bedrock://region` | — | — | — | 1–3 (AIMD) |

**Bedrock notes** — from `tests/benchmarks/results/bedrock_cache_isolation/results.json`:
- Prompt caching confirmed active when system prompt ≥ ~1,024 tokens
- Token savings confirmed; **latency savings not confirmed** at <8K token prompts (cache-miss P50: 2.32s, cache-hit P50: 3.24s)
- Concurrency starts at 1; AIMD controller raises it after 60s quiet

---

## Limitations

1. **GPU numbers are for 7B on a single A10G.** ~10 agents/sec for 2-turn tasks. 70B or multi-node results pending.

2. **Tool dedup only fires for concurrent callers.** On a real GPU with sequential-ish inference, each agent's turn-1 completes at different times. The 200× dedup in the GPU table required `cacheable=True`. With `cacheable=False` the mock shows 50× because the mock is fully parallel.

3. **Bedrock TTFT does not improve with prompt caching at <8K tokens.** Confirmed across 10 sequential identical requests. Cache writes tokens, not latency savings at this scale. Source: `bedrock_cache_isolation/results.json`.

4. **Distributed mode is a prototype.** `RedisStreamsStateStore` and `DistributedWaveScheduler` are tested with a mock Redis, not a real cluster. 1,000-agent benchmark requires 4 nodes and has not been run.

---

## Roadmap

| Phase | Status | Notes |
|---|---|---|
| 0 — Foundation | ✅ | Multi-turn loop, W5 semaphore fix, tool coalescing |
| 1 — Inference | ✅ | vLLM native, prefix warming, priority queue |
| 2 — Scale | ✅ | 500-agent benchmark, retry, compaction, checkpointing, reduce |
| 3A — KVFlow | ✅ mock / ⏳ GPU | Advisor + backpressure working; GPU KV prefetch needs vLLM patch |
| 3B — TokenDance | ✅ mock | 18.76× compression in synthetic test; live vLLM patch pending |
| 3C — SGLang | ✅ mock | Full adapter; live GPU test pending |
| 4 — Distributed | ⏳ prototype | Mock Redis tested; real cluster + 1,000-agent benchmark pending |
| **Publish** | **blocked** | Blocked on: 70B benchmark, live KVFlow measurement, cost comparison |
