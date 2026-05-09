# Batch Agent SDK

The orchestration layer for production multi-agent systems. Compatible with vLLM, SGLang, and NVIDIA Dynamo.

Run N LLM agents in parallel. Input list in, structured result list out. Results stream as each agent finishes — you don't wait for the slowest one.

```python
from batch_agent import BatchAgent, Tool
from pydantic import BaseModel

class ResearchPlan(BaseModel):
    items: list[str]

class ResearchAnswer(BaseModel):
    question: str
    answer: str

class SurveyPaper(BaseModel):
    title: str
    abstract: str
    sections: list[dict]

# Generate a 20-question survey paper on any topic
results, paper = await BatchAgent.run_with_map_reduce(
    plan_prompt="Generate 20 research questions about: {topic}",
    plan_inputs={"topic": "transformer attention optimization"},
    plan_output_schema=ResearchPlan,
    task="Research this question: {item}",
    output_schema=ResearchAnswer,
    reduce="Synthesize into a survey paper",
    reduce_schema=SurveyPaper,
    tools=[Tool.web_search, Tool.claude_code],
    model="gpt-5.5",
    backend="openai://",
)
# 20 parallel research agents -> 1 synthesized paper
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

## AutoResearch

`run_with_map_reduce()` is a three-tier topology:

```
planner agent -> N research agents -> reducer agent
```

The planner produces `items: list[str]`, each map agent receives one `{item}`, and the reducer receives every map result plus structured error entries for failures. The map stage uses the normal scheduler: tool deduplication, streaming tool dispatch, retries, checkpoints, KVFlow hints, and result streaming.

Run the example:

```bash
python examples/auto_research.py \
  --topic "KV cache optimization for multi-agent LLM inference" \
  --n-questions 20 \
  --backend openai:// \
  --output examples/output/kv_cache_survey.md
```

For `backend=openai://`, the example defaults planner, worker, and reducer agents to `gpt-5.5`. This is forced for private-access testing; override with `--model`, `--planner-model`, `--worker-model`, or `--reducer-model` if the API rejects that model ID.

Live AutoResearch cost/timing is not reported yet because the validation run was blocked by Anthropic API credits and no OpenAI validation run has been completed.

---

## NVIDIA Dynamo Compatibility

Use Dynamo through the OpenAI-compatible endpoint:

```python
results = await BatchAgent.run(
    task="Analyze {item}",
    inputs=[{"item": x} for x in items],
    output_schema=Answer,
    model="meta-llama/Llama-3.1-70B-Instruct",
    backend="dynamo://localhost:8000",
    nvext_agent_hints=True,
)
```

When enabled, BatchAgent attaches `nvext.agent_hints` with latency sensitivity, priority, speculative prefill, and output-length estimates derived from scheduler state. Dynamo-native `tool_call_dispatch` SSE events are parsed by the streaming dispatch path so tools can start before the full model response completes.

---

## Tool.claude_code

`Tool.claude_code` lets each batch agent launch a Claude Code subagent:

```python
results = await BatchAgent.run(
    task="Review this module: {path}",
    inputs=[{"path": p} for p in paths],
    tools=[Tool.claude_code, Tool.read_file],
    output_schema=Review,
    model="claude-sonnet-4-6",
    backend="anthropic://",
)
```

It requires the `claude` CLI plus either `ANTHROPIC_API_KEY` or a Claude subscription. BatchAgent strips session-variant preamble headers before prefix hashing by default, so Claude Code billing headers do not poison shared-prefix caching.

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

### v2 mock benchmark — W15/W16/Dynamo-era code

Source: `tests/benchmarks/results/v2_benchmark/results.json`

| Config | N | Wall | agents/s | Cache hit% | Note |
|---|---:|---:|---:|---:|---|
| D | 50 | 0.185s | 270.3 | 78.0% | naive one-turn mock |
| D | 100 | 0.310s | 322.6 | 86.0% | naive one-turn mock |
| D | 200 | 0.560s | 357.1 | 91.0% | naive one-turn mock |
| E | 50 | 1.159s | 43.1 | 96.8% | BatchAgent native mock with W15/W16 |
| E | 100 | 2.059s | 48.6 | 96.8% | BatchAgent native mock with W15/W16 |
| E | 200 | 3.859s | 51.8 | 96.8% | BatchAgent native mock with W15/W16 |
| F | 50 | 1.159s | 43.1 | 96.8% | KVFlow hints only; prefetch blocked |
| F | 100 | 2.059s | 48.6 | 96.8% | KVFlow hints only; prefetch blocked |
| F | 200 | 3.859s | 51.8 | 96.8% | KVFlow hints only; prefetch blocked |

The v2 mock records `0.101s` streaming tool-dispatch overlap and `96.8%` cache hit rate with billing-header stripping. It does **not** claim KVFlow prefetch speedup; F is equal to E until vLLM scheduler integration is implemented.

---

## Cost

Source: `tests/benchmarks/results/cost_comparison/results.json`

Cost comparison for N=100, input=3000 tokens, output=500 tokens, model=sonnet-4.6

| Mode | Cost / batch | Relative | Note |
|---|---:|---:|---|
| Naive API | $1.6500 | 1.000x | Parallel API calls, no cache discount. |
| Anthropic Batch API | $0.8250 | 0.500x | No tool calls, single turn only. |
| BatchAgent + API caching | $0.8659 | 0.525x | Uses measured 96.8% cache hit rate. |
| BatchAgent + self-hosted vLLM | $0.0041 | 0.002x | L4 at $0.805/hr, 19800 agents/hr. |
| BatchAgent + NVIDIA Dynamo | $0.0041 | 0.002x | Same L4 cost model as vLLM; nvext_agent_hints benefit is scheduling, not pricing. |

Batch API cost assumes single-turn, no tool calls. BatchAgent supports multi-turn tool calls at the caching price.

AutoResearch 20-question paper cost is not reported yet: the live Step 5 run was blocked by missing Anthropic and search API credentials, so `results.json` records it as blocked rather than estimated.

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

| Backend | URL | prefix_pinning | kvflow | diff_kv | nvext hints | max_safe_concurrent |
|---|---|---|---|---|---|---|
| Anthropic API | `anthropic://` | — | — | — | — | 5 |
| OpenAI API | `openai://host` | — | — | — | — | 5 |
| **vLLM** | `vllm://host:8000` | yes | hints only; scheduler integration pending | yes | — | 64 |
| **SGLang** | `sglang://host:30000` | — | yes | — | — | 64 |
| **NVIDIA Dynamo** | `dynamo://host:8000` | yes | via scheduler hints | yes | yes | 64 |
| AWS Bedrock | `bedrock://region` | — | — | — | — | 1-3 (AIMD) |

**Bedrock notes** — from `tests/benchmarks/results/bedrock_cache_isolation/results.json`:
- Prompt caching confirmed active when system prompt ≥ ~1,024 tokens
- Token savings confirmed; **latency savings not confirmed** at <8K token prompts (cache-miss P50: 2.32s, cache-hit P50: 3.24s)
- Concurrency starts at 1; AIMD controller raises it after 60s quiet

---

## Limitations

1. **GPU numbers are for 7B on a single A10G.** ~10 agents/sec for 2-turn tasks. 70B or multi-node results pending.

2. **Tool dedup only fires for concurrent callers.** On a real GPU with sequential-ish inference, each agent's turn-1 completes at different times. The 200× dedup in the GPU table required `cacheable=True`. With `cacheable=False` the mock shows 50× because the mock is fully parallel.

3. **Bedrock TTFT does not improve with prompt caching at <8K tokens.** Confirmed across 10 sequential identical requests. Cache writes tokens, not latency savings at this scale. Source: `bedrock_cache_isolation/results.json`.

4. **KVFlow prefetch is not yet verified.** Measurement-integrity rerun on A10G + Qwen2.5-7B produced A/B/C turn-2 TTFT P50 of `2.626960s` / `2.846764s` / `3.031180s`; the corrected patch moved `0` block pairs because vLLM 0.6.6 requires scheduler-owned CPU→GPU swap mappings, not `kv_key`-only hints. Source: `kvflow_measurement_integrity/results.json`.

5. **Distributed mode is a prototype.** `RedisStreamsStateStore` and `DistributedWaveScheduler` are tested with a mock Redis, not a real cluster. 1,000-agent benchmark requires 4 nodes and has not been run.

---

## Roadmap

| Phase | Status | Notes |
|---|---|---|
| 0 — Foundation | ✅ | Multi-turn loop, W5 semaphore fix, tool coalescing |
| 1 — Inference | ✅ | vLLM native, prefix warming, priority queue |
| 2 — Scale | ✅ | 500-agent benchmark, retry, compaction, checkpointing, reduce |
| 3A — KVFlow | ⚠️ blocked | Advisor emits hints, but vLLM prefetch needs scheduler integration; A/B/C measurement did not confirm prefetch benefit |
| 3B — TokenDance | ✅ mock | 18.76× compression in synthetic test; live vLLM patch pending |
| 3C — SGLang | ⚠️ mock | Parser fix covered locally; live GPU rerun still unresolved |
| 4 — Distributed | ⏳ prototype | Mock Redis tested; real cluster + 1,000-agent benchmark pending |
| 5 — Dynamo + AutoResearch | ✅ mock | Dynamo hints, streaming dispatch, map-reduce, and example; live demo blocked by credentials |
| **Publish** | **blocked** | Blocked on: 70B benchmark and final publish decision |
