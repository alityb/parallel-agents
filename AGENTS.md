# agents.md — Batch Agent SDK: Build Specification

> **Purpose:** This document is the authoritative build spec for the Batch Agent SDK.
> It is written to be read by an engineer starting from zero. Every design decision
> is stated, every weakness is named, every iteration is explicit.
> If something is ambiguous here, it is not yet designed.
>
> **Last updated: May 2026 (post-Phase 2)**
> Phase 0, 1, 2 complete. Phase 3 redesigned to include KVFlow-style prefetching
> as a parallel track alongside TokenDance diff storage. Phase 4 added (distributed).

---

## 0. What We Are Building

A Python SDK + server process that lets you do this:

```python
from batch_agent import BatchAgent, Tool

results = await BatchAgent.run(
    task="Summarize this paper and extract: benchmark name, primary metric, models tested.\n\nPaper: {paper_text}",
    inputs=[{"paper_text": text} for text in papers],
    tools=[Tool.read_file, Tool.web_search],
    output_schema=PaperSummary,
    model="meta-llama/Llama-3.1-70B-Instruct",
    backend="vllm://localhost:8000",
    max_concurrent=64,
    on_result=lambda r: print(r),
)
```

And get back a list of `PaperSummary` objects, one per input, in order.
Time-to-first-result: seconds. Time-to-all: proportional to longest agent, not sum.

**What this is not:** A general agent framework. Not LangChain. Not an agent OS.
This is a batch execution engine. Input list in, result list out, as fast as physics allows.

---

## 1. Principles

1. **Co-design the orchestration and inference layers.** Any design that treats the inference backend as a black box HTTP endpoint is leaving 60–70% of the performance on the table. The scheduler must know what the GPU is doing. The GPU must know what the scheduler is planning.

2. **Shared prefix is gold.** The system prompt is identical across all N agents. Computing its KV cache once and sharing it is the single highest-leverage optimization. Everything else is secondary.

3. **Agents finish at different times. Respect that.** Don't batch-wait for the slowest agent. Stream results. Free KV slots as agents complete. Next wave fills immediately.

4. **Failures are not exceptions. They are data.** At 500+ agents, some will fail. The SDK handles this with retry + structured error result. The caller never writes try/catch loops.

5. **The user writes a task template, not an agent.** The SDK is a batch execution engine that happens to use agents internally. Mental model: input list in, result list out.

6. **Self-hosted first, API-compatible second.** The deep optimizations require controlling the inference layer. Commercial API support degrades gracefully to "parallel API calls with rate limiting."

7. **The scheduler knows the future. Use it.** The Wave Scheduler knows which agents are about to activate, which are in TOOL_WAIT, and which are nearly complete. This information is enormously valuable to the inference layer — for prefetching KV tensors, for eviction priority, for slot allocation. Pass it down. This is the KVFlow principle and it is now a first-class design constraint.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          User Process                             │
│                     BatchAgent.run(...)                           │
└──────────────────────────────┬───────────────────────────────────┘
                               │ asyncio / gRPC
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Orchestration Server                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│  │ Task Compiler │  │  Wave Scheduler   │  │  Result Streamer  │   │
│  └──────┬───────┘  └────────┬─────────┘  └─────────┬─────────┘   │
│         │                   │                        │             │
│         │          ┌────────▼─────────┐              │             │
│         │          │  KVFlow Advisor   │◄─────────────┘             │
│         │          │ (prefetch hints)  │                            │
│         │          └────────┬─────────┘                            │
│         │                   │  "agent_47 activates in ~2s"         │
│  ┌──────▼───────────────────▼──────────────────────────────────┐  │
│  │                     Agent State Store                         │  │
│  │  (per-agent: messages, tool results, status, kv_key,         │  │
│  │   estimated_next_activation, steps_to_execution)             │  │
│  └──────────────────────────┬────────────────────────────────── ┘  │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                   Tool Execution Pool                          │   │
│  │  (deduplicated, batched, rate-limited, predictive pre-warm)   │   │
│  └──────────────────────────┬────────────────────────────────── ┘   │
└─────────────────────────────│───────────────────────────────────────┘
                              │ vLLM OpenAI-compat HTTP
                              │ + /internal/prefetch endpoint
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Inference Adapter                           │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐  │
│  │  Prefix Registry  │  │   KV Diff Encoder (TokenDance)       │  │
│  └──────────────────┘  └──────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │   KVFlow Prefetch Controller                                │   │
│  │   (moves KV tensors CPU→GPU before agent activates)        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │         vLLM / SGLang (self-hosted)                         │   │
│  │    OR   Anthropic / OpenAI API (degraded mode)              │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**New in this revision:** The KVFlow Advisor sits inside the orchestration layer and maintains a real-time map of `{job_id → estimated_next_activation_time}`. It pushes prefetch hints to the inference adapter continuously. The adapter moves KV tensors from CPU to GPU *before* an agent reactivates, eliminating cold KV reloads.

---

## 3. Component Specs

### 3.1 Task Compiler

**Input:** A `BatchSpec` — task template, input list, tools, output schema, model, backend config.

**Output:** An `ExecutionPlan` — list of `AgentJob` objects plus `SharedContext`.

**Responsibilities:**

1. **Extract the shared prefix.** `system_prompt` (always shared) is separated from `task_template` (per-agent). Shared variables detected across all inputs are auto-hoisted into the system prompt. The shared prefix is registered with the inference adapter *once* before any agents dispatch.

2. **Validate the output schema.** Pydantic model → JSON Schema injected into each agent's system prompt: "Your final message must be a valid JSON object matching this schema."

3. **Estimate per-agent token budget.** Agents exceeding `model.max_context - min_response_tokens` are flagged `OVERSIZED` and routed to a chunking sub-task.

4. **Build the DAG.** Default topology is flat. `reduce` function appends a `ReduceJob` node. `map_reduce` spec supports multi-level DAGs (map → partial reduce → final reduce). Foundation for hierarchical agent topologies.

5. **Annotate jobs with tool call predictions.** If a prior run exists in the checkpoint store, extract the historical tool call sequence and attach as `predicted_tool_sequence`. The KVFlow Advisor uses this to pre-warm tool results before the agent asks. Cold start: no prediction. Warm start: rerun with prediction.

---

### 3.2 Wave Scheduler

Core orchestration engine. Asyncio event loop.

**State machine per agent:**

```
PENDING → PREFLIGHT → RUNNING → TOOL_WAIT → RUNNING → ... → COMPLETE | FAILED
                                     ↑
                              semaphore released HERE (W5)
                              semaphore re-acquired before next RUNNING
```

**Wave logic:**

```python
async def run_wave(plan: ExecutionPlan, max_concurrent: int):
    semaphore = asyncio.Semaphore(max_concurrent)
    queue = PriorityQueue()

    for job in plan.jobs:
        queue.put((estimate_turns(job), job))

    async def run_job(job):
        while job.status != COMPLETE and job.turn < job.max_turns:
            async with semaphore:                    # wraps ONLY inference
                response = await backend.generate(job.state.messages)

            if response.tool_calls:
                job.status = TOOL_WAIT               # semaphore is FREE here
                kvflow_advisor.update(job, TOOL_WAIT)
                results = await tool_pool.execute_all(response.tool_calls)
                job.state.messages.append(results)
                kvflow_advisor.update(job, PENDING_REACTIVATION)
            else:
                job.status = COMPLETE
                result_queue.put(job)

    await asyncio.gather(*[asyncio.create_task(run_job(j)) for j in queue])
```

**Priority scoring:**
- Base: `turns_remaining` (lower = higher priority).
- Bonus for agents with short `predicted_tool_sequence`.
- Priority boost for agents that have been in `TOOL_WAIT` longer than their historical average.

**Staggered cold start:** Dispatch first `max_concurrent` agents immediately. As each enters `TOOL_WAIT` or `COMPLETE`, dispatch one new `PENDING` agent. Continuous flow, not burst-idle-burst.

**Adaptive concurrency:**
- Monitor `prefix_cache_hit_rate` from the inference adapter every 10 seconds.
- Hit rate < 90%: reduce `max_concurrent` by 10% (memory pressure).
- Hit rate > 98% and GPU utilization < 80%: increase `max_concurrent` by 10%.
- Simple proportional controller. No vLLM changes required.

---

### 3.3 KVFlow Advisor *(new component)*

The bridge between the scheduler's knowledge of the future and the inference engine's need to manage memory. A single asyncio task inside the orchestration server.

**What it knows (from the Wave Scheduler):**
- Which agents are in `TOOL_WAIT` and their expected tool completion times
- Which agents are `PENDING` and their estimated first-activation time
- Which agents are nearly complete
- Historical tool latency distribution per tool type (from the Tool Pool)

**What it computes:** `steps_to_execution` per agent:

```python
def estimate_steps_to_execution(job: AgentJob) -> float:
    tool_latencies = [tool_pool.p75_latency(tc.tool) for tc in job.pending_tool_calls]
    return max(tool_latencies)  # parallel tool execution, use P75 not P50 (conservative)
```

**What it emits:** Every 500ms, a prefetch hint batch to the inference adapter:

```python
hints = [
    PrefetchHint(job_id="job_47", kv_key=state.kv_key, priority=1.2, eta_seconds=0.8),
    PrefetchHint(job_id="job_103", kv_key=state.kv_key, priority=0.9, eta_seconds=2.1),
]
await adapter.send_prefetch_hints(hints)
```

The adapter translates these into vLLM/SGLang prefetch calls that move KV tensors from CPU to GPU ahead of time.

**Why this matters:** Without KVFlow, when an agent exits `TOOL_WAIT`, vLLM may need to reload its KV tensors from CPU (~50–200ms depending on context length). With KVFlow, those tensors are already on GPU. The KVFlow paper reports 1.83x speedup for single workflows and 2.19x for concurrent workflows vs SGLang. This is achievable without patching vLLM's allocation logic — it requires only a prefetch API endpoint.

**Implementation path:** vLLM does not have a public prefetch API. We add one: a `/internal/prefetch` route that accepts `(kv_key, priority)` pairs and triggers async CPU→GPU transfers. This reuses existing vLLM swap infrastructure (~100 lines of new code). SGLang's RadixAttention may support this more naturally via its existing token-level tree structure. Prototype both; choose the one that works first.

---

### 3.4 Tool Execution Pool

**Request Coalescing:** First agent to request a tool call creates a Future. All other agents awaiting the same call await that Future. Tool executes once. All agents receive the same result string.

**Tool result caching:** In-process LRU with configurable TTL. Web search: 60s. File reads: until mtime changes.

**Rate limiting:** Token bucket per tool type. Web search: 10/sec. SQL: 50/sec.

**Batching for DB tools:** `@batchable(key_arg="id")` enables automatic `IN (...)` batching.

**Predictive pre-warming *(new)*:** When the Task Compiler attaches a `predicted_tool_sequence` (from checkpoint history), the Tool Pool pre-executes those calls *before the agent asks*, caching results. The agent gets a zero-latency cache hit. Only enabled for tools annotated `@cacheable=True` (deterministic tools only — never `web_search`).

**Tool latency tracking:** P50/P75/P99 per tool type. Feeds the KVFlow Advisor's ETA estimates.

---

### 3.5 Inference Adapter

Two modes: **native** (self-hosted vLLM/SGLang) and **API** (Anthropic/OpenAI, degraded).

#### 3.5.1 Prefix Registry

Before the first wave:

```python
async def warm_prefix(shared_prefix: str) -> str:
    await vllm_client.post("/v1/completions", json={
        "model": model, "prompt": shared_prefix, "max_tokens": 0,
    })
    # Immediately pin the prefix block — never evict
    await vllm_internal.pin_blocks(compute_prefix_hash(shared_prefix))
    return compute_prefix_hash(shared_prefix)
```

Pinning requires a one-line patch to vLLM's `BlockManager`. Pinned blocks survive LRU eviction.

**Target impact:** 2,048-token system prompt on 70B model: TTFT drops from ~800ms to ~40ms. At N=500: ~380 GPU-seconds of prefill compute saved.

#### 3.5.2 KVFlow Prefetch Controller *(new)*

```python
async def handle_prefetch_hints(hints: list[PrefetchHint]):
    hints.sort(key=lambda h: (-h.priority, h.eta_seconds))
    for hint in hints:
        if hint.eta_seconds < prefetch_horizon:   # default: 2.0s
            await vllm_internal.prefetch_kv_blocks(
                kv_key=hint.kv_key,
                destination="gpu",
                async_transfer=True,   # non-blocking
            )
```

**vLLM implementation:** Add `/internal/prefetch` to vLLM's API server. Route calls `cache_engine.prefetch(block_ids, destination="gpu")`. `CacheEngine` already has CPU↔GPU transfer infrastructure for swap operations — we reuse it.

**SGLang alternative:** SGLang's RadixAttention tracks token-level prefix trees. Prefetching for a specific agent means prefetching the leaf node of its context tree. Prototype both vLLM and SGLang paths; pick whichever is less invasive.

#### 3.5.3 KV Diff Encoding (TokenDance)

After turn 1, each agent's context diverges. TokenDance approach:
- 1 master KV block (shared system prompt)
- N sparse diff blocks (each agent's unique context delta, encoded as block-sparse differences)

**Target:** 11–17x KV compression per agent. 2.7x more concurrent agents than stock vLLM prefix caching.

**Implementation path:**
- Requires a custom `DiffCacheEngine` subclass of vLLM's `CacheEngine`.
- The `All-Gather` step: after each turn, hash all agent KV blocks, deduplicate shared blocks, encode diffs.
- v0: simulate at orchestration layer (track identical blocks via content hashing, skip re-sending). No vLLM changes.
- v1: patch vLLM `CacheEngine` directly. Pin to specific vLLM version.
- Gate behind `diff_kv=True`. Default off.

**Relationship to KVFlow:** Complementary, not competing. KVFlow reduces KV reload *latency* (prefetch). TokenDance reduces KV storage *volume* (compression). Both enabled simultaneously in Phase 3.

#### 3.5.4 API Mode (Degraded)

Anthropic `cache_control` headers, OpenAI implicit caching. No wave scheduling (rate limits are the binding constraint). Tool deduplication still works. KV prefetch and diff encoding: not applicable.

---

### 3.6 Agent State Store

```python
@dataclass
class AgentState:
    job_id: str
    status: AgentStatus
    turn: int
    messages: list[Message]
    tool_calls_pending: list[ToolCall]
    tool_results: list[ToolResult]
    kv_key: str | None
    output: Any | None
    error: AgentError | None
    retry_count: int
    created_at: float
    last_updated: float
    # KVFlow additions:
    estimated_next_activation: float | None    # unix timestamp
    steps_to_execution: float | None           # seconds
    predicted_tool_sequence: list[str] | None  # from checkpoint history
    historical_turn_latencies: list[float]     # for ETA estimation
```

Single-machine: in-process dict. Distributed (Phase 4): Redis Streams (append-only, survives partial failures).

**Message compaction:** After every 3 turns, tool results older than 2 turns are summarized via a small model call (Llama-3.2-3B), routed through a separate low-priority queue. Phase 2 implemented heuristic truncation. Model-based compaction is the Phase 3 target.

---

### 3.7 Result Streamer

```python
async for result in BatchAgent.stream(...):
    print(result.job_id, result.output)   # arrives as each agent finishes
```

Internal `asyncio.Queue`. Wave Scheduler pushes to it as agents complete. `on_result` callback fires immediately per result. Full list returned at end.

---

## 4. Implementation Plan

### Phase 0 — Foundation ✅ COMPLETE

Multi-turn agent loop, W5 semaphore fix, tool coalescing, structured output validation, streaming results, Anthropic API backend.

### Phase 1 — Inference Integration ✅ COMPLETE

vLLM native mode, prefix warming, priority queue, staggered dispatch, heterogeneous scheduling test.

### Phase 2 — Scale + Robustness ✅ COMPLETE

500-agent benchmark (mocked), retry logic, message compaction (heuristic), `@batchable` tools, `reduce` topology, SQLite checkpointing, configurable timeouts.

**Known gaps from Phase 2:**
- Model-based compaction not implemented (heuristic truncation only)
- Live Anthropic test blocked by API credits
- Live vLLM test blocked by GPU availability
- `predicted_tool_sequence` not yet populated

---

### Phase 3A — KVFlow Prefetch Integration (Week 9–11)

**Goal:** The scheduler tells the GPU what's coming. The GPU acts on it.

**Why 3A before 3B:** KVFlow requires a smaller vLLM surface change (add prefetch API) than TokenDance (subclass CacheEngine). Delivers measurable speedup independently. Complete 3A first; 3B builds on the same infrastructure.

Tasks:
- [ ] Implement `KVFlowAdvisor`: maintain `steps_to_execution` per agent, emit hints every 500ms
- [ ] Add `estimated_next_activation` and `historical_turn_latencies` to `AgentState`
- [ ] Instrument turn latency tracking in Wave Scheduler
- [ ] Add `/internal/prefetch` route to vLLM (~100 lines, reuse swap infrastructure)
- [ ] Implement `KVFlowPrefetchController` in vLLM adapter
- [ ] Prototype same against SGLang — compare implementation complexity, choose primary backend
- [ ] Implement model-based message compaction (was stubbed in Phase 2)
- [ ] Benchmark: 100 agents, TTFT after TOOL_WAIT with and without prefetch
- [ ] Record real numbers in LOGS.md: cold KV reload latency vs prefetch latency, by context length

**Success criteria:**
- Agents returning from TOOL_WAIT: TTFT ≤ 50ms (vs ~200ms cold reload for a 4-turn context)
- KVFlow Advisor prefetch accuracy ≥ 80%
- No regression in Phase 2 benchmark results

---

### Phase 3B — TokenDance Diff-Aware KV Storage (Week 10–13)

**Goal:** Store N agent contexts as 1 master + N sparse diffs instead of N full copies.

Tasks:
- [ ] Read vLLM `CacheEngine` and `BlockManager` source. Document which methods to override.
- [ ] Design `DiffCacheEngine`: master block pool + sparse diff block store
- [ ] Implement `All-Gather` step with soft timeout (run every 500ms or when >80% of agents complete current turn — no hard barrier; see W11)
- [ ] Implement block-sparse diff encoding. Target: ≥10x compression at N=100
- [ ] Gate behind `diff_kv=True`. Default off. Zero impact when disabled.
- [ ] Pin to specific vLLM version. Maintain compatibility matrix.
- [ ] Benchmark: concurrent agent capacity vs stock vLLM prefix caching. Target: 2.7x (TokenDance result)
- [ ] Consider upstreaming `/internal/prefetch` and `DiffCacheEngine` to vLLM

**Success criteria:**
- KV storage per agent ≥ 10x reduction vs stock vLLM at N=100 with `diff_kv=True`
- Concurrent agent capacity ≥ 2x vs stock vLLM prefix caching
- `diff_kv=False` shows zero performance change

---

### Phase 3C — SGLang as Primary Backend (Week 11–13, parallel to 3B)

SGLang's RadixAttention uses a token-level radix tree — more fine-grained than vLLM's block-level hashing. For multi-agent workloads with long shared prefixes that diverge mid-sequence, RadixAttention may outperform vLLM prefix caching without any patching.

Tasks:
- [ ] Complete the SGLang adapter (currently a structural stub)
- [ ] Run Phase 2 benchmark suite against SGLang. Compare vs vLLM.
- [ ] Implement KVFlow prefetch hints against SGLang
- [ ] Document: for which workloads does SGLang outperform vLLM?
- [ ] `backend="sglang://localhost:30000"` as first-class public API option

---

### Phase 4 — Distributed Orchestration (Week 14–18)

**Goal:** Multiple orchestration servers, multiple inference nodes, single logical batch run.

**Architecture change:** Agent state moves from in-process dict to Redis Streams. The Wave Scheduler becomes a distributed coordinator — multiple instances, each managing a shard of the agent pool, coordinating via a shared priority queue in Redis.

Tasks:
- [ ] Redis Streams state store: `AgentState` checkpointed every turn
- [ ] Distributed Wave Scheduler: consistent hashing to shard agents across orchestration nodes
- [ ] Cross-node KV sharing via LMCache: KV blocks served from one node to another over the network instead of recomputing
- [ ] Multi-node vLLM cluster configuration in `deploy/`
- [ ] Fault tolerance: if an orchestration node dies, agents rebalance to surviving nodes from last checkpoint
- [ ] Optimistic locking on agent state: version number on every write; stale writes discarded (see W13)
- [ ] 1,000-agent benchmark: 4 orchestration nodes, 2 vLLM nodes (2×8×A100)

**Success criteria:**
- 1,000 agents across 4 nodes with ≤2% failure rate
- Linear scaling: 2 inference nodes ≥ 1.8x throughput vs 1 node
- Single orchestration node failure results in ≤5% agent loss (rest resume from checkpoint)
- Cross-node KV cache hit rate (LMCache) > 70% for shared prefix blocks

---

## 5. Benchmark Suite

### 5.1 Paper Summarization (Primary)

**Configurations:**

| Config | Description |
|---|---|
| A | Naive parallel Anthropic API |
| B | Anthropic Batch API (24h async, no tools) |
| C | Anthropic API + BatchAgent |
| D | vLLM naive (no prefix warming, no scheduling) |
| E | vLLM + BatchAgent native (Phase 1/2) |
| F | vLLM + BatchAgent + KVFlow prefetch (3A) |
| G | vLLM + BatchAgent + TokenDance diff KV (3B) |
| H | vLLM + BatchAgent + KVFlow + TokenDance (3A+3B) |
| I | SGLang + BatchAgent native (3C) |
| J | SGLang + BatchAgent + KVFlow (3A+3C) |

**Scale:** N = 10, 50, 100, 500.

**Metrics:** Wall-clock (first result, all results), GPU-hours, prefix cache hit rate, KV storage per agent (G/H only), concurrent agent capacity, TTFT P50/P95/P99 cold vs warm, TTFT after TOOL_WAIT (baseline vs KVFlow), output quality (human eval, 50 samples), failure rate.

---

### 5.2 Code Review at Scale

Multi-turn code review (read file → inspect dependency → form opinion). N=100. Configs C, E, F, I.

Additional metrics: average turns per agent, TOOL_WAIT time fraction, prefetch hit rate.

---

### 5.3 Stress Test — Heterogeneous Task Duration

Mix of 1-turn and 5-turn tasks. Slot utilization over time should be flat. `--chaos` flag: kill 10% of tool calls mid-flight. Verify graceful degradation.

---

### 5.4 Tool Deduplication Efficiency

100 agents, 10 shared reference documents. Verify: 1,000 reads → 10 reads via coalescing pool.

---

### 5.5 KVFlow Prefetch Accuracy *(new)*

50 agents, 4 turns, simulated tool calls with fixed latency. Measure: TTFT after TOOL_WAIT (cold vs prefetch), prefetch accuracy (hints that result in GPU hit), prefetch waste (blocks prefetched but evicted before use).

**Target:** ≥80% accuracy, ≤10% waste.

---

### 5.6 TokenDance Compression Ratio *(new)*

100 agents, 4 turns, 2,048-token shared system prompt, 500-token per-agent context. Measure: KV storage without vs with diff encoding, compression ratio, concurrent agent capacity in GPU memory.

Reproduce TokenDance paper numbers or honestly document divergence.

---

## 6. Known Weaknesses & Mitigations

### W1: Template parsing breaks with complex prompts

**Mitigation:** `system_prompt` parameter is the explicit escape hatch. `--no-hoist` flag disables auto-hoisting.

---

### W2: KV cache eviction under memory pressure

**Mitigation:** Pin shared prefix block (`BlockManager` one-line patch). Adaptive concurrency reduces `max_concurrent` if hit rate drops below 90%. Two-level prefix for very large prefixes (>4K tokens).

---

### W3: Tool result size variance causes context overflow

**Mitigation:** Per-tool `max_tokens` limit. Trigger compaction when context exceeds 85% of model limit. `@summarizable` annotation returns structured summary when space-constrained.

---

### W4: Output schema validation failures at scale

**Mitigation:** `json_repair` pre-validation (no model call). Repair-prompt retry (~100 tokens vs full context). Log warning if repair rate >10%.

---

### W5: Priority queue doesn't account for tool wait time ✅ FIXED

Semaphore wraps only inference calls. Agents in `TOOL_WAIT` release their slot. Verified in integration test with instrumented logging.

---

### W6: Redis state store single point of failure

**Mitigation:** v0–2: in-process dict + SQLite checkpoint. Phase 4: Redis Streams (append-only). Checkpoint every turn. Crash recovery resumes from last checkpoint.

---

### W7: Diff-aware KV storage requires vLLM internals access

**Mitigation:** `diff_kv=False` default. Compatibility matrix per vLLM version. SGLang RadixAttention as alternative. Pure prefix caching fallback delivers 60–70% of benefit with zero patching.

---

### W8: Self-hosted assumption excludes most users

**Mitigation:** Degraded API mode for everyone. Together AI, Fireworks, Anyscale as managed vLLM-compatible targets. One-command RunPod/Modal deploy in `deploy/`.

---

### W9: No stateful agents across batch runs

**Mitigation:** `state_store` hook. Users pass a custom `StateStore`. SDK calls `store.save(state)` each turn, `store.load(job_id)` on startup. Out of scope for v0–3; clean extension point.

---

### W10: KVFlow ETA estimation accuracy degrades for variable-latency tools *(new)*

**Problem:** Web search and external HTTP calls have high variance (P50=300ms, P99=4s). Bad ETA → prefetch too early (blocks evicted before use) or too late (agent reactivates cold).

**Mitigation:**
1. Use P75 latency for ETA (conservative; bias toward slightly early over late).
2. `prefetch_horizon` config (default: 2.0s): only prefetch if ETA < threshold. Agents waiting >2s don't consume GPU memory budget for speculative prefetch.
3. Track prefetch accuracy per tool type. If accuracy drops below 60% for a tool, widen ETA estimate dynamically.
4. Deterministic tools (`read_file`, SQL): always prefetch. Stochastic tools (`web_search`, `http_get`): apply horizon filter.

---

### W11: All-Gather step creates a synchronization barrier *(new)*

**Problem:** TokenDance's KV Collector deduplicates shared blocks in a collective step after each turn. This means slow agents stall fast ones.

**Mitigation:**
1. No hard barrier. Run All-Gather asynchronously: deduplicate blocks from agents that have completed their turn without waiting for stragglers.
2. A partial All-Gather (70% of agents) achieves 70% of compression benefit. Correctness is not affected.
3. Soft timeout: run All-Gather every 500ms or when >80% of agents complete the current turn, whichever comes first.

---

### W12: Predictive tool pre-warming can serve stale results *(new)*

**Problem:** Pre-warmed tool results from checkpoint history may be stale if files changed or web content updated.

**Mitigation:**
1. Only pre-warm `@cacheable=True` tools (explicitly declared deterministic by tool author).
2. `predictive_prewarm=False` default. Explicit opt-in.
3. File tools: check mtime before serving pre-warmed result. Re-fetch if changed.
4. `prewarm_max_age` parameter (default: 60s). Pre-warmed results older than this are discarded.
5. Log all pre-warm cache hits for auditability.

---

### W13: Distributed mode introduces split-brain risk *(new)*

**Problem:** Multiple orchestration nodes + network partition → two nodes believe they own the same agent → duplicate inference calls, corrupted state.

**Mitigation:**
1. Redis `SET NX` for job ownership. Each node acquires a lease before processing a job. Lease TTL = 2× expected turn latency. Renewed every turn.
2. If a node fails to renew (crash), another node picks up the job after TTL expires.
3. Optimistic locking: version number in every state write. Stale writes (version mismatch) are discarded; node re-reads before retrying.
4. Idempotent tool calls: results stored by content hash. Accidental duplicate execution produces identical result. No corruption.

---

### W14: LMCache cross-node KV sharing adds network latency *(new)*

**Problem:** Phase 4 uses LMCache to share KV blocks across inference nodes. Network transfer of KV blocks for a large context (e.g. 4K tokens × 70B model) can take longer than recomputing from scratch.

**Mitigation:**
1. Only use LMCache for blocks above a minimum size threshold (>512 tokens by default). Small KV blocks are cheaper to recompute than to transfer.
2. LMCache should operate on the same physical network as vLLM (infiniband or at least 25GbE). Benchmark cross-node KV transfer latency before enabling in production.
3. Prioritize LMCache for the shared system prompt block — it is large (1K–4K tokens), identical across all agents, and computed on only one node on first request. This is the highest-value cross-node sharing target.
4. If cross-node transfer latency > 50ms for a given block, skip LMCache and recompute locally.

---

## 7. File Structure

```
batch-agent/
├── pyproject.toml
├── README.md
├── agents.md                              ← this file
│
├── batch_agent/
│   ├── __init__.py                        # public API: BatchAgent, Tool, BatchSpec
│   ├── spec.py                            # BatchSpec, AgentJob, ExecutionPlan
│   ├── compiler.py                        # Task Compiler + tool sequence prediction
│   ├── scheduler.py                       # Wave Scheduler + adaptive concurrency
│   ├── state.py                           # AgentState, AgentStatus, stores
│   ├── kvflow.py                          # KVFlow Advisor (NEW)
│   ├── tools/
│   │   ├── __init__.py                    # Tool registry, @batchable, @cacheable, @summarizable
│   │   ├── pool.py                        # Tool Pool + predictive pre-warming
│   │   ├── builtin.py                     # read_file, http_get, web_search, python_eval
│   │   └── sql.py                         # SQL batch grouping
│   ├── backends/
│   │   ├── __init__.py                    # BackendAdapter ABC
│   │   ├── vllm.py                        # vLLM: prefix warming, KV diff, prefetch
│   │   ├── vllm_patch/
│   │   │   ├── prefetch_route.py          # /internal/prefetch endpoint (~100 lines)
│   │   │   └── diff_cache_engine.py       # DiffCacheEngine subclass
│   │   ├── sglang.py                      # SGLang + RadixAttention prefetch
│   │   ├── anthropic.py                   # Anthropic API + cache_control
│   │   ├── openai.py                      # OpenAI API
│   │   └── bedrock.py                     # AWS Bedrock Converse API
│   ├── compaction.py                      # Model-based message compaction
│   ├── repair.py                          # JSON repair + schema validation
│   ├── metrics.py                         # Prometheus + Grafana
│   └── cli.py                             # batch-agent CLI
│
├── tests/
│   ├── unit/
│   │   ├── test_compiler.py
│   │   ├── test_scheduler.py
│   │   ├── test_tool_pool.py
│   │   ├── test_repair.py
│   │   ├── test_kvflow_advisor.py         # NEW: ETA estimation, hint emission
│   │   └── test_diff_encoder.py           # NEW: block hashing, compression ratio
│   ├── integration/
│   │   ├── test_anthropic_backend.py
│   │   ├── test_vllm_backend.py
│   │   ├── test_sglang_backend.py         # NEW
│   │   ├── test_multiturn.py
│   │   ├── test_prefetch_accuracy.py      # NEW: prefetch hit rate
│   │   └── test_distributed.py            # Phase 4: multi-node + Redis
│   └── benchmarks/
│       ├── paper_summarization.py
│       ├── code_review.py
│       ├── heterogeneous_tasks.py
│       ├── tool_dedup_efficiency.py
│       ├── kvflow_prefetch_accuracy.py    # NEW
│       └── tokendance_compression.py      # NEW
│
└── deploy/
    ├── vllm_server.sh                     # One-command vLLM setup + patches applied
    ├── sglang_server.sh                   # One-command SGLang setup
    ├── runpod_template.json
    ├── modal_deploy.py
    ├── redis_cluster.yaml                 # Phase 4: Redis Streams
    └── grafana_dashboard.json
```

---

## 8. API Reference

```python
# Simple case
results = await BatchAgent.run(
    task="Summarize: {text}",
    inputs=[{"text": t} for t in texts],
    model="claude-sonnet-4-20250514",
    backend="anthropic://",
)

# AWS Bedrock Converse API (uses standard boto3 credential chain)
results = await BatchAgent.run(
    task="Summarize: {text}",
    inputs=[{"text": t} for t in texts],
    model="anthropic.claude-sonnet-4-20250514-v1:0",
    backend="bedrock://us-east-1/anthropic.claude-sonnet-4-5",
)

# Full control
results = await BatchAgent.run(
    system_prompt="You are a precise scientific summarizer...",
    task="Analyze this paper:\n\n{paper_text}",
    inputs=[{"paper_text": t} for t in papers],
    tools=[Tool.read_file, Tool.web_search, my_custom_tool],
    output_schema=PaperSummary,
    model="meta-llama/Llama-3.1-70B-Instruct",
    backend="vllm://localhost:8000",
    max_concurrent=64,
    max_turns=6,
    max_retries=3,
    timeout_per_agent=300,
    on_result=lambda r: db.insert(r),
    diff_kv=False,                 # Phase 3B flag. Default off.
    kvflow=True,                   # Phase 3A flag. Default True in native mode.
    predictive_prewarm=False,      # opt-in: pre-warm tools from checkpoint history
    checkpoint_dir="./checkpoints",
)

# Streaming
async for result in BatchAgent.stream(task=..., inputs=...):
    process(result)

# With reduce step
results, summary = await BatchAgent.run_with_reduce(
    task="Extract claims from: {text}",
    inputs=[{"text": t} for t in texts],
    reduce="You have received {n} claim lists. Deduplicate and rank by evidence strength.",
    output_schema=ClaimList,
    reduce_schema=RankedClaimList,
)

# Custom tool
@Tool.define(max_tokens=2000, cacheable=True, rate_limit=5)
async def fetch_citation(doi: str) -> str:
    """Fetch paper metadata from CrossRef by DOI."""
    ...

# Batchable tool
@Tool.define(max_tokens=500)
@Tool.batchable(key_arg="paper_id", batch_query="SELECT * FROM papers WHERE id IN ({ids})")
async def get_paper_metadata(paper_id: int) -> PaperMeta:
    ...
```

---

## 9. Current State vs Target

| Capability | Phase 2 (now) | Phase 3 target | Phase 4 target |
|---|---|---|---|
| Multi-turn agent loop | ✅ | ✅ | ✅ |
| W5 semaphore fix | ✅ | ✅ | ✅ |
| Tool coalescing | ✅ | ✅ | ✅ |
| Priority queue | ✅ | ✅ | ✅ |
| Prefix caching (API) | ✅ | ✅ | ✅ |
| Prefix caching (vLLM, live) | Untested on GPU | ✅ tested + pinned | ✅ |
| KVFlow prefetch | ❌ | ✅ | ✅ |
| TokenDance diff KV | ❌ | ✅ (flag-gated) | ✅ |
| SGLang backend | Stub | ✅ | ✅ |
| Model-based compaction | Heuristic only | ✅ | ✅ |
| Predictive tool pre-warming | ❌ | ✅ (opt-in) | ✅ |
| Adaptive concurrency | ❌ | ✅ | ✅ |
| Distributed orchestration | ❌ | ❌ | ✅ |
| Cross-node KV sharing (LMCache) | ❌ | ❌ | ✅ |
| 1,000-agent benchmark | ❌ | ❌ | ✅ |

### 9.1 Bedrock Mode Findings

| Capability | Bedrock mode |
|---|---|
| Prompt cache token savings | ✅ confirmed |
| Prompt cache latency savings | ❌ queue latency dominates at <8K tokens |
| KVFlow / TokenDance | ❌ managed service, no KV access |
| Reliability vs naive | ✅ 0% vs 35% failure rate |
| Max practical concurrency | 1–3 (quota-limited) |

---

## 10. Success Definition

The SDK is done when:

1. A researcher runs `pip install batch-agent` and summarizes 100 papers with a 10-line script.
2. Self-hosted vLLM delivers ≥3x better cost-per-task vs naive parallel API calls at N=100.
3. Prefix cache hit rate ≥95% in steady state.
4. Agents returning from TOOL_WAIT show ≤50ms TTFT with KVFlow enabled.
5. KV storage per agent with TokenDance ≥10x lower than full-copy storage at N=100.
6. 500 concurrent agents: no OOM, no unhandled exceptions, ≤2% failure rate.
7. 1,000 agents across 4 nodes with linear throughput scaling (Phase 4).
8. Benchmark results are published, reproducible, and honest — including conditions where the SDK does NOT help: single-agent tasks, highly heterogeneous prompts with no shared prefix, API-only users who can't self-host.

---

*This document is the source of truth. Code diverges from this document only in the direction of this document — update here first, then code.*
