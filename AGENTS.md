# agents.md — Batch Agent SDK: Build Specification

> **Purpose:** This document is the authoritative build spec for the Batch Agent SDK.
> It is written to be read by an engineer starting from zero. Every design decision
> is stated, every weakness is named, every iteration is explicit.
> If something is ambiguous here, it is not yet designed.

---

## 0. What We Are Building

A Python SDK + server process that lets you do this:

```python
from batch_agent import BatchAgent, Tool

results = await BatchAgent.run(
    task="Summarize this paper and extract: benchmark name, primary metric, models tested.\n\nPaper: {paper_text}",
    inputs=[{"paper_text": text} for text in papers],
    tools=[Tool.read_file, Tool.web_search],
    output_schema=PaperSummary,          # Pydantic model → structured JSON output
    model="meta-llama/Llama-3.1-70B-Instruct",
    backend="vllm://localhost:8000",
    max_concurrent=64,                   # wave size
    on_result=lambda r: print(r),        # streaming callback as agents finish
)
```

And get back a list of `PaperSummary` objects, one per input, in order.
Time-to-first-result: seconds. Time-to-all: proportional to longest agent, not sum.

---

## 1. Principles

1. **Co-design the orchestration and inference layers.** This is the entire point. Any design that treats the inference backend as a black box HTTP endpoint is leaving 60–70% of the performance on the table.

2. **Shared prefix is gold.** The system prompt is identical across all N agents. Computing its KV cache once and sharing it is the single highest-leverage optimization available. Everything else is secondary.

3. **Agents finish at different times. Respect that.** Don't batch-wait for the slowest agent. Stream results. Free KV slots as agents complete. Next wave fills immediately.

4. **Failures are not exceptions. They are data.** At 500+ agents, some will fail (model error, tool timeout, malformed output). The SDK handles this gracefully by default with retry + structured error result. The caller never writes try/catch loops.

5. **The user writes a task template, not an agent.** The SDK is not an agent framework (that's LangChain). It is a batch execution engine that happens to use agents internally. The user's mental model is: input list in, result list out.

6. **Self-hosted first, API-compatible second.** The deep optimizations (prefix sharing, wave scheduling, diff storage) require controlling the inference layer. Commercial API support is offered but degrades gracefully to "just parallel API calls with rate limiting."

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Process                          │
│                   BatchAgent.run(...)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ asyncio / gRPC
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Server                       │
│                                                              │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │ Task Compiler │  │ Wave Scheduler │  │ Result Streamer │  │
│  └──────┬───────┘  └───────┬────────┘  └────────┬────────┘  │
│         │                  │                     │           │
│  ┌──────▼──────────────────▼─────────────────────▼────────┐ │
│  │                   Agent State Store                      │ │
│  │  (per-agent: turn history, tool results, status, KV key) │ │
│  └──────────────────────────┬───────────────────────────── ┘ │
│                             │                                 │
│  ┌──────────────────────────▼──────────────────────────────┐ │
│  │                  Tool Execution Pool                      │ │
│  │  (deduplicated, batched, rate-limited tool calls)        │ │
│  └──────────────────────────┬───────────────────────────── ┘ │
└──────────────────────────── │ ──────────────────────────────┘
                              │ vLLM OpenAI-compat HTTP
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Inference Adapter                         │
│                                                              │
│  ┌─────────────────┐   ┌──────────────────────────────────┐ │
│  │  Prefix Registry │   │  KV Diff Encoder (TokenDance)   │ │
│  └─────────────────┘   └──────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │           vLLM / SGLang (self-hosted)                     ││
│  │      OR   Anthropic / OpenAI API (degraded mode)          ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specs

### 3.1 Task Compiler

**Input:** A `BatchSpec` — task template string, input list, tools list, output schema, model, backend config.

**Output:** A `ExecutionPlan` — a list of `AgentJob` objects plus a `SharedContext` block.

**Responsibilities:**

1. **Extract the shared prefix.** Parse the task template. Everything before the first `{variable}` substitution is the shared prefix. If the user passes a `system_prompt` separately, that is always the shared prefix. The shared prefix is registered with the inference adapter *once* before any agents are dispatched (see §3.4).

2. **Validate the output schema.** If `output_schema` is a Pydantic model, compile it to a JSON Schema and inject a structured output constraint into each agent's system prompt. Add a final instruction: "Your final message must be a valid JSON object matching this schema. Do not add prose after the JSON."

3. **Estimate per-agent token budget.** For each input, estimate the prompt token count (shared prefix + per-agent context). Agents whose prompts exceed `model.max_context - min_response_tokens` are flagged as `OVERSIZED` and routed to a chunking sub-task (see §6.2 Weakness: Long Inputs).

4. **Build the DAG.** Default topology is flat (all agents independent). If the user provides a `reduce` function, append a `ReduceJob` node that activates when all agents are complete.

**Weaknesses at this stage (see §6 for mitigations):**
- Template parsing is fragile with complex f-string-style templates
- Token estimation is approximate (tiktoken vs model-specific tokenizer mismatch)
- No support yet for hierarchical agent topologies (agent spawning sub-agents)

---

### 3.2 Wave Scheduler

This is the core orchestration engine. It runs as an asyncio event loop inside the orchestration server.

**State machine per agent:**

```
PENDING → PREFLIGHT → RUNNING → TOOL_WAIT → RUNNING → ... → COMPLETE | FAILED
```

- `PENDING`: Created, not yet dispatched.
- `PREFLIGHT`: Prefix cache is being warmed; agent is queued.
- `RUNNING`: Active inference request in flight.
- `TOOL_WAIT`: Agent issued tool calls; waiting for Tool Execution Pool.
- `COMPLETE`: Structured output extracted and validated.
- `FAILED`: Exceeded max retries or hard error.

**Wave logic:**

```python
async def run_wave(plan: ExecutionPlan, max_concurrent: int):
    semaphore = asyncio.Semaphore(max_concurrent)
    queue = PriorityQueue()  # priority = estimated_turns_remaining (lower = higher priority)

    for job in plan.jobs:
        queue.put((estimate_turns(job), job))

    async def run_job(job):
        async with semaphore:
            await execute_agent(job)

    tasks = [asyncio.create_task(run_job(j)) for j in queue]
    await asyncio.gather(*tasks)
```

**Priority scoring:**
- `estimate_turns_remaining` starts at the configured `max_turns` and decrements each time the agent completes a turn.
- Agents that have already completed 3 of 5 turns get higher priority than agents on turn 1.
- This biases the semaphore toward draining nearly-finished agents first, freeing KV slots sooner.
- **Why this matters:** A 70B model on 4×A100s has a finite KV cache pool. An agent's KV allocation grows each turn. Draining finished agents frees large KV blocks immediately.

**Staggered cold start:**
- Don't dispatch all N agents at t=0. Dispatch the first `max_concurrent` agents immediately.
- As each agent completes turn 1 and enters `TOOL_WAIT`, dispatch one new `PENDING` agent.
- This creates a continuous flow rather than a burst-idle-burst pattern that kills throughput.

---

### 3.3 Tool Execution Pool

Agents issue tool calls. Tools are: `read_file`, `web_search`, `http_get`, `python_eval`, `sql_query`, user-defined.

**The deduplication problem:**
If 80 agents all call `read_file("arxiv_dump/paper_0042.pdf")`, that is one disk read fanned out. Without deduplication, it is 80 disk reads and 80 tool result tokens injected into 80 separate contexts. The content is identical. Only the KV encoding differs (because each agent has a different preceding context).

**Solution — Request Coalescing:**

```python
class ToolPool:
    _inflight: dict[str, asyncio.Future] = {}

    async def call(self, tool: str, args: dict) -> str:
        key = f"{tool}:{stable_hash(args)}"
        if key in self._inflight:
            return await self._inflight[key]   # wait on the same future

        future = asyncio.get_event_loop().create_future()
        self._inflight[key] = future

        try:
            result = await execute_tool(tool, args)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            del self._inflight[key]
```

This means: first agent to request `read_file("paper_0042.pdf")` creates a Future. All other agents requesting the same call await that Future. Tool executes once. All agents receive the same result string.

**Tool result caching:**
Results are cached in an in-process LRU (configurable TTL). Web search results older than 60s are re-fetched by default (configurable). File reads are cached until file mtime changes.

**Rate limiting per tool:**
Each tool has a configurable `rate_limit` (requests/sec). Web search defaults to 10/sec. SQL queries default to 50/sec. The pool enforces this with a token bucket per tool type.

**Batching for DB tools:**
If 30 agents all call `sql_query("SELECT * FROM papers WHERE id = ?", [id])` with different IDs, the pool detects the same query template and batches into a single `SELECT * FROM papers WHERE id IN (...)`. Result is split and routed back to each agent. This requires tool authors to annotate their tools with `@batchable(key_arg="id")`.

---

### 3.4 Inference Adapter

The adapter is the interface between the orchestration layer and the inference backend. It has two modes: **native** (self-hosted vLLM/SGLang) and **API** (Anthropic/OpenAI, degraded).

#### 3.4.1 Native Mode — Prefix Registry

Before the first wave is dispatched, the adapter submits the shared prefix for caching:

```python
async def warm_prefix(shared_prefix: str) -> str:
    # vLLM: POST /v1/completions with prompt=shared_prefix, max_tokens=0
    # This forces vLLM to compute + cache the KV blocks for this prefix.
    # Returns the prefix hash that vLLM will use for cache lookup.
    response = await vllm_client.post("/v1/completions", json={
        "model": model,
        "prompt": shared_prefix,
        "max_tokens": 0,
    })
    return compute_prefix_hash(shared_prefix)
```

Subsequent requests with the same leading tokens hit the prefix cache immediately. The KV computation cost for the system prompt is paid **once**, regardless of N.

**Measured impact:** For a 2,048-token system prompt on a 70B model, TTFT (time-to-first-token) drops from ~800ms to ~40ms for cache-hitting requests. At N=500, this saves ~380 GPU-seconds of prefill compute.

#### 3.4.2 Native Mode — KV Diff Encoding (TokenDance)

After turn 1, each agent's context diverges. Agent 0 has seen [system_prompt + task_0 + response_0 + tool_result_0]. Agent 1 has seen [system_prompt + task_1 + response_1 + tool_result_1]. These share the system_prompt prefix but diverge after that.

The TokenDance approach: rather than storing N full KV caches, store:
- 1 master KV block (the shared system prompt)
- N sparse diff blocks (each agent's unique context delta)

The diff block encodes only the KV entries that differ from the master. Since agents share the system prompt (potentially 30–50% of total context length), the savings are substantial.

**Implementation path:**
- vLLM does not natively support diff-aware storage. This requires a custom `CacheEngine` subclass.
- For v0, we implement a **simulated** diff store: we track which KV blocks are identical across agents using content hashing, and skip re-sending them in multi-turn requests.
- For v1, we patch vLLM's `CacheEngine` directly (this is the ambitious path).

#### 3.4.3 API Mode (Degraded)

When backend is `anthropic://`, `openai://`:
- Prefix sharing uses prompt caching (Anthropic `cache_control` headers, OpenAI implicit caching).
- No wave scheduling — rate limiting is the binding constraint instead.
- Tool deduplication still works (it's in the orchestration layer).
- KV diff encoding is not applicable.
- Max throughput is determined by API tier rate limits, not GPU memory.

The SDK transparently detects which mode it's in and adjusts.

---

### 3.5 Agent State Store

Each `AgentJob` has a mutable state object:

```python
@dataclass
class AgentState:
    job_id: str
    status: AgentStatus
    turn: int                           # current turn number
    messages: list[Message]             # full conversation history
    tool_calls_pending: list[ToolCall]  # tool calls from last model response
    tool_results: list[ToolResult]      # results to inject next turn
    kv_key: str | None                  # vLLM KV cache key for this agent
    output: Any | None                  # final structured output
    error: AgentError | None
    retry_count: int
    created_at: float
    last_updated: float
```

State is stored in-process (dict, keyed by `job_id`) for single-machine deployments. For distributed deployments (multiple orchestration servers), state is stored in Redis with optimistic locking. The v0 implementation is single-machine only.

**Message compaction:**
After every 3 turns, tool results older than 2 turns are summarized (via a lightweight call to a small model, e.g. Llama-3.2-3B) and replaced with a compact summary. This prevents context length from growing unbounded in long-running agents. The compaction call is made using the same inference backend, routed through a separate low-priority queue.

---

### 3.6 Result Streamer

Results are streamed back to the caller as agents complete, not after all agents finish.

```python
async for result in BatchAgent.stream(...):
    print(result.job_id, result.output)   # arrives as each agent finishes
```

Internally this is an `asyncio.Queue` that the Wave Scheduler pushes to as agents transition to `COMPLETE` or `FAILED`. The caller iterates the queue. If `on_result` callback is provided, the SDK calls it internally and returns the full list at the end.

---

## 4. Implementation Plan

### Phase 0 — Foundation (Week 1–2)

**Goal:** Single-machine, API-backend (Anthropic), no inference optimizations. Just correct.

Tasks:
- [ ] `BatchSpec` and `AgentState` dataclasses
- [ ] Task Compiler: template parsing, schema injection, DAG construction
- [ ] Wave Scheduler: asyncio semaphore-based concurrency, no priority yet
- [ ] Tool Execution Pool: coalescing + LRU cache, built-in `read_file` + `http_get` tools
- [ ] Result Streamer: `asyncio.Queue` + `on_result` callback
- [ ] Anthropic API backend adapter (with `cache_control` prompt caching)
- [ ] Pydantic output schema validation + retry on malformed JSON
- [ ] CLI: `batch-agent run --spec spec.yaml` for testing

**Test:** Run 50 agents against Anthropic API on the paper summarization task. Measure wall-clock, verify output correctness, verify deduplication works for tool calls.

**Success criteria:** 50 paper summaries in <3 min, 0 unhandled exceptions, tool deduplication observable in logs.

---

### Phase 1 — Inference Integration (Week 3–5)

**Goal:** vLLM native mode. Prefix warming. Wave scheduling with priority queue.

Tasks:
- [ ] vLLM adapter: prefix warming on startup, prefix-keyed requests
- [ ] SGLang adapter: same interface, different client (SGLang `/generate` endpoint)
- [ ] Priority queue in Wave Scheduler (min-heap on `turns_remaining`)
- [ ] Staggered cold-start dispatch
- [ ] KV block content hashing (simulated diff awareness)
- [ ] Metrics collection: TTFT per request, cache hit rate, queue depth, agent turn latency
- [ ] Prometheus endpoint + Grafana dashboard template

**Test:** 100-agent run on local 4×A100 (70B model). Compare against naive parallel vLLM calls (same requests, no prefix warming, no wave scheduling).

**Success criteria:**
- Cache hit rate ≥ 95% on system prompt prefix
- TTFT for cache-hitting requests ≤ 50ms (vs ~800ms cold)
- Wall-clock 100 summaries ≤ 8 min (vs ≥ 20 min naive)
- Peak GPU memory ≤ 80% (wave scheduling prevents OOM)

---

### Phase 2 — Scale + Robustness (Week 6–8)

**Goal:** 500+ agents, failure handling, message compaction, batching for DB tools.

Tasks:
- [ ] Retry logic: exponential backoff, configurable max_retries, partial result on failure
- [ ] Message compaction (lightweight summarization of old turns)
- [ ] `@batchable` tool annotation + SQL/HTTP batch grouping
- [ ] `reduce` topology: aggregator agent that sees all N results
- [ ] Configurable timeouts per agent, per turn, per tool call
- [ ] Overflow to disk: if in-process state store exceeds memory limit, spill to SQLite
- [ ] 500-agent benchmark (full benchmark suite, see §5)

**Test:** Full benchmark suite. Chaos testing: kill 10% of tool calls mid-flight, verify retry + graceful degradation.

**Success criteria:**
- 500 agents complete with ≤ 2% failure rate (after retries)
- No OOM crashes at 500 agents on 4×A100
- Cost per task ≤ 15% of naive API approach

---

### Phase 3 — Diff-Aware KV Storage (Week 9–12)

**Goal:** Implement TokenDance-style KV diff encoding in vLLM.

This is the ambitious part. It requires patching vLLM internals.

Tasks:
- [ ] Study vLLM `CacheEngine` and `BlockManager` source
- [ ] Design `DiffCacheEngine` subclass: master block pool + sparse diff blocks
- [ ] Implement `All-Gather` step: after each turn, collect all agent KV blocks, deduplicate shared blocks, encode diffs
- [ ] Benchmark: concurrent agent capacity vs stock vLLM prefix caching
- [ ] Upstream PR to vLLM (optional, but increases visibility)

**Target:** Reproduce TokenDance's 2.7x concurrent agent improvement over stock vLLM with prefix caching.

**Risk:** vLLM internals change frequently. Pinning to a specific vLLM version is required. This phase may slip. It is not required for the SDK to be useful — Phase 2 is already a significant improvement over baseline.

---

## 5. Benchmark Suite

### 5.1 Paper Summarization (Primary Benchmark)

**Task:** Given N academic papers (PDF → text, avg 8,000 tokens each), extract structured metadata and write a 150-word summary.

**Output schema:**
```python
class PaperSummary(BaseModel):
    proposes_benchmark: bool
    benchmark_name: str | None
    primary_metric: str
    models_tested: list[str]
    summary: str  # 150 words
```

**Conditions:**

| Config | Description |
|---|---|
| A | Naive parallel Anthropic API (asyncio.gather, no rate limit respect) |
| B | Anthropic Batch API (async, 24h window, no tools) |
| C | Anthropic API + BatchAgent orchestration (Phase 0) |
| D | vLLM naive (no prefix warming, no wave scheduling) |
| E | vLLM + BatchAgent native mode (Phase 1) |
| F | vLLM + BatchAgent + diff KV (Phase 3) |

**Scale:** N = 10, 50, 100, 500. Report all combinations.

**Metrics per condition:**
- Wall-clock time to first result
- Wall-clock time to all results
- GPU-hours consumed (configs D/E/F only)
- Prefix cache hit rate (configs D/E/F only)
- Output quality score (human eval on 50 random samples, 1–5 scale)
- Cost in USD (configs A/B/C only)
- Cost in GPU-minutes (configs D/E/F only)
- Failure rate (% of agents that failed after max retries)

---

### 5.2 Code Review at Scale

**Task:** Given N GitHub PRs (diff + context), produce structured code review with: severity (P0/P1/P2), category (bug/perf/style/security), and 2-sentence comment per issue found.

**Why this tests differently:** Code review agents often need multiple turns (read file → inspect dependency → form opinion). Tests multi-turn scheduling.

**N = 100.** Compare configs C and E.

**Additional metric:** Average turns per agent (expect 2–4).

---

### 5.3 Stress Test — Heterogeneous Task Duration

**Task:** Mix of tasks with very different completion times — some finish in 1 turn, some require 5 turns with tool calls.

**Why:** Validates that wave scheduling's priority queue actually drains fast agents first and doesn't block on slow ones.

**Measure:** Distribution of slot utilization over time. A good scheduler keeps utilization flat. A bad scheduler has a burst-then-idle pattern.

---

### 5.4 Tool Deduplication Efficiency

**Task:** 100 agents, all of which need to read the same 10 shared reference documents.

**Measure:**
- Without deduplication: 100 × 10 = 1,000 file reads
- With deduplication: 10 file reads (1 per unique document)
- Verify via tool execution pool logs

**Expected result:** 100x reduction in file I/O. This also means 100x fewer tool result tokens injected into each unique context — they are still injected, but the *execution* is deduplicated.

---

## 6. Known Weaknesses & Mitigations

This section is the most important section of this document. Every design has failure modes. Name them early.

---

### W1: Template parsing breaks with complex prompts

**Problem:** The Task Compiler identifies the shared prefix by finding the first `{variable}` in the template. If the user writes a prompt like:

```
"You are an expert in {domain}. Analyze this paper: {paper_text}"
```

...then `{domain}` is shared context (same for all agents) but `{paper_text}` is per-agent. The compiler currently treats everything before the first variable as shared, which would set the shared prefix to `"You are an expert in "` — useless for prefix caching.

**Mitigation:** The `BatchSpec` API separates `system_prompt` (always shared, always the prefix) from `task_template` (per-agent). Users who want maximum prefix caching put their invariant instructions in `system_prompt`. The template compiler only looks at `task_template` for variable substitution. Shared variables in `task_template` that are the same across all inputs are detected by inspection and hoisted into the system prompt automatically.

**Residual risk:** Auto-hoisting can be wrong if a "constant" variable is actually meant to vary. Provide a `--no-hoist` flag to disable.

---

### W2: KV cache eviction under memory pressure

**Problem:** vLLM uses an LRU eviction policy for KV blocks. At 500 concurrent agents, the total KV cache demand can exceed GPU memory even with prefix sharing. When the shared prefix block is evicted (because it hasn't been accessed recently and 500 other agents' unique blocks are competing for space), every subsequent request becomes a cold start.

**Mitigation:**
1. Mark the shared prefix block as `pinned` in vLLM (requires a one-line patch to `BlockManager`). Pinned blocks are never evicted.
2. The Wave Scheduler monitors the `prefix_cache_hit_rate` metric. If it drops below 90%, it reduces `max_concurrent` to ease memory pressure.
3. For very large shared prefixes (>4K tokens), split into two levels: a "static" prefix (always pinned) and a "dynamic" shared header (evictable).

---

### W3: Tool result size variance causes context overflow

**Problem:** `web_search` returns wildly varying result sizes. One search might return 200 tokens; another might return 8,000 tokens. For a long-context agent that has already consumed 5 turns, a large tool result can push the total context over the model's limit.

**Mitigation:**
1. Each tool has a configurable `max_tokens` output limit. Results are truncated with a `[TRUNCATED — {n} tokens omitted]` suffix.
2. The Wave Scheduler tracks each agent's running context length. When an agent's estimated context exceeds `model.max_context × 0.85`, it triggers message compaction immediately (see §3.5) before the next turn.
3. Tools annotated with `@summarizable` can return a structured summary instead of raw output when space is constrained. `web_search` defaults to returning only title + first paragraph per result when in space-constrained mode.

---

### W4: Output schema validation fails silently

**Problem:** The model produces a response that is almost valid JSON — maybe a trailing comma, maybe a missing quote. The SDK's schema validator raises an exception. By default, the agent retries (up to `max_retries=3`). But each retry costs a full forward pass. At 500 agents with a 5% malformed output rate, that's 25 extra forward passes.

**Mitigation:**
1. Use `json_repair` (a lightweight Python library) as a pre-validation step. It fixes common JSON syntax errors without a model call.
2. Add a "repair prompt" retry path: instead of resending the full conversation, send only the last response with the instruction "Fix the JSON syntax error in this response and return only valid JSON." This is a ~100-token request vs a full-context request.
3. Instrument the repair rate. If it exceeds 10%, the system prompt's JSON instruction is likely too vague — log a warning and suggest schema simplification.

---

### W5: Priority queue doesn't account for tool wait time

**Problem:** The priority scorer uses `turns_remaining` as a proxy for time-remaining. But an agent on turn 3 of 5 that just called `web_search` (500ms latency) will block its semaphore slot while waiting for the tool, even though the GPU could be serving other agents during that time.

**Mitigation:**
1. Agents in `TOOL_WAIT` state **release their semaphore slot immediately**. The semaphore is re-acquired only when tool results are ready and the agent is about to make its next inference request. This means `max_concurrent` is the number of *active inference requests*, not the number of *active agents*. An agent waiting for a web search doesn't count against the limit.
2. This requires restructuring the semaphore from wrapping the entire agent lifecycle to wrapping only each individual inference call. This is the correct design.
3. With this fix, effective GPU utilization goes from ~60% (blocked by tool waits) to ~90%+.

**This is a critical fix. Implement in Phase 0.**

---

### W6: Redis state store is a single point of failure

**Problem:** In distributed mode (multiple orchestration servers), all agent state is stored in Redis. If Redis goes down mid-run, all in-flight agents lose their state.

**Mitigation:**
- v0 is single-machine only. Redis is not used. In-process dict.
- v1 distributed mode: Use Redis Streams for the result queue (append-only, survives partial failures). Agent state is checkpointed to Redis every turn (not every message). On crash recovery, agents are resumed from their last checkpointed turn.
- Provide a `--checkpoint-dir` option for single-machine runs that checkpoints to local SQLite. If the orchestration server crashes, re-running with the same `--checkpoint-dir` skips already-completed agents.

---

### W7: Diff-aware KV storage requires vLLM internals access

**Problem:** The TokenDance approach requires hooking into vLLM's `CacheEngine` to intercept block allocation and implement diff storage. vLLM's internals are not stable across versions. A patch written for vLLM 0.6.x may break on vLLM 0.7.x.

**Mitigation:**
1. Maintain a compatibility matrix: tested vLLM versions per SDK version.
2. The diff KV feature is gated behind an explicit flag: `diff_kv=True`. Default is off. This means the SDK ships and is useful even if Phase 3 slips.
3. Explore whether SGLang's RadixAttention can be used as a drop-in for the diff storage problem. SGLang's token-level radix tree is more naturally suited to the multi-agent divergence pattern and may not require the same level of patching.
4. Maintain a "pure prefix caching" fallback that gets 60-70% of the benefit with zero patching.

---

### W8: The self-hosted assumption excludes most potential users

**Problem:** Running a 70B model on 4×A100s is not trivial. Most teams who want to batch 100 agents don't have a GPU cluster. This limits the SDK's addressable market significantly.

**Mitigation:**
1. The SDK works with commercial APIs (Anthropic, OpenAI) in degraded mode. Users can start with the API, get the orchestration and deduplication benefits, and migrate to self-hosted when scale justifies it.
2. Target cloud-hosted inference as a near-term priority: Together AI, Fireworks, Anyscale all offer vLLM-compatible endpoints where prefix caching is available without managing the cluster. The SDK should test against these explicitly.
3. Add a `RunPod` / `Modal` deploy script that provisions a vLLM server with one command. Lower the barrier to "self-hosted" to mean "one GPU on a cloud provider."

---

### W9: No support for stateful agents that need persistent memory

**Problem:** The SDK is designed for stateless batch tasks. But some use cases require agents that persist state across multiple batch runs (e.g. a research agent that builds a knowledge base over weeks). The `AgentState.messages` list is ephemeral — it lives for the duration of a single `BatchAgent.run()` call.

**Mitigation:**
- This is intentionally out of scope for v0. The SDK is a *batch execution engine*, not a *persistent agent platform*.
- Provide a `state_store` hook: users can pass a custom `StateStore` implementation that persists `AgentState` to their own storage. The SDK calls `store.save(state)` after each turn and `store.load(job_id)` on startup. This is a clean extension point.
- The "Agent Memory Below the Prompt" paper (Feb 2026) is the research basis for persistent KV caching across sessions. That is a future direction, not a v0 requirement.

---

## 7. File Structure

```
batch-agent/
├── pyproject.toml
├── README.md
├── agents.md                    ← this file
│
├── batch_agent/
│   ├── __init__.py              # public API: BatchAgent, Tool, BatchSpec
│   ├── spec.py                  # BatchSpec, AgentJob, ExecutionPlan dataclasses
│   ├── compiler.py              # Task Compiler
│   ├── scheduler.py             # Wave Scheduler (asyncio)
│   ├── state.py                 # AgentState, AgentStatus, in-process store
│   ├── tools/
│   │   ├── __init__.py          # Tool registry, @batchable decorator
│   │   ├── pool.py              # Tool Execution Pool (coalescing, caching)
│   │   ├── builtin.py           # read_file, http_get, web_search, python_eval
│   │   └── sql.py               # SQL batch grouping
│   ├── backends/
│   │   ├── __init__.py          # BackendAdapter ABC
│   │   ├── vllm.py              # vLLM native adapter (prefix warming, KV diff)
│   │   ├── sglang.py            # SGLang adapter
│   │   ├── anthropic.py         # Anthropic API adapter (prompt caching)
│   │   └── openai.py            # OpenAI API adapter
│   ├── compaction.py            # Message compaction (lightweight summarization)
│   ├── repair.py                # JSON repair + structured output validation
│   ├── metrics.py               # Prometheus metrics, Grafana dashboard template
│   └── cli.py                   # batch-agent CLI
│
├── tests/
│   ├── unit/
│   │   ├── test_compiler.py
│   │   ├── test_scheduler.py
│   │   ├── test_tool_pool.py
│   │   └── test_repair.py
│   ├── integration/
│   │   ├── test_anthropic_backend.py
│   │   └── test_vllm_backend.py
│   └── benchmarks/
│       ├── paper_summarization.py    # Primary benchmark
│       ├── code_review.py
│       ├── heterogeneous_tasks.py
│       └── tool_dedup_efficiency.py
│
└── deploy/
    ├── vllm_server.sh           # One-command vLLM setup
    ├── runpod_template.json     # RunPod GPU instance template
    ├── modal_deploy.py          # Modal serverless deploy
    └── grafana_dashboard.json   # Metrics dashboard
```

---

## 8. API Reference (Target)

```python
# Simple case
results = await BatchAgent.run(
    task="Summarize: {text}",
    inputs=[{"text": t} for t in texts],
    model="claude-sonnet-4-20250514",
    backend="anthropic://",
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
    timeout_per_agent=300,       # seconds
    on_result=lambda r: db.insert(r),
    diff_kv=False,               # Phase 3 feature flag
    checkpoint_dir="./checkpoints",
)

# Streaming
async for result in BatchAgent.stream(task=..., inputs=...):
    # Result arrives as each agent finishes, not at the end
    process(result)

# With reduce step
results, summary = await BatchAgent.run_with_reduce(
    task="Extract claims from: {text}",
    inputs=[{"text": t} for t in texts],
    reduce="You have received {n} claim lists. Deduplicate and rank by evidence strength.",
    output_schema=ClaimList,
    reduce_schema=RankedClaimList,
)

# Custom tool definition
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

## 9. Success Definition

The SDK is done when:

1. A researcher can run `pip install batch-agent` and summarize 100 papers with a 10-line script.
2. The self-hosted path (vLLM) delivers ≥3x better cost-per-task vs naive parallel API calls at N=100.
3. The prefix cache hit rate is ≥95% in steady state for any shared-system-prompt workload.
4. The SDK handles 500 concurrent agents without OOM or unhandled exceptions.
5. Failure rate (after retries) is ≤2% for well-formed tasks on a stable backend.
6. The benchmark results are published, reproducible, and honest — including the conditions under which the SDK does NOT help (single-agent tasks, highly heterogeneous prompts with no shared prefix, API-only users who can't self-host).

---

*Last updated: May 2026. This document is the source of truth. If code diverges from this document, update this document first, then update the code.*
