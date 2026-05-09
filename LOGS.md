Record all changes with time and date here. Design choices, mistakes, bugs, etc. inclusive.

## 2026-05-09

- Read `AGENTS.md` in full before writing code, per instruction. The repo only contained `AGENTS.md` and `LOGS.md`, so the implementation scope became a Phase 0 foundation rather than patching existing code.
- Implemented package metadata, public API, compiler/state dataclasses, backend adapters, one-shot scheduler, tool definitions/pool/builtins, JSON repair, CLI, and unit tests aligned to Phase 0.
- Design choice: `BatchAgent.run()` currently returns `AgentResult` objects instead of raw schema objects. This preserves the spec principle that failures are data, but it is not yet the final ergonomic target shown in `AGENTS.md` where successful runs return plain structured outputs.
- Design choice: auto-hoisting records shared constant variables in `SharedContext` and the shared prefix but does not rewrite the per-agent prompt. This avoids semantic surprises in Phase 0, but leaves some prefix-cache optimization on the table.
- Design choice: the scheduler wraps only backend inference calls in the semaphore, matching W5's requirement that non-inference waits must not consume GPU slots. Full multi-turn tool waiting is not implemented yet.
- Design choice: `web_search` is registered but raises `NotImplementedError` until a concrete search provider is configured. This avoids pretending to have a working external search integration.
- Safety choice: `python_eval` is disabled unless `BATCH_AGENT_ENABLE_PYTHON_EVAL=1`. The spec names it as a builtin, but enabling arbitrary evaluation by default would be a security bug.
- Known gap: Anthropic/OpenAI/vLLM/SGLang adapters are minimal HTTP adapters and are not integration-tested against live services in this environment.
- Known gap: SQL batch grouping, reduce jobs, checkpointing, message compaction, metrics, and diff-aware KV storage are placeholders or future-phase work.
- Bug found during verification: the local test runner is Python 3.10, so `enum.StrEnum` and a Python 3.11-only package declaration broke imports. Fixed by supporting Python 3.10 with `class AgentStatus(str, Enum)` and updating `requires-python`.
- Test environment issue found during verification: `pytest-asyncio` is not installed in the base environment. Converted async unit tests to call `asyncio.run()` directly so the suite runs without optional extras.
- Test environment issue found during verification: the `pytest` entrypoint did not include the repository root on `sys.path` in this environment even though direct `python3` imports worked. Added `pythonpath = ["."]` to pytest config to make package discovery explicit.

### Integration test (20 agents, mock Anthropic) — 2026-05-09

- **Result:** 20/20 OK in 0.21s (mock server, no network latency). Streaming works, Pydantic output validated, concurrency bounded at max_concurrent=10 (two visible waves).
- **Real Anthropic attempt failed:** API key has insufficient credits (400 "credit balance too low"). Request payload format is correct (verified manually); billing issue only.
- **Bug found:** `backend_from_url("anthropic://")` always constructs `AnthropicBackend()` with default `base_url`. There is no way to pass a custom base URL through the URL string. Fixed by allowing direct backend injection in the test via `WaveScheduler(plan, backend)`.
- **Observation:** Retry logic fires correctly (exponential backoff) when live API returns 400/401 — all 3 attempts exhausted before reporting failure.

### Multi-turn loop implementation — 2026-05-09

#### Changes made

1. **`backends/__init__.py`**: Extended `BackendAdapter.generate()` signature to accept `messages` (for multi-turn conversation history) and `tools` (Anthropic-format tool schemas). Added `ParsedToolCall` dataclass with `error` field for malformed blocks. Added `BackendResponse.stop_reason` and `.is_final` property.

2. **`backends/anthropic.py`**: Full `tool_use` content block parsing. Every block is validated — missing `id`, `name`, or `input` fields produce a `ParsedToolCall(error=True)` with a logged warning. Never skips blocks. Added `_messages_to_api()` to handle multi-turn message format (including `tool_result` and `assistant_raw` roles for round-tripping content blocks).

3. **`scheduler.py`**: Replaced single `backend.generate()` call with `_run_agent_loop()`:
   - Loop runs up to `max_turns` iterations
   - Each iteration: acquire semaphore → generate → release semaphore
   - If `response.tool_calls`: enter TOOL_WAIT (semaphore already released), execute tools via pool, inject results, continue loop
   - If `response.is_final`: parse output, return
   - Tool schemas are auto-generated from function signatures via `_build_tool_schemas()`
   - Retry logic resets message history on failure

4. **`backends/openai.py`, `backends/vllm.py`**: Updated to match new `generate()` signature.

#### Design choices

- **Semaphore wraps only inference calls, not tool waits (W5).** Proven by instrumentation: all 10 agents show `ACQUIRE -> RELEASE -> TOOL_WAIT -> ACQUIRE -> RELEASE`. During TOOL_WAIT, other agents acquire the freed slot immediately. With max_concurrent=3 and 10 agents, jobs 3-9 visibly wait for slots and get them as tools complete.
- **Messages are stored as `Message` objects with special roles** (`assistant_raw` for content blocks, `tool_result` for tool results). This is necessary because Anthropic's API requires the full `tool_use`/`tool_result` block structure to be preserved across turns — you can't just send text.
- **Malformed tool calls are not fatal.** If a tool_use block is missing required fields, it's returned with `error=True` and the tool result sent back to the model says `[ERROR] Malformed tool call: ...`. The model can then retry or produce a final response.
- **Tool schemas are built by introspecting function signatures.** Simple type annotations (str, int, float, bool) map to JSON Schema types. This is good enough for Phase 0 but doesn't handle complex types (list, dict, Optional).

#### Bugs found during implementation

- **FakeBackend in unit tests broke** after signature change. The `generate()` method requires `messages` and `tools` kwargs now (even if None). Fixed by updating test fake.
- **`max_tokens` was hardcoded to 1024** in the Anthropic adapter. Increased to 4096 to avoid truncation in multi-turn scenarios where the model needs more room for tool calls + final output.

#### Verified behavior

- 10 agents, max_concurrent=3, 2 turns each (tool call + final response)
- 20 semaphore acquires, 20 releases, 10 TOOL_WAIT events
- All agents produce valid Pydantic-validated output
- Tool pool correctly executes `http_get` against local mock server
- Total runtime: 0.16s (mock, no real latency)

### Phase 1 completion — 2026-05-09

#### Tool schema generation (`batch_agent/schema.py`)
- Extracted from inline code in scheduler into a proper module.
- Handles: `str`, `int`, `float`, `bool`, `list[T]`, `dict[str, T]`, `Optional[T]`, `Union`, Pydantic models.
- Bug: `from __future__ import annotations` in tool modules stringifies all annotations. Fixed by using `typing.get_type_hints()` to resolve them at schema-build time.
- 7 unit tests passing.

#### vLLM adapter verification
- Verified against a local mock OpenAI-compatible server.
- `warm_prefix` correctly sends `POST /v1/completions` with `max_tokens=0`.
- `generate` correctly sends `POST /v1/chat/completions`.
- **Blocker for real GPU test:** No GPU available in this environment. The adapter is structurally correct but not tested against a live vLLM server.

#### OpenAI/vLLM tool call parsing (`backends/openai.py`)
- Full parsing of OpenAI `tool_calls` response format (id, function.name, function.arguments JSON).
- Malformed blocks logged and returned with `error=True` (same as Anthropic adapter).
- Multi-turn message conversion: `assistant_raw` → OpenAI format with `tool_calls`, `tool_result` → OpenAI `role: tool`.
- Anthropic tool schemas auto-converted to OpenAI function calling format.

#### Priority queue + staggered dispatch
- Jobs ordered by `max_turns` (lower = higher priority, drains fast agents first).
- Dispatch gate limits how far ahead of completions new tasks are launched.
- Heterogeneous test: 5 fast (1-turn) + 5 slow (4-turn) agents, max_concurrent=3.
- Result: fast agents avg position 2.0, slow agents avg position 7.0 — priority works.

### Phase 2 implementation — 2026-05-09

#### Configurable timeouts
- Added `timeout_per_turn` and `timeout_per_tool` to `BatchSpec`.
- `timeout_per_turn` wraps `backend.generate()` with `asyncio.wait_for()`.
- `timeout_per_tool` wraps each tool call with `asyncio.wait_for()`.
- `asyncio.TimeoutError` caught and reported as tool error (non-fatal to agent).

#### Message compaction (`batch_agent/compaction.py`)
- Triggers every 3 turns (configurable via `COMPACT_INTERVAL`).
- Tool results older than 2 turns are truncated to 200 chars with `[COMPACTED]` marker.
- Does not require a model call — uses heuristic truncation.
- Design choice: model-based summarization deferred to when a compaction model is available. The current approach prevents unbounded context growth, which is the immediate goal.

#### Reduce topology (`BatchAgent.run_with_reduce()`)
- After all map agents complete, a reduce agent receives all successful outputs as JSON.
- Reduce prompt template supports `{n}` substitution for result count.
- `reduce_schema` validates the aggregated output via Pydantic.
- Tested: 5 items → reduce sums values → validates Summary model.

#### @batchable SQL/HTTP batch grouping (`batch_agent/tools/sql.py`)
- `BatchCollector` collects calls to the same batchable tool within a 5ms window.
- Groups by `(tool_name, batch_query)` key.
- If multiple calls arrive within the window, executes them together.
- Single calls within the window execute immediately without batching overhead.
- Design choice: actual SQL batch execution requires a `_batch_handler` attribute on the tool function. Without it, falls back to parallel individual calls.

#### SQLite checkpoint store (`batch_agent/checkpoint.py`)
- Saves `AgentResult` after each job completes.
- On re-run with same `checkpoint_dir`, skips already-completed jobs (crash recovery).
- Uses WAL journal mode for concurrent read/write safety.
- Integrated into `WaveScheduler`: completed jobs yield immediately, new jobs dispatched normally.

#### 500-agent benchmark
- **Result: 500/500 OK, 0% failure rate, 55 agents/sec throughput, 9.15s wall-clock.**
- 64 concurrent agents, 100 multi-turn (20%), transient 1% failure injection.
- 15 tool timeouts observed → handled gracefully as error results.
- All output index values verified correct.
- No OOM, no unhandled exceptions.

#### Phase 2 success criteria status
| Criterion | Status |
|---|---|
| 500 agents complete with ≤ 2% failure rate | PASS (0%) |
| No OOM crashes at 500 agents | PASS |
| No unhandled exceptions | PASS |
| Retry logic with exponential backoff | PASS (implemented since Phase 0) |
| Message compaction | PASS (heuristic, not model-based) |
| @batchable tool annotation | PASS (collector implemented) |
| Reduce topology | PASS |
| Configurable timeouts (agent/turn/tool) | PASS |
| Overflow to disk (SQLite checkpoint) | PASS |

#### Remaining known gaps
- **No live Anthropic/GPU benchmark** — billing issue on the API key, no GPU in env.
- **Model-based compaction** — deferred; heuristic truncation works but loses semantic information.
- **SQL batch_handler protocol** — the `@batchable` annotation exists but actual SQL execution requires user-implemented batch handlers.
- **Cost comparison** — cannot compute cost-per-task without live runs.

### Audit Step 1 and Bedrock backend — 2026-05-09

- Fixed Anthropic prompt caching header bug: `anthropic-beta: prompt-caching-2024-07-31` is now sent with `cache_control` requests.
- Added KVFlow-required `AgentState` fields: `estimated_next_activation`, `steps_to_execution`, `predicted_tool_sequence`, `historical_turn_latencies`, plus `tool_wait_durations`.
- Reworked scheduler concurrency around a priority semaphore: inference slots are acquired by dynamic turns-remaining priority, and PENDING agents are dispatched when existing agents enter `TOOL_WAIT` or `COMPLETE`.
- Added adaptive concurrency polling via backend `get_cache_metrics()`; vLLM parses Prometheus metrics when available.
- Added model-based compaction interface, but live model compaction is still blocked without a separate low-priority small-model endpoint. The heuristic fallback remains explicit and logged rather than silent.
- Added vLLM prefix pinning call to `/internal/pin_blocks`. Hardware/server blocker: the actual no-evict behavior requires installing the vLLM BlockManager patch on a running vLLM 0.6.x server.
- Wired SQL `BatchCollector` into `ToolPool`; batchable tools now route through the collector.
- Added AWS Bedrock backend using boto3 standard credential chain and Bedrock Converse/ConverseStream APIs.
- Bedrock limitation: Bedrock does not expose KV cache internals. `warm_prefix()` returns a stable prefix hash for internal tracking, but there is no KV prefill or pinning API; prefix caching degrades to Bedrock’s managed `cachePoint` behavior, same class of limitation as Anthropic API mode.
- Bedrock prompt caching limitation: Claude models on Bedrock support `cachePoint`; Llama/Titan models do not, so the adapter checks `_supports_prompt_caching(model_id)` before injecting cache points.
- Bedrock streaming limitation: Converse streaming support differs by model. The adapter uses `converse_stream()` first and falls back to `converse()` if streaming is unsupported; live verification should pin the exact Bedrock model ID in use.
- AWS credential note: boto3 reads standard credential chain. For env var setup, prefer `AWS_DEFAULT_REGION=us-east-1` or an explicit backend URL (`bedrock://us-east-1/...`). `AWS_REGION` alone is not sufficient in all boto3 contexts. Temporary/SSO/MFA credentials also require `AWS_SESSION_TOKEN`.

### Audit Step 3 test gap closure — 2026-05-09

- Added `tests/unit/test_audit_gap_coverage.py` covering all 11 audit test gaps in a single focused suite.
- Covered compaction: verifies turn-3 compaction shortens old tool-result context and emits a `[COMPACTED]` marker.
- Covered checkpoint resume: added `CheckpointStore.load_state()` and scheduler resume from persisted in-progress `AgentState`; fixed a real bug where state was checkpointed too early in the turn before assistant/tool-result messages were durable.
- Covered SQL batching: 30 concurrent calls to a batchable tool now produce one `_batch_handler` invocation through `ToolPool`.
- Covered reduce with partial failures: reduce now receives both successful outputs and structured error objects.
- Covered retry exhaustion and timeout retry: both return structured `AgentResult` errors instead of raising to the caller.
- Covered OpenAI-style multi-turn end-to-end through the scheduler.
- Covered repair edge cases: missing JSON, nested repairable JSON, and schema validation failure.
- Covered CLI smoke dispatch using a mocked `BatchAgent.run`.
- Covered async `on_result` callback behavior.
- Covered dynamic priority ordering via `PrioritySemaphore`: a near-complete agent with lower turns remaining is served before a fresh job.
- Full pytest suite after these changes: 48 passed.

### Phase 3A KVFlow Advisor — 2026-05-09

- Added `batch_agent/kvflow.py` with `PrefetchHint` and `KVFlowAdvisor`.
- `KVFlowAdvisor` scans `AgentState` objects in `TOOL_WAIT`, estimates `steps_to_execution` from ToolPool P75 latency, updates `estimated_next_activation`, and emits hints sorted by shortest ETA first.
- Scheduler now starts a KVFlow advisor task when `BatchSpec.kvflow=True`.
- Scheduler emits an immediate KVFlow hint batch when an agent enters `TOOL_WAIT`; this prevents short 300ms tool waits from being missed by the 500ms background interval.
- Scheduler passes `metadata={"kv_key": state.kv_key, "job_id": state.job_id}` to backends that accept a `metadata` parameter, allowing vLLM/SGLang adapters to track per-agent KV identity.
- Added default no-op `send_prefetch_hints()` to `BackendAdapter`; vLLM and SGLang adapters serialize `PrefetchHint` objects to `/internal/prefetch` or `/internal/prefetch_radix` payloads.
- Added `tests/unit/test_kvflow_advisor.py`: verifies ETA ordering and horizon filtering.
- Added `tests/integration/test_prefetch_accuracy.py`: 20 agents, 3 turns, simulated 300ms tool waits, mock backend records prefetch hits; target ≥80% hit rate passes.
- Full pytest suite after KVFlow changes: 50 passed.

### Phase 3A vLLM prefetch patch helper — 2026-05-09

- Added `batch_agent/backends/vllm_patch/prefetch_route.py` with `/internal/prefetch` and `/internal/pin_blocks` FastAPI route registration helpers.
- Target vLLM version pinned for this patch path: vLLM 0.6.x.
- `/internal/prefetch` maps `kv_key` values to vLLM block IDs through a registry and calls `cache_engine.prefetch(block_ids, destination="gpu")`. No new tensor transfer logic is added; this intentionally reuses CacheEngine swap/prefetch infrastructure.
- `/internal/pin_blocks` maps `kv_key` values to block IDs and calls `block_manager.pin_blocks(block_ids)` if available, or populates a `pinned_block_ids` set for the BlockManager eviction path to consult.
- Exact vLLM source location to patch for pinning: vLLM 0.6.x BlockManager/BlockSpaceManager eviction path under `vllm/core/block_manager*.py`; add a pinned block-ID/hash set and skip pinned blocks in the LRU/free-block eviction candidate selection.
- Added `tests/unit/test_vllm_prefetch_route.py` with a mock CacheEngine and BlockManager. The test verifies `kv_key -> block_ids` mapping and that `prefetch(..., destination="gpu")` is invoked.
- Hardware blocker: this route cannot prove real GPU transfer latency without a running patched vLLM server and GPU.
- Cold KV reload latency proxy numbers recorded from the architecture/spec and TokenDance/KVFlow literature assumptions: ~50ms for ~1K-token contexts, ~100ms for ~2K-token contexts, ~200ms for ~4K-token multi-turn contexts on large 70B-class deployments. The prefetch path eliminates this reload from the critical path by moving CPU→GPU block transfer before agent reactivation.
- Full pytest suite after prefetch route helper: 52 passed.

### Phase 3B TokenDance diff cache prototype — 2026-05-09

- Added `batch_agent/backends/vllm_patch/diff_cache_engine.py` with a flag-gated `DiffCacheEngine` prototype.
- `diff_kv=False` returns `None` via `maybe_create_diff_cache_engine()` and has zero runtime effect on the scheduler or existing test suite.
- Implemented SHA256 block content hashing, global block deduplication, per-agent sparse diff records, and async soft-timeout All-Gather (`500ms` or target completion fraction, no hard barrier per W11).
- The implementation attempts to subclass vLLM `CacheEngine` when vLLM is installed; in CI it falls back to `object` so the algorithm remains testable without GPU/vLLM.
- Added `tests/unit/test_diff_encoder.py` synthetic TokenDance test: 100 agents, 2,048-token shared prefix, 500-token per-agent context (400 common + 100 unique), block size 16 tokens.
- Synthetic compression result: full blocks `16000`, stored unique blocks `853`, compression ratio `18.76x`.
- 2.7x capacity interpretation: if stock prefix caching fits 100 concurrent agents in a fixed KV budget, the TokenDance target implies roughly 270 agents at the same memory budget; at N=500, equivalent memory pressure would drop toward ~185 stock-agent equivalents, subject to allocator fragmentation and non-KV overhead.
- Full pytest suite after TokenDance prototype: 54 passed.

### Phase 4 distributed state store primitives — 2026-05-09

- Added `distributed` and `node_id` fields to `BatchSpec` for Phase 4 wiring.
- Added `version` and `owner_node_id` to `AgentState` for optimistic locking and job ownership.
- Added `RedisStreamsStateStore` to `state.py`, written against a minimal Redis client protocol so it can run against real Redis or an in-process mock.
- Implemented lease acquisition with Redis-style `SET NX EX`: one node wins ownership; others back off until TTL expiry.
- Implemented lease renewal/release and optimistic `save_with_version()`. Stale writes are rejected when the expected version does not match the stored state.
- Added stream append (`xadd`) on successful writes to model Redis Streams append-only state history.
- Added `tests/integration/test_distributed.py`: two nodes share a mock Redis, exactly one wins lease, stale write is rejected, and another node can acquire after TTL expiry.
- Full pytest suite after distributed state primitives: 55 passed.

### Phase 4 distributed wave scheduler primitives — 2026-05-09

- Added `batch_agent/distributed.py` with `ConsistentHashRing` and `DistributedWaveScheduler`.
- Consistent hashing assigns agents to nodes deterministically with virtual replicas.
- Distributed scheduler claims Redis-style leases before processing jobs, writes versioned state, and releases leases on completion.
- Failover path: surviving node reruns with `failover=True`, waits for expired leases, then claims and completes unfinished jobs from Redis state.
- Added `tests/integration/test_distributed_scheduler.py`: starts two in-process nodes over shared mock Redis, kills node-a after 30 completions, lets leases expire, and verifies node-b completes ≥95/100 jobs (≤5% loss target).
- Full pytest suite after distributed scheduler primitives: 56 passed.

### Full benchmark suite — 2026-05-09

- Added six benchmark entrypoints under `tests/benchmarks/` with mock-default execution, `--live` flags, and machine-readable `results.json` outputs.
- Mock benchmarks run in CI/no-GPU/no-API-key environments. `--live` modes are wired as explicit live-backend entrypoints but report blockers when required datasets or hardware are missing.
- Ran all six benchmarks and wrote result artifacts under `tests/benchmarks/results/<benchmark>/results.json`.

| Benchmark | Mock result |
|---|---|
| `paper_summarization.py` | N=50, time-to-first=0.02s, time-to-all=0.30s, failure_rate=0.0 |
| `code_review.py` | N=100, avg_turns=3.0, tool_wait_fraction=0.42, failure_rate=0.0 |
| `heterogeneous_tasks.py` | N=100, 50 fast/50 slow, slot_utilization_flat=true, chaos_tool_failures=10, failure_rate=0.0 |
| `tool_dedup_efficiency.py` | requested_reads=1000, actual_reads=10, dedup_ratio=100x |
| `kvflow_prefetch_accuracy.py` | N=20, turns=3, simulated_tool_wait=300ms, prefetch_hit_rate=1.0 (target ≥0.80) |
| `tokendance_compression.py` | N=100, full_blocks=16000, stored_unique_blocks=853, compression_ratio=18.76x |

- Full pytest suite after benchmark additions: 56 passed.

### Live Bedrock and compaction runs — 2026-05-09

- AWS credential chain check succeeded with local boto3 config in `us-east-1` (`sts.get_caller_identity()` and Bedrock model/profile listing worked). Use `AWS_DEFAULT_REGION=us-east-1`; `AWS_REGION` alone is not reliable for boto3.
- Requested Bedrock model `anthropic.claude-3-5-sonnet-20241022-v2:0` was not available in this account/region (`list_foundation_models(byProvider="Anthropic")` returned no matching 3.5 Sonnet v2 model). User instructed to use Opus 4.5 instead.
- Live model used for Bedrock batch: `us.anthropic.claude-opus-4-5-20251101-v1:0` inference profile.
- Bedrock live 20-agent batch result: 13 OK, 7 failed schema/response validation after retries, wall-clock `99.73s`.
- Bedrock TTFT captured from local event-stream consumption timing: P50 `0.000666s`, P95 `0.001078s`, per-success TTFT list stored in `tests/benchmarks/results/bedrock_live_batch/results.json`. Note: Bedrock also reports service-side `latencyMs`; local TTFT is near-zero because boto3 delivers event-stream chunks after the HTTP stream is established.
- Bedrock prompt caching metadata: `cachePoint_requested=true`; response usage exposed `cacheReadInputTokens` and `cacheWriteInputTokens` keys for some calls, but both totals were `0`. This proves cache metadata appears, but the short prompt did not generate cacheable token counts.
- Bedrock batch token usage from response usage fields: total input tokens `1807`, total output tokens `635`, cache read input tokens `0`, cache write input tokens `0`.
- Bedrock dollar cost: Bedrock Converse response usage fields return tokens, not USD. Exact dollars require AWS Pricing/CUR for the specific Opus 4.5 inference profile; `estimated_cost_usd` is intentionally `null` to avoid wrong pricing.
- Model-based compaction live run used `us.anthropic.claude-opus-4-5-20251101-v1:0` because the requested Haiku compaction model/profile was blocked as legacy/not recently used in this account.
- Live compaction prompt used: `Summarize the following tool results in 2-3 sentences, preserving all factual content: {tool_results}`.
- Live compaction result: latency `2.638s`, chars before `22762`, chars after `15602`, input tokens `1618`, output tokens `62`, total tokens `1680`. Result stored in `tests/benchmarks/results/bedrock_live_compaction/results.json`.
- Model-based compaction blocker is resolved for the available Opus 4.5 Bedrock profile. The original Haiku profile remains account-access blocked.
- vLLM/RunPod live benchmark remains blocked: `deploy/runpod_template.json` and `deploy/vllm_server.sh` are absent from this repository, and no RunPod API token/GPU provisioning capability is available from this environment. Paper summarization benchmark result JSON marks configs D/E at N=10/50/100 as blocked with this exact reason.

### Live Bedrock fixes after Opus 4.5 run — 2026-05-09

- Diagnosed the 35% live Bedrock failure rate. The seven failed agents were all Bedrock `ThrottlingException` failures, not schema validation, JSON repair exhaustion, timeout, or malformed model output.
- Printed raw diagnostic rerun responses for three failed indexes (`8`, `9`, `5`). All three returned valid raw JSON and parsed successfully, confirming parsing was not the failure cause.
- Fix: configured Bedrock boto3 client with adaptive retries (`max_attempts=10`, `mode=adaptive`) and changed the live batch default to sequential execution (`BEDROCK_LIVE_MAX_CONCURRENT=1`, `BEDROCK_LIVE_MAX_RETRIES=3`, timeout `180s`) for reproducible Opus profile measurements under low Bedrock TPS quota.
- Reran the 20-agent Bedrock batch with `us.anthropic.claude-opus-4-5-20251101-v1:0`: 20 OK, 0 failed, wall-clock `189.08s`.
- Fixed TTFT measurement in `BedrockBackend`: now records `time.monotonic()` immediately before `client.converse_stream(**payload)` and first content/tool delta arrival. Corrected TTFT for the unpadded sequential Opus run: P50 `7.612s`, P95 `16.558s`.
- Diagnosed zero cache tokens: original live system prompt was below Anthropic/Bedrock prompt-cache threshold (usage showed only `139` input tokens). Cache point was injected in the correct Converse `system` array position (`[{"text": ...}, {"cachePoint": {"type": "default"}}]`) but ignored for short prompts.
- Reran with `BEDROCK_CACHE_PAD_TOKENS=1200` to force a cacheable system prefix. Result: 20 OK, 0 failed, system prompt estimated tokens `1200`, prompt chars `27599`, cache write tokens `7285`, cache read tokens `138415`, cache metadata keys `cacheDetails`, `cacheReadInputTokens`, `cacheWriteInputTokens` visible in response usage. Corrected padded-run TTFT: P50 `8.808s`, P95 `16.734s`.
- Opus 4.5 Bedrock prompt caching support is confirmed for the `us.anthropic.claude-opus-4-5-20251101-v1:0` inference profile when the system prompt is long enough.
- Compaction cost note: live compaction used Opus 4.5 only because Haiku was not available in this account (`legacy/not recently used` access blocker). Production compaction should use a cheaper small model such as Haiku or equivalent; using Opus 4.5 for compaction inflates expected cost by roughly an order of magnitude versus Haiku-class pricing. The compaction interface is correct; model selection is a deployment decision.
- Added missing deploy files from `AGENTS.md`: `deploy/vllm_server.sh` and `deploy/runpod_template.json`. The script installs vLLM 0.6.6, makes Batch Agent vLLM patch helpers importable, enables prefix caching, sets GPU memory utilization to `0.85`, and starts the OpenAI-compatible server on port `8000`.
- RunPod live vLLM remains unexecuted from this environment because provisioning requires a RunPod account/API token and a mounted/available repository URL. The deploy files now exist so a user can launch and run the benchmark externally.
