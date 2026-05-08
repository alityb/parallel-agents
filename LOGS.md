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
