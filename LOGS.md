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
