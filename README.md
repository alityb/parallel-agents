# Batch Agent SDK

Phase 0 foundation for the Batch Agent SDK described in `AGENTS.md`.

This repository currently provides:

- `BatchSpec`, `AgentJob`, `ExecutionPlan`, and agent state dataclasses.
- Task compilation with shared prefix extraction, schema instruction injection, and token budget flags.
- A one-shot async wave scheduler that streams results as jobs finish and returns failures as data.
- Tool definitions, builtin tools, request coalescing, LRU-style TTL caching, and token-bucket rate limiting.
- HTTP backend adapters for Anthropic and OpenAI-compatible endpoints, plus vLLM/SGLang wrappers.
- JSON extraction/repair and Pydantic validation.
- A minimal `batch-agent run --spec spec.yaml` CLI.

The implementation intentionally stays Phase 0: no Redis, no persistent memory, no patched vLLM cache engine, and no multi-turn tool-call protocol yet.
