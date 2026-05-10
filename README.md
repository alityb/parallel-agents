# BatchAgent

BatchAgent is a Python SDK for running many LLM agents against many inputs.

It is built for workloads like:

```text
"Read these 100 papers and return one structured report per paper."
"Review these 80 repositories and extract the same findings from each."
"Run 200 research agents, then reduce their answers into one synthesis."
```

The interface is deliberately small: input list in, structured result list out.
BatchAgent handles the agent loop, tool execution, retries, validation, result
streaming, and backend scheduling.

It works with API backends, but the main target is self-hosted inference through
vLLM, SGLang, and NVIDIA Dynamo, where shared prefixes and request scheduling
actually matter.

## Why this exists

There are good tools for parallel coding agents, and good tools for fast LLM
serving. The gap is the layer between them.

Existing orchestration tools can spawn many agents, but usually do not reason
about inference efficiency. Inference engines like vLLM and SGLang can reuse KV
cache and batch requests, but they do not know that 100 requests are sibling
agents in the same workflow.

BatchAgent is that orchestration layer:

- Extract and reuse a shared system prompt prefix.
- Run N multi-turn agents concurrently.
- Release inference slots while agents wait on tools.
- Coalesce duplicate tool calls.
- Return Pydantic-validated outputs instead of raw strings.
- Stream each result as soon as it finishes.
- Attach Dynamo `nvext.agent_hints` when using Dynamo.

This is not a general agent framework. It is a batch execution engine for
multi-agent workloads.

## Install

```bash
pip install batch-agent
```

Optional extras:

```bash
pip install "batch-agent[bedrock]"
pip install "batch-agent[vllm]"
pip install "batch-agent[redis]"
pip install "batch-agent[dashboard]"
```

Requires Python 3.10+.

## Basic Usage

```python
from batch_agent import BatchAgent, Tool
from pydantic import BaseModel

class PaperSummary(BaseModel):
    title: str
    benchmark: str
    main_result: str

results = await BatchAgent.run(
    task="Summarize this paper and extract benchmark details:\n\n{paper}",
    inputs=[{"paper": text} for text in papers],
    tools=[Tool.web_search, Tool.read_file],
    output_schema=PaperSummary,
    model="Qwen/Qwen2.5-7B-Instruct",
    backend="sglang://localhost:30000",
    max_inflight=32,
    max_turns=3,
)

for result in results:
    if result.ok:
        print(result.output.benchmark)
    else:
        print(result.error)
```

Stream results as they finish:

```python
async for result in BatchAgent.stream(...):
    handle(result)
```

Plan -> map -> reduce:

```python
results, synthesis = await BatchAgent.run_with_map_reduce(
    plan_prompt="Generate 20 research questions about: {topic}",
    plan_inputs={"topic": "KV cache optimization"},
    plan_output_schema=ResearchPlan,   # must expose items: list[str]
    task="Answer this research question: {item}",
    output_schema=ResearchAnswer,
    reduce="Synthesize the {n} answers into a survey.",
    reduce_schema=SurveyPaper,
    tools=[Tool.web_search],
    model="Qwen/Qwen2.5-7B-Instruct",
    backend="sglang://localhost:30000",
)
```

## Backends

| Backend | URL | Status |
|---|---|---|
| vLLM | `vllm://localhost:8000` | Live A10G tested |
| SGLang | `sglang://localhost:30000` | Live A10G tested |
| NVIDIA Dynamo | `dynamo://localhost:8000` | Live A10G smoke tested with `nvext.agent_hints` |
| Anthropic API | `anthropic://` | Implemented, API-mode only |
| OpenAI API | `openai://` | Implemented, API-mode only |
| AWS Bedrock | `bedrock://us-east-1` | Live tested, conservative concurrency |

API backends are useful, but degraded: you do not control the inference
scheduler or KV cache. The strongest path is self-hosted vLLM/SGLang/Dynamo.

## What BatchAgent Optimizes

| Optimization | Implemented | Verified |
|---|---:|---|
| Multi-turn agent loop | yes | tests |
| Structured Pydantic output | yes | tests |
| Tool coalescing | yes | live slow-tool benchmark |
| Release inference slots during tool wait | yes | live slow-tool benchmark |
| Shared-prefix cache usage | backend-dependent | vLLM/SGLang live |
| Streaming result delivery | yes | tests |
| Dynamo `nvext.agent_hints` | yes | live smoke test |
| KVFlow-style prefetch | hints only | not verified |
| TokenDance-style diff KV | prototype | mock only |
| Distributed Redis orchestration | prototype | mock Redis only |

Important distinction: tool deduplication is not KV-cache magic. In the N=100
slow-tool benchmarks below, 100 agents intentionally use 10 repeated tool keys,
so the maximum possible tool saving is 90 calls. That benchmark proves concurrent
tool coalescing and scheduling behavior, not KVFlow prefetch.

## Results

All numbers below are from result JSON files committed under
`tests/benchmarks/results/`.

### Raw SGLang/Dynamo vs BatchAgent

Workload: 100 two-turn agents, 2048-token shared system prompt, one simulated
800ms tool wait per agent, 10 unique repeated tool queries, Qwen2.5-7B on one
A10G.

Source: `tests/benchmarks/results/backend_raw_vs_batchagent/`

| Backend | Mode | Wall | Agents/s | Tool calls | Cache hit |
|---|---|---:|---:|---:|---:|
| SGLang standalone | raw endpoint loop | 14.73s | 6.79 | 100/100 | 98.50% |
| SGLang standalone | BatchAgent | 8.40s | 11.90 | 10/100 | 98.57% |
| Dynamo + SGLang worker | raw endpoint loop | 13.02s | 7.68 | 100/100 | n/a |
| Dynamo + SGLang worker | BatchAgent | 8.41s | 11.88 | 10/100 | n/a |

Interpretation: raw SGLang/Dynamo is the right baseline for one-shot inference.
BatchAgent wins on this multi-turn workload because duplicate tool calls are
coalesced and inference concurrency is not held while tools are waiting.

### vLLM Slow-Tool Benchmark

Workload: same shape as above, vLLM 0.6.6.post1, Qwen2.5-7B, one A10G.

Source: `tests/benchmarks/results/definitive_benchmark/results.json`

| Mode | Wall | Agents/s | Tool calls | Prefix cache hit |
|---|---:|---:|---:|---:|
| Naive `asyncio.gather` | 13.07s | 7.65 | 100/100 | 98.68% |
| BatchAgent | 9.65s | 10.36 | 10/100 | 98.95% |

BatchAgent was 3.42s faster, a 26.2% wall-clock reduction, while preserving a
high shared-prefix cache hit rate.

### SGLang Live Probe

Source: `tests/benchmarks/results/sglang_live/sglang_benchmark.json`

| N | OK | Wall | Agents/s | Prefix cache hit |
|---:|---:|---:|---:|---:|
| 10 | 10/10 | 2.37s | 4.22 | 98.80% |
| 50 | 50/50 | 2.99s | 16.75 | 98.81% |

### Dynamo Live Probe

Source: `tests/benchmarks/results/dynamo_live/dynamo_benchmark.json`

| Request | Result | Wall |
|---|---:|---:|
| Unary with `nvext.agent_hints` | OK | 0.148s |
| Streaming with `nvext.agent_hints` | OK | 0.142s |

This verifies the adapter path against a live Dynamo frontend and SGLang worker.
It is not a full Dynamo throughput benchmark.

## Architecture

```text
BatchAgent.run(...)
    |
    v
TaskCompiler
    - builds one job per input
    - extracts shared prefix
    - injects output schema
    |
    v
WaveScheduler
    - runs inference turns with bounded concurrency
    - releases the semaphore during TOOL_WAIT
    - prioritizes near-complete agents
    - emits KVFlow/Dynamo scheduling hints
    |
    v
ToolPool
    - coalesces concurrent identical calls
    - caches deterministic tools
    - rate-limits external tools
    |
    v
Backend adapter
    - vLLM / SGLang / Dynamo / API backend
    - parse tool calls
    - collect metrics where available
```

The critical scheduling invariant is simple: the inference semaphore wraps model
calls only. Tool waits do not occupy GPU request capacity.

## Limitations

- The strongest live numbers are for Qwen2.5-7B on a single A10G. 70B and
  multi-node results are not done.
- Tool coalescing only saves calls when multiple agents ask for the same
  cacheable tool call at overlapping times.
- KVFlow prefetch is not verified. The A/B/C measurement rerun showed no
  prefetch-specific improvement from the current vLLM API-route approach.
  Correct vLLM prefetch needs scheduler-integrated CPU->GPU block mappings.
  Source: `tests/benchmarks/results/kvflow_measurement_integrity/results.json`.
- TokenDance diff-KV is still a mock/prototype path. The synthetic compression
  number is not a live vLLM result.
- Distributed mode uses Redis abstractions but has not been validated as a real
  multi-node 1,000-agent deployment.
- API backends cannot expose the same KV/cache controls as self-hosted
  inference engines.

## Research Context

BatchAgent is motivated by the systems gap between orchestration and inference:
orchestrators do not manage KV/cache efficiency, and inference engines do not
understand multi-agent workflow structure.

Relevant systems:

- vLLM: continuous batching, PagedAttention, automatic prefix caching.
- SGLang: RadixAttention for token-level prefix reuse.
- NVIDIA Dynamo: production inference serving with request extensions such as
  `nvext.agent_hints`.
- LMCache: hierarchical KV cache movement across GPU, CPU, disk, and remote
  storage.

Related orchestration and API baselines:

- Agent Orchestrator: parallel coding agents in isolated workspaces.
- Sculptor and amux: local supervision UIs/TUIs for parallel coding agents.
- OpenAI Batch API: asynchronous batch processing for offline request batches;
  useful for bulk completions, but not a substitute for low-latency multi-turn
  tool-using agents.

Relevant papers and docs:

- TokenDance: `https://arxiv.org/abs/2604.03143`
- KVFlow: `https://arxiv.org/abs/2507.07400`
- LMCache: `https://arxiv.org/abs/2510.09665`
- vLLM prefix caching: `https://docs.vllm.ai/en/stable/design/prefix_caching.html`
- Agent Orchestrator: `https://github.com/ComposioHQ/agent-orchestrator`
- OpenAI Batch API: `https://platform.openai.com/docs/guides/batch/overview`

## Development

Run tests:

```bash
pytest -q
```

Run the raw-backend vs BatchAgent benchmark:

```bash
PYTHONPATH=. python tests/benchmarks/backend_raw_vs_batchagent.py \
  --base-url http://127.0.0.1:30000 \
  --label sglang_standalone \
  --n 100 \
  --tool-latency 800 \
  --system-prompt-tokens 2048 \
  --max-inflight 32
```

Run the vLLM definitive benchmark:

```bash
PYTHONPATH=. python tests/benchmarks/definitive_benchmark.py \
  --backend vllm://localhost:8000 \
  --n 100 \
  --slow-tools \
  --tool-latency 800 \
  --system-prompt-tokens 2048
```
