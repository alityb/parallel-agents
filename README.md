# BatchAgent

BatchAgent is a Python SDK for running many tool-using LLM agents against one
shared inference backend.

It is built for workloads like:

```text
Review these 100 files and return one structured bug report per file.
Research these 100 papers and return one page of findings per paper.
Run 50 coding/research agents, stream each answer, then reduce them into one synthesis.
```

The point is simple: if 100 agents share the same system prompt, tool set, and
workflow shape, they should not each recompute the same prefill or execute the
same tool call independently. BatchAgent coordinates the agent loop, tool pool,
prefix warming, result streaming, and backend scheduling so vLLM, SGLang, or
Dynamo can reuse shared work.

This is not a general agent framework. It is a batch execution engine: input
list in, structured result list out.

## Results

### OpenCode sessions on shared SGLang

Hardware: H100, `Qwen/Qwen2.5-32B-Instruct`, real OpenCode CLI sessions
(`opencode run` subprocesses).

Condition A is isolated execution with a cache flush between sessions. Condition
C is parallel execution against one shared SGLang server with the shared prefix
warmed once.

| N | Isolated wall-clock | Shared SGLang wall-clock | Speedup | Prefill tokens isolated | Prefill tokens shared | Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 28.77s | 10.14s | 2.84x | 63,903 | 1,981 | 97% |
| 10 | 53.56s | 19.86s | 2.70x | 128,182 | 4,920 | 96% |
| 20 | 115.39s | 36.32s | 3.18x | 257,094 | 9,874 | 96% |
| 50 | 286.72s | 98.00s | 2.93x | 643,918 | 23,605 | 96% |
| 100 | 573.21s | 190.80s | 3.00x | 1,287,681 | 50,806 | 96% |

The prefill column is the core result. At N=100, isolated inference computes
1.28M prefill tokens. Shared SGLang computes 50K. Same output, 96% less prefill
compute.

### Real multi-turn research task

Hardware: H100, `Qwen/Qwen2.5-32B-Instruct`, 20 fixed arXiv paper IDs,
2048-token shared prompt target, two turns per agent, deterministic 400ms
`web_search` grouped by topic.

Condition A is a raw endpoint loop that holds its concurrency slot during tool
wait. Condition C is BatchAgent using the same backend forwards plus tool
coalescing and scheduler release during tool wait.

Source files:

- `tests/benchmarks/results/research_summary/dynamo_h100_results.json`
- `tests/benchmarks/results/research_summary/sglang_h100_results.json`
- `tests/benchmarks/results/research_summary/vllm_h100_results.json`

| Backend | Condition | Wall | TTFT P50 | TTFT P95 | Prefix cache hit | Tool dedup |
|---|---:|---:|---:|---:|---:|---:|
| Dynamo on SGLang | A | 5.105s | 0.201s | 1.381s | 93.9% | 1.00x |
| Dynamo on SGLang | C | 3.242s | 0.194s | 0.238s | 98.8% | 2.22x |
| SGLang | A | 6.361s | 0.187s | 2.590s | 0.0% | 1.00x |
| SGLang | C | 3.158s | 0.191s | 0.219s | 98.6% | 2.22x |
| vLLM | A | 4.765s | 0.265s | 1.175s | 96.2% | 1.00x |
| vLLM | C | 2.766s | 0.244s | 0.308s | 97.5% | 2.22x |

BatchAgent speedup on this task:

- Dynamo: 1.57x
- SGLang: 2.01x
- vLLM: 1.72x

In every BatchAgent run, 20 requested web searches were reduced to 9 actual
executions through tool coalescing.

## Why This Exists

Existing tools solve different halves of the problem:

- Orchestration tools can spawn many agents, but usually do not reason about KV
  cache, prefill, or backend scheduling.
- Inference engines can batch and cache requests, but they do not know that 100
  requests are sibling agents in the same workflow.

BatchAgent is the layer between them. It gives the inference backend the workload
shape it can optimize:

- shared system prompt prefix
- many sibling agent sessions
- bounded inference concurrency
- tool waits that do not occupy GPU request slots
- duplicate tool calls that can be coalesced
- scheduler hints for Dynamo-compatible backends

## Install

PyPI: `https://pypi.org/project/batch-agent/`

```bash
pip install batch-agent
```

From source:

```bash
git clone https://github.com/alityb/batchagent.git
cd batchagent
pip install -e .
```

Requires Python 3.10+.

## Basic Usage

```python
from batch_agent import BatchAgent, Tool
from pydantic import BaseModel

class PaperSummary(BaseModel):
    title: str
    contribution: str
    relevance: str

results = await BatchAgent.run(
    task="Summarize this paper and extract the key result:\n\n{paper}",
    inputs=[{"paper": text} for text in papers],
    tools=[Tool.web_search, Tool.read_file],
    output_schema=PaperSummary,
    model="Qwen/Qwen2.5-32B-Instruct",
    backend="sglang://localhost:30000",
    max_inflight=32,
    max_turns=3,
)
```

Stream results as they finish:

```python
async for result in BatchAgent.stream(...):
    print(result.job_id, result.output)
```

Plan -> map -> reduce:

```python
results, paper = await BatchAgent.run_with_map_reduce(
    plan_prompt="Generate 20 research questions about: {topic}",
    plan_inputs={"topic": "KV cache optimization for multi-agent inference"},
    plan_output_schema=ResearchPlan,  # items: list[str]
    task="Research this question: {item}",
    output_schema=ResearchAnswer,
    reduce="Synthesize the answers into a survey.",
    reduce_schema=SurveyPaper,
    tools=[Tool.web_search],
    model="Qwen/Qwen2.5-32B-Instruct",
    backend="sglang://localhost:30000",
)
```

OpenCode runtime:

```python
from batch_agent import BatchAgent
from batch_agent.runtimes import OpenCodeRuntime

results = await BatchAgent.run(
    runtime=OpenCodeRuntime(
        backend="sglang://localhost:30000",
        model="Qwen/Qwen2.5-32B-Instruct",
    ),
    task="Review this file for bugs: {file}",
    inputs=[{"file": f} for f in files],
    max_agents=20,
)
```

Each worker runs `opencode run` as a subprocess with
`OPENCODE_CONFIG_CONTENT` pointed at the shared SGLang/vLLM server.

## Backends

| Backend | URL | Notes |
|---|---|---|
| SGLang | `sglang://localhost:30000` | Best current path for shared-prefix OpenCode sessions |
| vLLM | `vllm://localhost:8000` | Live H100 tested with prefix caching |
| NVIDIA Dynamo | `dynamo://localhost:8001` | Live H100 tested with `nvext.agent_hints` path |
| Anthropic API | `anthropic://` | Degraded mode; no direct KV control |
| OpenAI API | `openai://` | Degraded mode; no direct KV control |
| AWS Bedrock | `bedrock://us-east-1` | Degraded mode; managed prompt caching only |

API backends are useful for compatibility, but the main performance story is
self-hosted inference where the SDK can align orchestration with the serving
engine.

## What BatchAgent Optimizes

| Optimization | Status | Verified |
|---|---|---|
| Multi-turn agent loop | implemented | tests |
| Structured Pydantic output | implemented | tests |
| Streaming result delivery | implemented | tests |
| Tool coalescing | implemented | H100 research-summary benchmark |
| Release inference slots during tool wait | implemented | H100 research-summary benchmark |
| Shared-prefix cache usage | backend-dependent | H100 SGLang/vLLM/Dynamo |
| OpenCode shared backend runtime | implemented | H100 OpenCode benchmark |
| Dynamo `nvext.agent_hints` | implemented | live Dynamo smoke + benchmark |
| KVFlow-style prefetch | hints only | not verified as a latency win |
| TokenDance-style diff KV | prototype | mock only |
| Distributed Redis orchestration | prototype | local/mock only |

## Architecture

```text
BatchAgent.run(...)
    |
    v
Task compiler
    - creates one job per input
    - separates shared prefix from per-agent input
    - injects output schema
    |
    v
Wave scheduler
    - runs model turns with bounded concurrency
    - releases the inference slot during TOOL_WAIT
    - streams completed results immediately
    - emits backend hints where supported
    |
    v
Tool pool
    - coalesces concurrent identical calls
    - caches deterministic calls
    - rate-limits external tools
    |
    v
Backend/runtime
    - SGLang / vLLM / Dynamo / API backend
    - OpenCode subprocess runtime
    - metrics collection
```

The key invariant: inference slots wrap model calls only. Tool waits do not hold
GPU request capacity.

## Limitations

- KVFlow prefetch is not verified. The current vLLM API-route approach cannot
  safely turn BatchAgent `kv_key` hints into CPU->GPU block mappings; that needs
  scheduler-level integration.
- TokenDance-style diff KV is a prototype. The compression number is synthetic,
  not a live vLLM `CacheEngine` result.
- Distributed Redis orchestration exists as a prototype but is not yet validated
  as a 1,000-agent, multi-node deployment.
- API backends cannot expose the same cache/scheduler controls as self-hosted
  vLLM, SGLang, or Dynamo.
- Tool coalescing only helps when agents request the same cacheable tool call
  during overlapping windows.

## Research Context

BatchAgent is motivated by the gap between agent orchestration and inference
serving:

- vLLM provides continuous batching, PagedAttention, and prefix caching.
- SGLang provides RadixAttention, which is well suited to shared-prefix and
  branching-context workloads.
- NVIDIA Dynamo exposes production serving hooks such as `nvext.agent_hints`.
- LMCache targets hierarchical KV movement across GPU, CPU, disk, and remote
  storage.
- TokenDance and KVFlow show the systems opportunity for multi-agent KV sharing
  and workflow-aware cache movement.

References:

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

Run the H100 research-summary benchmark:

```bash
PYTHONPATH=. python tests/benchmarks/bench_research_summary.py \
  --base-url http://localhost:30000 \
  --backend sglang://localhost:30000 \
  --label sglang_qwen25_32b_h100 \
  --max-inflight 8 \
  --tool-latency 400 \
  --system-prompt-tokens 2048
```

Run the raw-backend vs BatchAgent slow-tool benchmark:

```bash
PYTHONPATH=. python tests/benchmarks/backend_raw_vs_batchagent.py \
  --base-url http://localhost:30000 \
  --label sglang_h100 \
  --n 100 \
  --tool-latency 800 \
  --system-prompt-tokens 2048 \
  --max-inflight 32
```
