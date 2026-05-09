from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    name: str
    args: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class AgentError:
    type: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class AgentResult:
    job_id: str
    index: int
    output: Any | None
    error: AgentError | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class BatchSpec:
    task: str
    inputs: Sequence[Mapping[str, Any]]
    system_prompt: str = ""
    tools: Sequence[Any] = field(default_factory=tuple)
    output_schema: Any | None = None
    model: str = "claude-sonnet-4-20250514"
    backend: str = "anthropic://"
    max_concurrent: int = 10
    max_turns: int = 1
    max_retries: int = 3
    timeout_per_agent: float | None = 300
    timeout_per_turn: float | None = 60
    timeout_per_tool: float | None = 30
    min_response_tokens: int = 1024
    model_max_context: int = 200_000
    on_result: Callable[[AgentResult], Any] | None = None
    diff_kv: bool = False
    checkpoint_dir: str | Path | None = None
    no_hoist: bool = False
    reduce: str | None = None
    reduce_schema: Any | None = None

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must not be empty")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


@dataclass(frozen=True)
class SharedContext:
    prefix: str
    schema: dict[str, Any] | None = None
    hoisted_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentJob:
    job_id: str
    index: int
    input_data: Mapping[str, Any]
    prompt: str
    estimated_prompt_tokens: int
    oversized: bool = False
    max_turns: int = 1


@dataclass(frozen=True)
class ExecutionPlan:
    shared: SharedContext
    jobs: list[AgentJob]
    spec: BatchSpec
