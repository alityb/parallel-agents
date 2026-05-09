from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .spec import AgentError, Message, ToolCall, ToolResult


class AgentStatus(str, Enum):
    PENDING = "PENDING"
    PREFLIGHT = "PREFLIGHT"
    RUNNING = "RUNNING"
    TOOL_WAIT = "TOOL_WAIT"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class AgentState:
    job_id: str
    status: AgentStatus = AgentStatus.PENDING
    turn: int = 0
    messages: list[Message] = field(default_factory=list)
    tool_calls_pending: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    kv_key: str | None = None
    output: Any | None = None
    error: AgentError | None = None
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    # KVFlow fields (Phase 3A) — required by KVFlowAdvisor
    estimated_next_activation: float | None = None   # unix timestamp of predicted reactivation
    steps_to_execution: float | None = None          # seconds until this agent needs GPU again
    predicted_tool_sequence: list[str] | None = None # tool names from checkpoint history
    historical_turn_latencies: list[float] = field(default_factory=list)  # per-turn generate() durations (seconds)
    tool_wait_durations: list[float] = field(default_factory=list)        # per TOOL_WAIT durations (seconds)

    def set_status(self, status: AgentStatus) -> None:
        self.status = status
        self.last_updated = time.time()

    def record_turn_latency(self, seconds: float) -> None:
        """Append a generate() wall-clock duration to history."""
        self.historical_turn_latencies.append(seconds)

    def record_tool_wait(self, seconds: float) -> None:
        """Append a TOOL_WAIT wall-clock duration to history."""
        self.tool_wait_durations.append(seconds)

    def p75_turn_latency(self) -> float | None:
        """Return P75 generate latency, or None if insufficient data."""
        if not self.historical_turn_latencies:
            return None
        data = sorted(self.historical_turn_latencies)
        idx = int(len(data) * 0.75)
        return data[min(idx, len(data) - 1)]

    def p75_tool_wait(self) -> float | None:
        """Return P75 tool-wait duration, or None if insufficient data."""
        if not self.tool_wait_durations:
            return None
        data = sorted(self.tool_wait_durations)
        idx = int(len(data) * 0.75)
        return data[min(idx, len(data) - 1)]


class InMemoryStateStore:
    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}

    def create(self, job_id: str) -> AgentState:
        state = AgentState(job_id=job_id)
        self._states[job_id] = state
        return state

    def get(self, job_id: str) -> AgentState:
        return self._states[job_id]

    def save(self, state: AgentState) -> None:
        state.last_updated = time.time()
        self._states[state.job_id] = state

    def all(self) -> list[AgentState]:
        return list(self._states.values())

    def all_in_status(self, status: AgentStatus) -> list[AgentState]:
        return [s for s in self._states.values() if s.status == status]
