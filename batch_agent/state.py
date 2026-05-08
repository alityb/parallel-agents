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

    def set_status(self, status: AgentStatus) -> None:
        self.status = status
        self.last_updated = time.time()


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
