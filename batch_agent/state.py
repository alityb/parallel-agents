from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .spec import AgentError, Message, ToolCall, ToolResult
from .utils import p75


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
    version: int = 0                                                       # optimistic-lock version
    owner_node_id: str | None = None                                      # distributed lease owner

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
        return p75(self.historical_turn_latencies)

    def p75_tool_wait(self) -> float | None:
        """Return P75 tool-wait duration, or None if insufficient data."""
        return p75(self.tool_wait_durations)


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


class RedisStreamsStateStore:
    """Redis-style distributed state store with leases and optimistic locking.

    This class is written against a minimal Redis client protocol (`get`, `set`,
    `delete`, `xadd`) so it can be tested with an in-process mock Redis. With a
    real Redis client, use redis-py methods with compatible parameters.
    """

    def __init__(self, redis_client: object, *, node_id: str) -> None:
        self.redis = redis_client
        self.node_id = node_id

    def acquire_lease(self, job_id: str, ttl_seconds: float) -> bool:
        key = self._lease_key(job_id)
        return bool(self.redis.set(key, self.node_id, nx=True, **_redis_ttl_kwargs(ttl_seconds)))

    def renew_lease(self, job_id: str, ttl_seconds: float) -> bool:
        key = self._lease_key(job_id)
        owner = self.redis.get(key)
        if owner != self.node_id:
            return False
        self.redis.set(key, self.node_id, **_redis_ttl_kwargs(ttl_seconds))
        return True

    def release_lease(self, job_id: str) -> None:
        key = self._lease_key(job_id)
        if self.redis.get(key) == self.node_id:
            self.redis.delete(key)

    def load(self, job_id: str) -> AgentState | None:
        raw = self.redis.get(self._state_key(job_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return _state_from_json(raw)

    def save_with_version(self, state: AgentState, expected_version: int) -> bool:
        current = self.load(state.job_id)
        if current is not None and current.version != expected_version:
            return False
        state.version = expected_version + 1
        state.owner_node_id = self.node_id
        self.redis.set(self._state_key(state.job_id), _state_to_json(state))
        if hasattr(self.redis, "xadd"):
            self.redis.xadd("agent_state_stream", {"job_id": state.job_id, "version": state.version})
        return True

    def _state_key(self, job_id: str) -> str:
        return f"agent:{job_id}:state"

    def _lease_key(self, job_id: str) -> str:
        return f"agent:{job_id}:lease"


def _redis_ttl_kwargs(ttl_seconds: float) -> dict[str, int]:
    if ttl_seconds <= 0:
        raise ValueError("Redis lease TTL must be positive")
    if float(ttl_seconds).is_integer():
        return {"ex": int(ttl_seconds)}
    return {"px": max(1, int(ttl_seconds * 1000))}


def _state_to_json(state: AgentState) -> str:
    return json.dumps({
        "job_id": state.job_id,
        "status": state.status.value,
        "turn": state.turn,
        "messages": [{"role": m.role, "content": m.content} for m in state.messages],
        "kv_key": state.kv_key,
        "retry_count": state.retry_count,
        "created_at": state.created_at,
        "last_updated": state.last_updated,
        "estimated_next_activation": state.estimated_next_activation,
        "steps_to_execution": state.steps_to_execution,
        "predicted_tool_sequence": state.predicted_tool_sequence,
        "historical_turn_latencies": state.historical_turn_latencies,
        "tool_wait_durations": state.tool_wait_durations,
        "version": state.version,
        "owner_node_id": state.owner_node_id,
    })


def _state_from_json(raw: str) -> AgentState:
    data = json.loads(raw)
    return AgentState(
        job_id=data["job_id"],
        status=AgentStatus(data["status"]),
        turn=data.get("turn", 0),
        messages=[Message(role=m["role"], content=m["content"]) for m in data.get("messages", [])],
        kv_key=data.get("kv_key"),
        retry_count=data.get("retry_count", 0),
        created_at=data.get("created_at", time.time()),
        last_updated=data.get("last_updated", time.time()),
        estimated_next_activation=data.get("estimated_next_activation"),
        steps_to_execution=data.get("steps_to_execution"),
        predicted_tool_sequence=data.get("predicted_tool_sequence"),
        historical_turn_latencies=data.get("historical_turn_latencies", []),
        tool_wait_durations=data.get("tool_wait_durations", []),
        version=data.get("version", 0),
        owner_node_id=data.get("owner_node_id"),
    )
