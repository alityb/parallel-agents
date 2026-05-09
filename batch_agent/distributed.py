"""Distributed Wave Scheduler primitives for Phase 4.

This module provides:
  - ConsistentHashRing for agent-to-node assignment
  - DistributedWaveScheduler, a lightweight distributed wrapper that uses
    RedisStreamsStateStore leases and optimistic locking.

It is intentionally conservative: the single-machine WaveScheduler remains the
default. Distributed mode is opt-in via BatchSpec.distributed / this class.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .backends import BackendAdapter
from .compiler import TaskCompiler
from .repair import parse_and_validate_output
from .spec import AgentJob, AgentResult, BatchSpec, ExecutionPlan, Message
from .state import AgentError, AgentState, AgentStatus, RedisStreamsStateStore


class NodeStopped(RuntimeError):
    pass


class ConsistentHashRing:
    def __init__(self, nodes: list[str], replicas: int = 100) -> None:
        if not nodes:
            raise ValueError("nodes must not be empty")
        self.nodes = nodes
        self.ring: list[tuple[int, str]] = []
        for node in nodes:
            for i in range(replicas):
                digest = hashlib.sha256(f"{node}:{i}".encode()).hexdigest()
                self.ring.append((int(digest[:16], 16), node))
        self.ring.sort(key=lambda x: x[0])

    def get_node(self, key: str) -> str:
        digest = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        for point, node in self.ring:
            if digest <= point:
                return node
        return self.ring[0][1]


@dataclass
class DistributedRunStats:
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    stopped: bool = False


class DistributedWaveScheduler:
    def __init__(
        self,
        *,
        spec: BatchSpec,
        backend: BackendAdapter,
        redis_client: Any,
        node_id: str,
        nodes: list[str],
        lease_ttl_seconds: float = 0.2,
    ) -> None:
        self.spec = spec
        self.plan = TaskCompiler().compile(spec)
        self.backend = backend
        self.store = RedisStreamsStateStore(redis_client, node_id=node_id)
        self.node_id = node_id
        self.nodes = nodes
        self.ring = ConsistentHashRing(nodes)
        self.lease_ttl_seconds = lease_ttl_seconds

    async def run(self, *, failover: bool = False, stop_after: int | None = None) -> DistributedRunStats:
        stats = DistributedRunStats()
        for job in self.plan.jobs:
            if stop_after is not None and stats.completed >= stop_after:
                stats.stopped = True
                raise NodeStopped(f"{self.node_id} stopped after {stats.completed} jobs")

            if not failover and self.ring.get_node(job.job_id) != self.node_id:
                stats.skipped += 1
                continue

            existing = self.store.load(job.job_id)
            if existing and existing.status == AgentStatus.COMPLETE:
                stats.skipped += 1
                continue

            if not self.store.acquire_lease(job.job_id, ttl_seconds=self.lease_ttl_seconds):
                stats.skipped += 1
                continue

            try:
                result = await self._run_one(job)
                state = AgentState(
                    job_id=job.job_id,
                    status=AgentStatus.COMPLETE if result.ok else AgentStatus.FAILED,
                    turn=1,
                    messages=[Message("user", job.prompt)],
                    output=_jsonable(result.output),
                    error=result.error,
                )
                current = self.store.load(job.job_id)
                expected_version = current.version if current else 0
                saved = self.store.save_with_version(state, expected_version=expected_version)
                if not saved:
                    stats.skipped += 1
                    continue
                if result.ok:
                    stats.completed += 1
                else:
                    stats.failed += 1
            finally:
                self.store.release_lease(job.job_id)
        return stats

    async def _run_one(self, job: AgentJob) -> AgentResult:
        try:
            response = await self.backend.generate(
                shared=self.plan.shared,
                job=job,
                messages=[Message("user", job.prompt)],
                model=self.spec.model,
                timeout=self.spec.timeout_per_agent,
            )
            output = parse_and_validate_output(response.content, self.spec.output_schema)
            return AgentResult(job_id=job.job_id, index=job.index, output=output)
        except Exception as exc:
            return AgentResult(
                job_id=job.job_id,
                index=job.index,
                output=None,
                error=AgentError(type=exc.__class__.__name__, message=str(exc)),
            )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value
