from __future__ import annotations

import time

from batch_agent.state import AgentState, AgentStatus, RedisStreamsStateStore


class MockRedis:
    def __init__(self) -> None:
        self.values = {}
        self.expiry = {}
        self.stream = []

    def _cleanup(self, key: str) -> None:
        expires = self.expiry.get(key)
        if expires is not None and time.time() >= expires:
            self.values.pop(key, None)
            self.expiry.pop(key, None)

    def get(self, key: str):
        self._cleanup(key)
        return self.values.get(key)

    def set(self, key: str, value, nx: bool = False, ex: int | None = None, px: int | None = None):
        self._cleanup(key)
        if nx and key in self.values:
            return False
        self.values[key] = value
        if ex is not None:
            self.expiry[key] = time.time() + ex
        if px is not None:
            self.expiry[key] = time.time() + (px / 1000)
        return True

    def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.expiry.pop(key, None)

    def xadd(self, stream: str, fields: dict) -> None:
        self.stream.append((stream, fields))


def test_two_nodes_lease_and_optimistic_locking() -> None:
    redis = MockRedis()
    node_a = RedisStreamsStateStore(redis, node_id="node-a")
    node_b = RedisStreamsStateStore(redis, node_id="node-b")

    assert node_a.acquire_lease("job-1", ttl_seconds=1.0) is True
    assert node_b.acquire_lease("job-1", ttl_seconds=1.0) is False

    initial = AgentState(job_id="job-1", status=AgentStatus.RUNNING, turn=1, version=0)
    assert node_a.save_with_version(initial, expected_version=0) is True

    stale = AgentState(job_id="job-1", status=AgentStatus.RUNNING, turn=2, version=0)
    assert node_b.save_with_version(stale, expected_version=0) is False

    latest = node_b.load("job-1")
    assert latest is not None
    assert latest.version == 1
    assert latest.owner_node_id == "node-a"

    # After lease expiry, another node can pick up the job.
    time.sleep(1.05)
    assert node_b.acquire_lease("job-1", ttl_seconds=1.0) is True
