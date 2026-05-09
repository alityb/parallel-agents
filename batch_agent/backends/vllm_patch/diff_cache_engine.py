"""TokenDance-style diff-aware KV storage prototype.

This module is intentionally isolated and flag-gated. It does not affect normal
execution unless `diff_kv=True` code paths instantiate it.

Target vLLM version: 0.6.x. In a live vLLM patch this class should subclass
vLLM's CacheEngine. In this repository, vLLM is optional, so we fall back to
`object` when vLLM is not importable; the hashing/dedup/diff algorithms remain
unit-testable.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Iterable

try:  # pragma: no cover - vLLM is not installed in CI
    from vllm.worker.cache_engine import CacheEngine as _VLLMCacheEngine
except Exception:  # pragma: no cover
    _VLLMCacheEngine = object


@dataclass(frozen=True)
class AgentKVSnapshot:
    job_id: str
    tokens: tuple[int, ...]
    turn: int


@dataclass(frozen=True)
class EncodedAgentDiff:
    job_id: str
    block_hashes: tuple[str, ...]
    unique_block_hashes: tuple[str, ...]
    shared_block_hashes: tuple[str, ...]


@dataclass(frozen=True)
class CompressionStats:
    full_blocks: int
    stored_unique_blocks: int
    compression_ratio: float
    agents_encoded: int


class BlockHasher:
    def __init__(self, block_size_tokens: int = 16) -> None:
        self.block_size_tokens = block_size_tokens

    def split_blocks(self, tokens: Iterable[int]) -> list[tuple[int, ...]]:
        values = list(tokens)
        return [tuple(values[i:i + self.block_size_tokens]) for i in range(0, len(values), self.block_size_tokens)]

    def hash_block(self, block: tuple[int, ...]) -> str:
        payload = ",".join(str(t) for t in block).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def hash_tokens(self, tokens: Iterable[int]) -> list[str]:
        return [self.hash_block(block) for block in self.split_blocks(tokens)]


class DiffCacheEngine(_VLLMCacheEngine):
    """Block-hashing diff cache encoder.

    v0 behavior: records block hashes, deduplicates shared blocks globally, and
    stores per-agent sparse diffs as references to already-known blocks plus the
    unique blocks for that agent.
    """

    def __init__(self, *args: Any, block_size_tokens: int = 16, **kwargs: Any) -> None:
        if _VLLMCacheEngine is not object:
            super().__init__(*args, **kwargs)  # type: ignore[misc]
        self.hasher = BlockHasher(block_size_tokens=block_size_tokens)
        self.global_block_store: dict[str, tuple[int, ...]] = {}
        self.reference_counts: dict[str, int] = {}
        self.agent_diffs: dict[str, EncodedAgentDiff] = {}

    def encode_snapshot(self, snapshot: AgentKVSnapshot) -> EncodedAgentDiff:
        blocks = self.hasher.split_blocks(snapshot.tokens)
        block_hashes = [self.hasher.hash_block(block) for block in blocks]
        unique: list[str] = []
        shared: list[str] = []

        for block_hash, block in zip(block_hashes, blocks):
            if block_hash in self.global_block_store:
                shared.append(block_hash)
                self.reference_counts[block_hash] += 1
            else:
                unique.append(block_hash)
                self.global_block_store[block_hash] = block
                self.reference_counts[block_hash] = 1

        diff = EncodedAgentDiff(
            job_id=snapshot.job_id,
            block_hashes=tuple(block_hashes),
            unique_block_hashes=tuple(unique),
            shared_block_hashes=tuple(shared),
        )
        self.agent_diffs[snapshot.job_id] = diff
        return diff

    async def all_gather(
        self,
        snapshots: Iterable[AgentKVSnapshot],
        *,
        soft_timeout_seconds: float = 0.5,
        completion_fraction: float = 0.80,
    ) -> list[EncodedAgentDiff]:
        """Soft-timeout All-Gather without a hard synchronization barrier.

        Stops when either >80% snapshots are encoded or 500ms elapses.
        Correctness is unaffected by partial gathers; missed snapshots can be
        encoded on the next gather pass.
        """
        snapshot_list = list(snapshots)
        if not snapshot_list:
            return []
        target = max(1, int(len(snapshot_list) * completion_fraction))
        started = time.monotonic()
        encoded: list[EncodedAgentDiff] = []
        for snapshot in snapshot_list:
            encoded.append(self.encode_snapshot(snapshot))
            if len(encoded) >= target:
                break
            if time.monotonic() - started >= soft_timeout_seconds:
                break
            await asyncio.sleep(0)
        return encoded

    def stats(self, snapshots: Iterable[AgentKVSnapshot]) -> CompressionStats:
        snapshot_list = list(snapshots)
        full_blocks = sum(len(self.hasher.split_blocks(s.tokens)) for s in snapshot_list)
        stored = len(self.global_block_store)
        ratio = (full_blocks / stored) if stored else 1.0
        return CompressionStats(
            full_blocks=full_blocks,
            stored_unique_blocks=stored,
            compression_ratio=ratio,
            agents_encoded=len(self.agent_diffs),
        )


def maybe_create_diff_cache_engine(diff_kv: bool, **kwargs: Any) -> DiffCacheEngine | None:
    """Feature gate. diff_kv=False must have zero runtime effect."""
    if not diff_kv:
        return None
    return DiffCacheEngine(**kwargs)
