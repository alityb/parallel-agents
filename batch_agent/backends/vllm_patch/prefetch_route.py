"""vLLM /internal/prefetch and /internal/pin_blocks route helpers.

Target vLLM version: 0.6.x.

These helpers are intentionally small and reuse vLLM's existing CacheEngine swap
infrastructure. They do not implement tensor transfer logic themselves, and they
do not treat resident PagedAttention block-table IDs as swap-in mappings.

Expected integration point in vLLM 0.6.x:
  - Register these routes in vllm/entrypoints/openai/api_server.py after the app
    and engine/client are initialized.
  - Provide `cache_engine` (object exposing `prefetch(block_pairs)` or
    `swap_in(src_to_dst)`)
    and a `kv_registry` mapping kv_key -> explicit CPU->GPU block pairs.

The SDK sends hints shaped like:
  {"job_id": "job-1", "kv_key": "abc", "priority": 1.0, "eta_seconds": 0.3}
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class InternalPrefetchHint:
    job_id: str
    kv_key: str
    priority: float = 1.0
    eta_seconds: float = 0.0


def _coerce_hints(payload: Mapping[str, Any]) -> list[InternalPrefetchHint]:
    raw_hints = payload.get("hints", [])
    hints: list[InternalPrefetchHint] = []
    for raw in raw_hints:
        hints.append(InternalPrefetchHint(
            job_id=str(raw.get("job_id", "")),
            kv_key=str(raw["kv_key"]),
            priority=float(raw.get("priority", 1.0)),
            eta_seconds=float(raw.get("eta_seconds", 0.0)),
        ))
    return hints


def _coerce_block_pairs(blocks: Sequence[Any]) -> list[list[int]]:
    """Return vLLM swap_in pairs from explicit CPU->GPU mappings.

    vLLM 0.6.6 CacheEngine.swap_in expects a CPU tensor shaped ``[-1, 2]``:
    ``[cpu_block_id, gpu_block_id]``. Resident GPU block-table IDs returned by
    BlockSpaceManager.get_block_table() are not valid swap_in source IDs.
    """
    pairs: list[list[int]] = []
    for block in blocks:
        if isinstance(block, (list, tuple)) and len(block) == 2:
            pairs.append([int(block[0]), int(block[1])])
        else:
            raise ValueError(
                "prefetch requires explicit [cpu_block_id, gpu_block_id] pairs; "
                "resident GPU block IDs are not valid swap_in input"
            )
    return pairs


async def _prefetch_block_pairs(cache_engine: Any, block_pairs: list[list[int]]) -> None:
    """Execute CacheEngine prefetch/swap_in without blocking the event loop."""
    if not block_pairs:
        return

    def _run() -> None:
        if hasattr(cache_engine, "prefetch"):
            cache_engine.prefetch(block_pairs)
            return
        if hasattr(cache_engine, "swap_in"):
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - host vLLM has torch
                raise RuntimeError("torch is required to call CacheEngine.swap_in") from exc
            cache_engine.swap_in(torch.tensor(block_pairs, device="cpu", dtype=torch.int64))
            return
        raise AttributeError("cache_engine must expose prefetch() or swap_in()")

    await asyncio.to_thread(_run)


async def handle_prefetch_request(
    payload: Mapping[str, Any],
    *,
    cache_engine: Any,
    kv_registry: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Map request blocks/kv_keys to block pairs and call CacheEngine swap-in."""
    direct_blocks = payload.get("block_ids") or payload.get("blocks")
    if direct_blocks:
        try:
            block_pairs = _coerce_block_pairs(direct_blocks)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "prefetched": {}, "missing": []}
        await _prefetch_block_pairs(cache_engine, block_pairs)
        return {"ok": True, "prefetched": {"__direct__": block_pairs}, "missing": []}

    hints = _coerce_hints(payload)
    # Highest priority first, then soonest ETA.
    hints.sort(key=lambda h: (-h.priority, h.eta_seconds))

    prefetched: dict[str, list[list[int]]] = {}
    missing: list[str] = []
    for hint in hints:
        raw_blocks = kv_registry.get(hint.kv_key)
        hint_block_ids = None
        for raw_hint in payload.get("hints", []):
            if str(raw_hint.get("kv_key", "")) == hint.kv_key:
                hint_block_ids = raw_hint.get("block_ids") or raw_hint.get("blocks")
                break
        block_ids = hint_block_ids or raw_blocks
        if not block_ids:
            missing.append(hint.kv_key)
            continue
        try:
            block_pairs = _coerce_block_pairs(block_ids)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "prefetched": prefetched, "missing": missing}
        await _prefetch_block_pairs(cache_engine, block_pairs)
        prefetched[hint.kv_key] = block_pairs

    return {"ok": True, "prefetched": prefetched, "missing": missing}


async def handle_pin_blocks_request(
    payload: Mapping[str, Any],
    *,
    block_manager: Any,
    kv_registry: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Pin blocks so the shared prefix survives LRU eviction."""
    kv_keys = payload.get("kv_keys", [])
    pinned: dict[str, list[int]] = {}
    missing: list[str] = []
    for kv_key in kv_keys:
        block_ids = kv_registry.get(kv_key)
        if not block_ids:
            missing.append(kv_key)
            continue
        try:
            block_ids = [pair[1] for pair in _coerce_block_pairs(block_ids)]
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "pinned": pinned, "missing": missing}
        if hasattr(block_manager, "pin_blocks"):
            block_manager.pin_blocks(block_ids)
        else:
            # Minimal expected patch: a pinned_block_ids set checked by eviction.
            pinned_set = getattr(block_manager, "pinned_block_ids", None)
            if pinned_set is None:
                pinned_set = set()
                setattr(block_manager, "pinned_block_ids", pinned_set)
            pinned_set.update(block_ids)
        pinned[kv_key] = list(block_ids)
    return {"ok": True, "pinned": pinned, "missing": missing}


def register_prefetch_routes(
    app: Any,
    *,
    cache_engine: Any,
    block_manager: Any,
    kv_registry: Mapping[str, Sequence[Any]],
) -> None:
    """Register FastAPI routes on a vLLM API server app.

    FastAPI is imported only by the host process that calls this function.
    """

    @app.post("/internal/prefetch")
    async def _prefetch(payload: dict[str, Any]) -> dict[str, Any]:
        return await handle_prefetch_request(
            payload,
            cache_engine=cache_engine,
            kv_registry=kv_registry,
        )

    @app.post("/internal/pin_blocks")
    async def _pin(payload: dict[str, Any]) -> dict[str, Any]:
        return await handle_pin_blocks_request(
            payload,
            block_manager=block_manager,
            kv_registry=kv_registry,
        )
