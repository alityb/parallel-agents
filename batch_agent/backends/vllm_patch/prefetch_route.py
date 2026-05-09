"""vLLM /internal/prefetch and /internal/pin_blocks route helpers.

Target vLLM version: 0.6.x.

These helpers are intentionally small and reuse vLLM's existing CacheEngine swap
infrastructure. They do not implement tensor transfer logic themselves.

Expected integration point in vLLM 0.6.x:
  - Register these routes in vllm/entrypoints/openai/api_server.py after the app
    and engine/client are initialized.
  - Provide `cache_engine` (object exposing `prefetch(block_ids, destination)`)
    and a `kv_registry` mapping kv_key -> list[int block_ids].

The SDK sends hints shaped like:
  {"job_id": "job-1", "kv_key": "abc", "priority": 1.0, "eta_seconds": 0.3}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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


async def handle_prefetch_request(
    payload: Mapping[str, Any],
    *,
    cache_engine: Any,
    kv_registry: Mapping[str, list[int]],
) -> dict[str, Any]:
    """Map kv_keys to block IDs and call cache_engine.prefetch()."""
    hints = _coerce_hints(payload)
    # Highest priority first, then soonest ETA.
    hints.sort(key=lambda h: (-h.priority, h.eta_seconds))

    prefetched: dict[str, list[int]] = {}
    missing: list[str] = []
    for hint in hints:
        block_ids = kv_registry.get(hint.kv_key)
        if not block_ids:
            missing.append(hint.kv_key)
            continue
        cache_engine.prefetch(block_ids, destination="gpu")
        prefetched[hint.kv_key] = list(block_ids)

    return {"ok": True, "prefetched": prefetched, "missing": missing}


async def handle_pin_blocks_request(
    payload: Mapping[str, Any],
    *,
    block_manager: Any,
    kv_registry: Mapping[str, list[int]],
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


def register_prefetch_routes(app: Any, *, cache_engine: Any, block_manager: Any, kv_registry: Mapping[str, list[int]]) -> None:
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
