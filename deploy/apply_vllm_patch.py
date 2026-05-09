# Targets vLLM 0.6.6. Verify against cache_engine.py before applying to other versions.
"""Apply BatchAgent's KVFlow prefetch route to a vLLM 0.6.6 install.

This patch is intentionally concrete:
  1. vllm/worker/cache_engine.py gets CacheEngine.prefetch(), implemented as
     a small wrapper around the existing CacheEngine.swap_in() CPU->GPU path.
  2. vllm/entrypoints/openai/api_server.py gets /internal/prefetch and
     /internal/pin_blocks routes wired to the in-process engine's CacheEngine.
     The route only calls swap_in() with explicit CPU->GPU block pairs; vLLM
     resident block-table IDs from get_block_table() are diagnostic only.

The route requires --disable-frontend-multiprocessing. With vLLM's frontend
multiprocessing client, the FastAPI process cannot safely reach worker-local
CacheEngine objects.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


MARKER = "BatchAgent KVFlow prefetch patch"
ROUTE_VERSION_MARKER = "BatchAgent KVFlow prefetch patch v3 safe swap mappings"


def _find_vllm_src() -> Path:
    env_src = os.getenv("VLLM_SRC")
    if env_src:
        return Path(env_src).expanduser().resolve()

    default_src = Path.home() / "vllm-src"
    if default_src.exists():
        return default_src

    try:
        import vllm  # type: ignore
    except ImportError:
        print("ERROR: vLLM source not found. Set VLLM_SRC or install vLLM first.")
        sys.exit(1)
    return Path(vllm.__file__).resolve().parents[1]


def _patch_cache_engine(cache_engine: Path) -> bool:
    if not cache_engine.exists():
        print(f"ERROR: {cache_engine} not found")
        sys.exit(1)

    text = cache_engine.read_text()
    prefetch_method = """    # BatchAgent KVFlow prefetch patch.
    def prefetch(self, block_ids) -> None:
        \"\"\"Move CPU KV blocks to GPU using the existing swap_in path.

        Requires explicit ``[[cpu_block_id, gpu_block_id], ...]`` mappings.
        vLLM's native swap_in API expects a CPU int64 tensor shaped [-1, 2].
        Resident GPU block-table IDs from BlockSpaceManager.get_block_table()
        are not valid swap_in source IDs.
        \"\"\"
        if isinstance(block_ids, torch.Tensor):
            src_to_dst = block_ids.to(device=\"cpu\", dtype=torch.int64)
        else:
            src_to_dst = torch.tensor(block_ids, device=\"cpu\", dtype=torch.int64)
        if src_to_dst.numel() == 0:
            return
        if src_to_dst.dim() != 2 or src_to_dst.size(-1) != 2:
            raise ValueError(\"CacheEngine.prefetch requires explicit CPU->GPU block pairs\")
        src_to_dst = src_to_dst.view(-1, 2)
        self.swap_in(src_to_dst)

"""
    if "def prefetch(self, block_ids)" in text:
        if "CacheEngine.prefetch requires explicit CPU->GPU block pairs" in text:
            print("cache_engine.py already has safe CacheEngine.prefetch — skipping")
            return False
        start = text.find("    # BatchAgent KVFlow prefetch patch.\n    def prefetch(self, block_ids)")
        end = text.find("\n    def copy(", start)
        if start == -1 or end == -1:
            print("ERROR: could not find existing CacheEngine.prefetch patch block")
            sys.exit(1)
        cache_engine.write_text(text[:start] + prefetch_method + text[end + 1:])
        print(f"Updated {cache_engine}")
        return True

    needle = """    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

"""
    if needle not in text:
        print("ERROR: could not find CacheEngine.swap_out insertion point")
        sys.exit(1)

    cache_engine.write_text(text.replace(needle, needle + prefetch_method))
    print(f"Patched {cache_engine}")
    return True


def _patch_api_server(api_server: Path) -> bool:
    if not api_server.exists():
        print(f"ERROR: {api_server} not found")
        sys.exit(1)

    text = api_server.read_text()
    if ROUTE_VERSION_MARKER in text:
        print("api_server.py already patched — skipping")
        return False

    route_code = f'''

# ─── {MARKER} ─────────────────────────────────────────────
# {ROUTE_VERSION_MARKER}
def _batchagent_cache_engines(engine_client):
    """Return worker-local CacheEngine objects for in-process vLLM."""
    engine = getattr(engine_client, "engine", None)
    model_executor = getattr(engine, "model_executor", None)
    driver_worker = getattr(model_executor, "driver_worker", None)
    cache_engines = getattr(driver_worker, "cache_engine", None)
    if cache_engines is None:
        return []
    if isinstance(cache_engines, (list, tuple)):
        return list(cache_engines)
    return [cache_engines]


def _batchagent_schedulers(engine_client):
    engine = getattr(engine_client, "engine", None)
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is None:
        return []
    if isinstance(scheduler, (list, tuple)):
        return list(scheduler)
    return [scheduler]


def _batchagent_seq_key_matches(seq, seq_group, kv_key):
    candidates = [
        getattr(seq, "seq_id", None),
        getattr(seq, "request_id", None),
        getattr(seq_group, "request_id", None),
    ]
    return any(candidate is not None and str(candidate) == str(kv_key)
               for candidate in candidates)


def _batchagent_block_table_for_key(engine_client, kv_key):
    for scheduler in _batchagent_schedulers(engine_client):
        block_manager = getattr(scheduler, "block_manager", None)
        if block_manager is None:
            continue

        block_tables = getattr(block_manager, "block_tables", {{}})
        if kv_key in block_tables:
            return list(block_tables[kv_key].physical_block_ids)
        try:
            kv_key_int = int(kv_key)
        except (TypeError, ValueError):
            kv_key_int = None
        if kv_key_int is not None and kv_key_int in block_tables:
            return list(block_tables[kv_key_int].physical_block_ids)

        for queue_name in ("running", "waiting", "swapped"):
            for seq_group in list(getattr(scheduler, queue_name, []) or []):
                try:
                    seqs = seq_group.get_seqs()
                except Exception:
                    seqs = []
                for seq in seqs:
                    if _batchagent_seq_key_matches(seq, seq_group, kv_key):
                        return list(block_manager.get_block_table(seq))
    return []


def _batchagent_block_pairs(raw_blocks):
    pairs = []
    for block in raw_blocks or []:
        if isinstance(block, (list, tuple)) and len(block) == 2:
            pairs.append([int(block[0]), int(block[1])])
        else:
            raise ValueError(
                "prefetch requires explicit [cpu_block_id, gpu_block_id] pairs; "
                "resident GPU block IDs from get_block_table() are not swap_in inputs")
    return pairs


def _batchagent_destination_block_ids(block_pairs):
    return [int(pair[1]) for pair in block_pairs]


@app.post("/internal/prefetch")
async def _batchagent_prefetch(request: Request):
    payload = await request.json()
    raw_blocks = payload.get("block_ids") or payload.get("blocks") or []
    kv_keys = []
    for hint in payload.get("hints", []):
        raw_blocks.extend(hint.get("block_ids") or hint.get("blocks") or [])
        if hint.get("kv_key"):
            kv_keys.append(hint.get("kv_key"))
    kv_keys.extend(payload.get("kv_keys") or [])

    engine_client = request.app.state.engine_client
    try:
        block_pairs = _batchagent_block_pairs(raw_blocks)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={{"ok": False, "error": str(exc)}})

    registry = getattr(engine_client, "_batchagent_kv_block_pairs", {{}})
    missing = []
    resident = {{}}
    for kv_key in kv_keys:
        if kv_key in registry:
            block_pairs.extend(_batchagent_block_pairs(registry[kv_key]))
        else:
            missing.append(kv_key)
            block_table = _batchagent_block_table_for_key(engine_client, kv_key)
            if block_table:
                # get_block_table() returns resident GPU physical IDs. They are
                # useful for diagnostics, but not valid CPU->GPU swap mappings.
                resident[kv_key] = block_table

    if not block_pairs:
        return {{"ok": True, "prefetched": {{}}, "missing": missing,
                "resident_block_tables": resident,
                "note": "kv_key-only hints cannot be safely converted to swap_in pairs; "
                        "vLLM swap_in requires CPU->GPU mappings from BlockSpaceManager.swap_in(seq_group)"}}

    cache_engines = _batchagent_cache_engines(engine_client)
    if not cache_engines:
        return JSONResponse(
            status_code=503,
            content={{
                "ok": False,
                "error": "CacheEngine unavailable. Start vLLM with --disable-frontend-multiprocessing.",
            }},
        )

    loop = asyncio.get_running_loop()
    for cache_engine in cache_engines:
        await loop.run_in_executor(None, cache_engine.prefetch, block_pairs)
    return {{"ok": True, "prefetched": {{"block_pairs": block_pairs}}, "missing": missing,
            "resident_block_tables": resident}}


@app.post("/internal/pin_blocks")
async def _batchagent_pin_blocks(request: Request):
    payload = await request.json()
    engine_client = request.app.state.engine_client
    registry = getattr(engine_client, "_batchagent_kv_block_pairs", None)
    if registry is None:
        registry = {{}}
        setattr(engine_client, "_batchagent_kv_block_pairs", registry)

    raw_blocks = payload.get("block_ids") or payload.get("blocks") or []
    try:
        block_pairs = _batchagent_block_pairs(raw_blocks)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={{"ok": False, "error": str(exc)}})

    kv_keys = payload.get("kv_keys", [])
    pinned = {{}}
    missing = []
    for kv_key in kv_keys:
        if block_pairs:
            registry[kv_key] = block_pairs
            pinned[kv_key] = _batchagent_destination_block_ids(block_pairs)
        else:
            missing.append(kv_key)

    # Prefix block pinning needs BlockManager eviction integration. Keep the
    # endpoint explicit so SDK startup can distinguish "patched" from 404.
    return {{"ok": True, "pinned": pinned, "missing": missing,
            "note": "pin_blocks requires BlockManager eviction integration"}}
# ─────────────────────────────────────────────────────────────────────────────

'''
    route_code = "\n".join(("    " + line) if line else line for line in route_code.splitlines()) + "\n"

    needle = "    mount_metrics(app)\n"
    if MARKER in text:
        start = text.find(f"# ─── {MARKER}")
        end = text.find("# ─────────────────────────────────────────────────────────────────────────────", start)
        if start == -1 or end == -1:
            print("ERROR: could not find existing BatchAgent patch block")
            sys.exit(1)
        end = text.find("\n", end)
        if end == -1:
            end = len(text)
        api_server.write_text(text[:start] + route_code + text[end + 1:])
        print(f"Updated {api_server}")
        return True

    if needle not in text:
        print("ERROR: could not find build_app mount_metrics insertion point")
        sys.exit(1)
    api_server.write_text(text.replace(needle, needle + route_code, 1))
    print(f"Patched {api_server}")
    return True


def main() -> None:
    vllm_src = _find_vllm_src()
    cache_engine = vllm_src / "vllm/worker/cache_engine.py"
    api_server = vllm_src / "vllm/entrypoints/openai/api_server.py"

    patched_cache = _patch_cache_engine(cache_engine)
    patched_api = _patch_api_server(api_server)

    api_text = api_server.read_text()
    cache_text = cache_engine.read_text()
    if ROUTE_VERSION_MARKER not in api_text or "def prefetch(self, block_ids)" not in cache_text:
        print("ERROR: vLLM patch verification failed")
        sys.exit(1)
    print("BatchAgent vLLM patch verification OK")
    if not patched_cache and not patched_api:
        print("No changes needed")


if __name__ == "__main__":
    main()
