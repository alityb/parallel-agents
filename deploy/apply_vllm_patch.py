# Targets vLLM 0.6.6. Verify against cache_engine.py before applying to other versions.
"""Apply BatchAgent's KVFlow prefetch route to a vLLM 0.6.6 install.

This patch is intentionally concrete:
  1. vllm/worker/cache_engine.py gets CacheEngine.prefetch(), implemented as
     a small wrapper around the existing CacheEngine.swap_in() CPU->GPU path.
  2. vllm/entrypoints/openai/api_server.py gets /internal/prefetch and
     /internal/pin_blocks routes wired to the in-process engine's CacheEngine.

The route requires --disable-frontend-multiprocessing. With vLLM's frontend
multiprocessing client, the FastAPI process cannot safely reach worker-local
CacheEngine objects.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


MARKER = "BatchAgent KVFlow prefetch patch"


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
    if "def prefetch(self, block_ids)" in text:
        print("cache_engine.py already has CacheEngine.prefetch — skipping")
        return False

    needle = """    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

"""
    if needle not in text:
        print("ERROR: could not find CacheEngine.swap_out insertion point")
        sys.exit(1)

    prefetch_method = """    # BatchAgent KVFlow prefetch patch.
    def prefetch(self, block_ids) -> None:
        \"\"\"Move CPU KV blocks to GPU using the existing swap_in path.

        Accepts either ``[block_id, ...]`` or explicit
        ``[[cpu_block_id, gpu_block_id], ...]`` mappings. vLLM's native
        swap_in API expects the latter as a CPU int64 tensor shaped [-1, 2].
        \"\"\"
        if isinstance(block_ids, torch.Tensor):
            src_to_dst = block_ids.to(device=\"cpu\", dtype=torch.int64)
        else:
            src_to_dst = torch.tensor(block_ids, device=\"cpu\", dtype=torch.int64)
        if src_to_dst.numel() == 0:
            return
        if src_to_dst.dim() == 1:
            src_to_dst = torch.stack((src_to_dst, src_to_dst), dim=1)
        src_to_dst = src_to_dst.view(-1, 2)
        self.swap_in(src_to_dst)

"""
    cache_engine.write_text(text.replace(needle, needle + prefetch_method))
    print(f"Patched {cache_engine}")
    return True


def _patch_api_server(api_server: Path) -> bool:
    if not api_server.exists():
        print(f"ERROR: {api_server} not found")
        sys.exit(1)

    text = api_server.read_text()
    if MARKER in text:
        print("api_server.py already patched — skipping")
        return False

    route_code = f'''

# ─── {MARKER} ─────────────────────────────────────────────
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


def _batchagent_block_pairs(raw_blocks):
    pairs = []
    for block in raw_blocks or []:
        if isinstance(block, (list, tuple)) and len(block) == 2:
            pairs.append([int(block[0]), int(block[1])])
        else:
            bid = int(block)
            pairs.append([bid, bid])
    return pairs


@app.post("/internal/prefetch")
async def _batchagent_prefetch(request: Request):
    payload = await request.json()
    raw_blocks = payload.get("block_ids") or payload.get("blocks") or []
    for hint in payload.get("hints", []):
        raw_blocks.extend(hint.get("block_ids") or hint.get("blocks") or [])
    block_pairs = _batchagent_block_pairs(raw_blocks)
    if not block_pairs:
        return {{"ok": True, "prefetched": {{}}, "missing": [
            h.get("kv_key") for h in payload.get("hints", []) if h.get("kv_key")
        ], "note": "no block_ids supplied"}}

    cache_engines = _batchagent_cache_engines(request.app.state.engine_client)
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
    return {{"ok": True, "prefetched": {{"block_pairs": block_pairs}}, "missing": []}}


@app.post("/internal/pin_blocks")
async def _batchagent_pin_blocks(request: Request):
    payload = await request.json()
    # Prefix block pinning needs BlockManager eviction integration. Keep the
    # endpoint explicit so SDK startup can distinguish "patched" from 404.
    return {{"ok": True, "pinned": {{}}, "missing": payload.get("kv_keys", []),
            "note": "pin_blocks requires BlockManager eviction integration"}}
# ─────────────────────────────────────────────────────────────────────────────

'''
    route_code = "\n".join(("    " + line) if line else line for line in route_code.splitlines()) + "\n"

    needle = "    mount_metrics(app)\n"
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
    if MARKER not in api_text or "def prefetch(self, block_ids)" not in cache_text:
        print("ERROR: vLLM patch verification failed")
        sys.exit(1)
    print("BatchAgent vLLM patch verification OK")
    if not patched_cache and not patched_api:
        print("No changes needed")


if __name__ == "__main__":
    main()
