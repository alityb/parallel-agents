"""
Apply the BatchAgent prefetch patch to a vLLM 0.6.x source tree.
Run this from inside ~/vllm-src after cloning vllm v0.6.6.

What it does:
  1. Reads our backends/vllm_patch/prefetch_route.py
  2. Injects a call to register_prefetch_routes(app, ...) into
     vllm/entrypoints/openai/api_server.py
  3. Adds a minimal KVRegistry stub so /internal/prefetch returns 200
     (actual block tracking requires deeper integration — stub is enough
     to prove the endpoint works and hints are received by vLLM)
"""
from __future__ import annotations
import sys
from pathlib import Path

VLLM_SRC = Path(__file__).parent.parent  # ~/vllm-src when called as deploy/apply_vllm_patch.py

# Actually we need the vllm-src tree, which is ~/vllm-src when invoked via next_gpu_session.sh
import os
vllm_src = Path(os.getenv("VLLM_SRC", Path.home() / "vllm-src"))
api_server = vllm_src / "vllm/entrypoints/openai/api_server.py"

if not api_server.exists():
    print(f"ERROR: {api_server} not found. Run from vllm-src root.")
    sys.exit(1)

original = api_server.read_text()

# Already patched?
if "internal/prefetch" in original:
    print("vLLM api_server.py already patched — skipping")
    sys.exit(0)

# Find insertion point — after "app = FastAPI(" or equivalent
# In vLLM 0.6.x this is near the top of the file
insertion_after = "app = FastAPI("
if insertion_after not in original:
    # fallback for different vLLM versions
    insertion_after = "@asynccontextmanager"

patch_code = '''

# ─── BatchAgent prefetch patch ───────────────────────────────────────────────
# Provides /internal/prefetch and /internal/pin_blocks endpoints.
# Source: batch_agent/backends/vllm_patch/prefetch_route.py
try:
    import sys as _sys
    # Allow importing batch_agent if it's installed
    from batch_agent.backends.vllm_patch.prefetch_route import register_prefetch_routes as _reg_prefetch
    # Stub registry and cache engine for Phase 3A demo
    # (real implementation requires hooking into vLLM worker CacheEngine)
    _kv_registry = {}  # kv_key → list[int block_ids]  — populated by warm_prefix calls

    class _StubCacheEngine:
        """Stub that accepts prefetch calls without doing actual GPU transfer.
        Replace with real CacheEngine reference for production KVFlow."""
        def prefetch(self, block_ids, destination="gpu"):
            pass  # no-op until real CacheEngine integration

    class _StubBlockManager:
        def pin_blocks(self, block_ids):
            pass
        pinned_block_ids = set()

    _reg_prefetch(app,
                  cache_engine=_StubCacheEngine(),
                  block_manager=_StubBlockManager(),
                  kv_registry=_kv_registry)
    print("[BatchAgent] /internal/prefetch and /internal/pin_blocks registered")
except ImportError as _e:
    print(f"[BatchAgent] Prefetch patch skipped (batch_agent not installed): {_e}")
# ─────────────────────────────────────────────────────────────────────────────

'''

# Find where app is first defined and insert after its closing line
lines = original.split('\n')
insert_at = None
for i, line in enumerate(lines):
    if 'FastAPI(' in line or 'app = FastAPI' in line:
        # Find the closing ) of the FastAPI call (may be multi-line)
        depth = 0
        for j in range(i, min(i+20, len(lines))):
            depth += lines[j].count('(') - lines[j].count(')')
            if depth <= 0:
                insert_at = j + 1
                break
        break

if insert_at is None:
    # Simpler: just append before the main uvicorn.run call
    for i, line in enumerate(lines):
        if 'uvicorn.run' in line or 'serve(' in line:
            insert_at = i
            break

if insert_at is None:
    print("ERROR: Could not find insertion point in api_server.py")
    sys.exit(1)

lines.insert(insert_at, patch_code)
patched = '\n'.join(lines)
api_server.write_text(patched)
print(f"Patched {api_server}")
print(f"Insertion at line {insert_at}")
