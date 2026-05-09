#!/usr/bin/env bash
set -euo pipefail

# One-command vLLM server bootstrap for Batch Agent native mode.
# Target: one A100 80GB node (RunPod or equivalent).
#
# Environment variables:
#   MODEL_ID                 Hugging Face model id (default: meta-llama/Llama-3.1-70B-Instruct)
#   PORT                     Server port (default: 8000)
#   GPU_MEMORY_UTILIZATION   vLLM GPU memory fraction (default: 0.85)
#   BATCH_AGENT_REPO         Path to this repo inside the container (default: /workspace/parallel-agents)
#   VLLM_VERSION             vLLM version (default: 0.6.6)

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
BATCH_AGENT_REPO="${BATCH_AGENT_REPO:-/workspace/parallel-agents}"
VLLM_VERSION="${VLLM_VERSION:-0.6.6}"

python3 -m pip install --upgrade pip
python3 -m pip install "vllm==${VLLM_VERSION}" fastapi uvicorn

# Install this SDK in editable mode if the repo is mounted. This makes
# batch_agent.backends.vllm_patch importable from the vLLM process.
if [[ -d "${BATCH_AGENT_REPO}" ]]; then
  python3 -m pip install -e "${BATCH_AGENT_REPO}"
  export PYTHONPATH="${BATCH_AGENT_REPO}:${PYTHONPATH:-}"
fi

# Patch availability check. The actual /internal/prefetch and /internal/pin_blocks
# route registration is provided by batch_agent.backends.vllm_patch.prefetch_route.
# vLLM 0.6.x integrators should import register_prefetch_routes(app, ...) from
# vllm/entrypoints/openai/api_server.py once cache_engine/block_manager are available.
python3 - <<'PY'
from batch_agent.backends.vllm_patch.prefetch_route import register_prefetch_routes
from batch_agent.backends.vllm_patch.diff_cache_engine import DiffCacheEngine
print("Batch Agent vLLM patch helpers import OK")
PY

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
