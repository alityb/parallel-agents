#!/usr/bin/env bash
set -euo pipefail

# One-command vLLM server bootstrap for Batch Agent native mode.
# Target: one A100 80GB node (RunPod or equivalent).
#
# Environment variables:
#   MODEL_ID                 Hugging Face model id (default: meta-llama/Llama-3.1-70B-Instruct)
#   PORT                     Server port (default: 8000)
#   GPU_MEMORY_UTILIZATION   vLLM GPU memory fraction (default: 0.85)
#   BATCH_AGENT_REPO         Path to this repo inside the container (default: /workspace/batchagent)
#   VLLM_VERSION             vLLM version (default: 0.6.6)

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
BATCH_AGENT_REPO="${BATCH_AGENT_REPO:-/workspace/batchagent}"
VLLM_VERSION="${VLLM_VERSION:-0.6.6}"

python3 -m pip install --upgrade pip
python3 -m pip install "vllm==${VLLM_VERSION}" fastapi uvicorn

# Install this SDK in editable mode if the repo is mounted. This makes
# batch_agent.backends.vllm_patch importable from the vLLM process.
if [[ -d "${BATCH_AGENT_REPO}" ]]; then
  python3 -m pip install -e "${BATCH_AGENT_REPO}"
  export PYTHONPATH="${BATCH_AGENT_REPO}:${PYTHONPATH:-}"
fi

# Apply and verify the vLLM source/site-package patch before serving.
if [[ ! -d "${BATCH_AGENT_REPO}" ]]; then
  echo "ERROR: BATCH_AGENT_REPO=${BATCH_AGENT_REPO} does not exist; cannot apply vLLM patch" >&2
  exit 1
fi
python3 "${BATCH_AGENT_REPO}/deploy/apply_vllm_patch.py"
python3 - <<'PY'
from pathlib import Path
import vllm

root = Path(vllm.__file__).resolve().parents[1]
api = root / "vllm/entrypoints/openai/api_server.py"
cache = root / "vllm/worker/cache_engine.py"
assert "BatchAgent KVFlow prefetch patch" in api.read_text()
assert "def prefetch(self, block_ids)" in cache.read_text()
print("Batch Agent vLLM patch verification OK")
PY

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --disable-frontend-multiprocessing
