#!/usr/bin/env bash
# next_gpu_session.sh — run this on a fresh g6.xlarge (L4 24GB, Ubuntu 22.04)
# Covers: vLLM patch, KVFlow benchmark, SGLang, distributed with Redis
# Budget: ~$3.22 for 4 hours
set -euo pipefail

REPO=/home/ubuntu/parallel-agents
MODEL=Qwen/Qwen2.5-7B-Instruct
RESULTS=/tmp/session_results
mkdir -p $RESULTS

cd $REPO

echo "=== Setup ==="
sudo apt-get update -y -q
sudo apt-get install -y -q git tmux redis-server python3-pip python3-venv build-essential ninja-build

# Create venv if not present
[ -d ~/vllm-env ] || python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# HOUR 1: vLLM from source + prefetch patch
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "HOUR 1: Installing vLLM from source + applying prefetch patch"
echo "=========================================================="

# Install our SDK first
pip install -e $REPO --quiet

# Clone vLLM 0.6.6 if not present
if [ ! -d ~/vllm-src ]; then
    git clone --depth 1 --branch v0.6.6 https://github.com/vllm-project/vllm ~/vllm-src
fi
cd ~/vllm-src

# Apply the prefetch + pin_blocks patch to the API server
python3 $REPO/deploy/apply_vllm_patch.py

# Build and install
pip install -e . --no-build-isolation --quiet

# Start patched vLLM
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "
  python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --host 0.0.0.0 --port 8000 \
    --enable-prefix-caching --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --dtype bfloat16 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    2>&1 | tee /tmp/vllm_patched.log
"

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && echo "vLLM ready after ${i}x5s" && break
    sleep 5
done

# Verify prefetch endpoint returns 200
echo "Verifying /internal/prefetch..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/internal/prefetch \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{"hints": [{"job_id": "test", "kv_key": "abc", "priority": 1.0, "eta_seconds": 0.5}]}')
echo "  /internal/prefetch status: $RESPONSE"
[ "$RESPONSE" = "200" ] && echo "  PATCH APPLIED SUCCESSFULLY" || echo "  PATCH VERIFICATION FAILED"

echo "$RESPONSE" > $RESULTS/prefetch_endpoint_status.txt

# ─────────────────────────────────────────────────────────────────────────────
# HOUR 2: KVFlow TTFT-after-TOOL_WAIT benchmark
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "HOUR 2: KVFlow TTFT benchmark (with vs without prefetch)"
echo "=========================================================="

cd $REPO
PYTHONPATH=$REPO python3 /tmp/kvflow_benchmark.py

# ─────────────────────────────────────────────────────────────────────────────
# HOUR 3: SGLang install + test
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "HOUR 3: SGLang install + test_sglang_backend.py"
echo "=========================================================="

# Kill vLLM to free VRAM for SGLang
tmux kill-session -t vllm 2>/dev/null || true
sleep 5

pip install "sglang[srt]>=0.3" flashinfer-python --quiet

# Start SGLang
tmux new-session -d -s sglang "
    python -m sglang.launch_server \
        --model $MODEL --host 0.0.0.0 --port 30000 \
        --tp 1 --dtype bfloat16 \
        2>&1 | tee /tmp/sglang.log
"

echo "Waiting for SGLang..."
for i in $(seq 1 120); do
    curl -sf http://localhost:30000/health >/dev/null 2>&1 && echo "SGLang ready after ${i}x5s" && break
    sleep 5
done

# Run integration tests
cd $REPO
PYTHONPATH=$REPO python3 /tmp/sglang_benchmark.py

# ─────────────────────────────────────────────────────────────────────────────
# HOUR 4: Distributed with real Redis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "HOUR 4: Distributed mode with real Redis"
echo "=========================================================="

# Start Redis
redis-server --daemonize yes --port 6379
sleep 1
redis-cli ping | grep -q PONG && echo "Redis ready" || echo "Redis failed"

cd $REPO
PYTHONPATH=$REPO python3 -m pytest \
    tests/integration/test_distributed.py \
    tests/integration/test_distributed_scheduler.py \
    -v --tb=short 2>&1 | tee $RESULTS/distributed_pytest.txt

# Real Redis chaos test
PYTHONPATH=$REPO python3 /tmp/real_redis_chaos.py

echo ""
echo "=========================================================="
echo "SESSION COMPLETE — results in $RESULTS/"
echo "=========================================================="
ls -la $RESULTS/
