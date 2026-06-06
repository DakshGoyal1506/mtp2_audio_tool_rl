#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --exclude=cn15-dgx
#SBATCH --job-name=bench-judge
#SBATCH --output=logs/bench_judge_slurm_%j.log

# =============================================================================
# Benchmark vLLM judge throughput on a single DGX GPU
#
# Usage:
#   sbatch grpo_single_phase_llm/benchmark/bench_judge.sh
#   sbatch grpo_single_phase_llm/benchmark/bench_judge.sh --num-requests 100
# =============================================================================

eval "$(/home/speech-nlp-cse/24m0756/anaconda3/bin/conda shell.bash hook)"
conda activate vllm-env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR=/home/speech-nlp-cse/24m0756/.cache/tiktoken-rs-cache

cd "${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(dirname $(realpath $0))))}"
echo "Working dir: $PWD"

mkdir -p logs

NODE_IP=$(hostname -i | awk '{print $1}')
NODE_NAME=$(hostname -s)

echo "=============================================="
echo "  SGLang Judge Server"
echo "=============================================="
echo "  Model: openai/gpt-oss-20b"
echo "  GPU:   1x (TP=1)"
echo "  Job:   ${SLURM_JOB_ID:-local}"
echo "  Node:  ${NODE_NAME}"
echo "  IP:    ${NODE_IP}"
echo "  Port:  8000"
echo "=============================================="
echo ""
echo ">>> To use this judge in training, set in config:"
echo ">>>   judge_host: \"${NODE_IP}\""
echo ">>>   judge_port: 8000"
echo ""

# --- Auto-update training config with this judge's IP ---
CONFIG_FILE="grpo_single_phase_llm/configs/optimized_v4.yaml"
if [ -f "$CONFIG_FILE" ]; then
    sed -i "s/^judge_host:.*/judge_host: \"${NODE_IP}\"/" "$CONFIG_FILE"
    echo ">>> Updated $CONFIG_FILE with judge_host: \"${NODE_IP}\""
else
    echo ">>> WARNING: $CONFIG_FILE not found, skipping auto-update"
fi
echo ""

# --- Auto-restart loop + keepalive ---
# SGLang's detokenizer can die during long idle periods with Mamba models.
# This loop restarts the server automatically and a background keepalive
# pings it every 60s to prevent idle-related crashes.

SGLANG_CMD="python -m sglang.launch_server \
    --model-path openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 \
    --tp-size 2 \
    --mem-fraction-static 0.8 \
    --context-length 16384 \
    --dtype bfloat16 \
    --trust-remote-code \
    --cuda-graph-max-bs ${CUDA_GRAPH_MAX_BS:-256} \
    --watchdog-timeout 300"

# SGLANG_CMD="python -m sglang.launch_server \
#     --model-path Qwen/Qwen3.5-27B \
#     --host 0.0.0.0 --port 8000 \
#     --tp-size 2 \
#     --mem-fraction-static 0.8 \
#     --context-length 16384 \
#     --dtype bfloat16 \
#     --trust-remote-code \
#     --cuda-graph-max-bs ${CUDA_GRAPH_MAX_BS:-64} \
#     --reasoning-parser qwen3 \
#     --watchdog-timeout 300"

# SGLANG_CMD="python -m sglang.launch_server \
# --model-path Qwen/Qwen3.5-27B \
# --host 0.0.0.0 \
# --port 8000 \
# --tp-size 1 \
# --mem-fraction-static 0.8 \
# --context-length 16384 \
# --reasoning-parser qwen3 \
# --dtype bfloat16"



$SGLANG_CMD 2>&1
    # --disable-cuda-graph \
# MAX_RESTARTS=5
# for attempt in $(seq 1 $MAX_RESTARTS); do
#     echo ""
#     echo "[$(date)] Starting SGLang server (attempt $attempt/$MAX_RESTARTS)..."

#     # Launch server
#     $SGLANG_CMD 2>&1 &
#     SERVER_PID=$!

#     # Wait for server to become ready
#     for i in $(seq 1 120); do
#         if curl -s --max-time 3 http://localhost:8000/health | grep -q ""; then
#             STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://localhost:8000/health)
#             if [ "$STATUS" = "200" ]; then
#                 echo "[$(date)] Server ready (attempt $attempt)."
#                 break 2  # break out of both loops — server is ready
#             fi
#         fi
#         if ! kill -0 $SERVER_PID 2>/dev/null; then
#             echo "[$(date)] Server process died during startup."
#             break  # break inner loop, retry outer
#         fi
#         sleep 5
#     done

#     # If we're here from the wait loop (not break 2), check if server started
#     if kill -0 $SERVER_PID 2>/dev/null; then
#         echo "[$(date)] Server ready (attempt $attempt)."
#         break
#     fi

#     echo "[$(date)] Server failed to start. Retrying..."
#     sleep 5
# done

# # Start keepalive in background — ping every 60s to prevent idle death
# (
#     while true; do
#         sleep 60
#         if ! curl -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
#             echo "[$(date)] Keepalive: health check FAILED"
#         fi
#     done
# ) &
# KEEPALIVE_PID=$!

# # Wait for server process
# wait $SERVER_PID
# SERVER_EXIT=$?
# echo "[$(date)] Server exited with code $SERVER_EXIT"

# # Clean up keepalive
# kill $KEEPALIVE_PID 2>/dev/null

# # If server died, restart it
# if [ $SERVER_EXIT -ne 0 ] && [ $attempt -lt $MAX_RESTARTS ]; then
#     echo "[$(date)] Restarting server..."
#     exec "$0" "$@"
# fi
#  --disable-cuda-graph
# vllm serve Qwen/Qwen3.5-35B-A3B \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.8 \
#   --max-model-len 4096 \
#   --enable-reasoning \
#   --reasoning-parser qwen3 \
#   --dtype bfloat16 2>&1

# vllm serve DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.8 \
#   --max-model-len 4096 \
#   --dtype auto \
#   2>&1

  # --trust-remote-code \
# python grpo_single_phase_llm/benchmark/bench_judge.py \
#     --model Qwen/Qwen3.5-35B-A3B \
#     --port 8899 \
#     --gpu 0 \
#     --startup-timeout 3600 \
#     --num-requests "${1:-50}" \
#     --batch-sizes "1,4,8,16,32,64"
