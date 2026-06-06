#!/bin/bash
#SBATCH --job-name=desta-vllm-bench
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --exclude=cn14-dgx,cn22-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=results/vllm/bench_%j.log

# =============================================================================
# DeSTA2.5 + vLLM Benchmark — MMAU Tool-Assisted Evaluation
#
# Runs the desta_vllm benchmark inside the pre-built vLLM Apptainer image.
# The DeSTA model (Llama-3.1-8B-Instruct) is loaded from HF cache and
# precomputed QFormer embeddings are used instead of running Whisper/QFormer.
#
# Usage:
#   sbatch desta_vllm/run_benchmark.sh
#   sbatch desta_vllm/run_benchmark.sh --direct        # no tool selection
#   sbatch desta_vllm/run_benchmark.sh --lora checkpoints/grpo-desta-XXXX
# =============================================================================

# ── Conda (needed for apptainer to be on PATH) ───────────────────────────────
__conda_setup="$('/home/speech-nlp-cse/24m0756/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate gemma

set -eo pipefail

# ── Parse CLI args passed after sbatch ────────────────────────────────────────
MODE="tools"        # "tools" or "direct"
LORA_PATH=""
DATA_FILE="mmau-test-mini-cached.json"
EMBED_DIR="precomputed_embeds"
MAX_MODEL_LEN=4096
MAX_NEW_TOKENS=2048
GPU_MEM=0.9
PARTITION=""         # override --partition if passed

BATCH_SIZE=64

while [[ $# -gt 0 ]]; do
    case "$1" in
        --direct)       MODE="direct"; shift ;;
        --lora)         LORA_PATH="$2"; shift 2 ;;
        --data)         DATA_FILE="$2"; shift 2 ;;
        --embed-dir)    EMBED_DIR="$2"; shift 2 ;;
        --max-len)      MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-tokens)   MAX_NEW_TOKENS="$2"; shift 2 ;;
        --gpu-mem)      GPU_MEM="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
# SLURM copies the script to /tmp, so BASH_SOURCE[0] won't resolve.
# Use SLURM_SUBMIT_DIR (set to cwd at sbatch time) like run_trl_v4.sh does.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi
cd "$PROJECT_DIR"
echo "Working dir: $PWD"

IFILE="./gemma4.sif"
if [ ! -f "$IFILE" ]; then
    echo "ERROR: $IFILE not found!"
    echo "Pull it first: apptainer pull gemma4.sif docker://vllm/vllm-openai:latest"
    exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR="results/vllm"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LORA_TAG=""
if [ -n "$LORA_PATH" ]; then
    LORA_TAG="-lora-$(basename "$LORA_PATH")"
fi
OUTPUT_FILE="${RESULTS_DIR}/${MODE}${LORA_TAG}-${SLURM_JOB_ID:-${TIMESTAMP}}.json"

# ── Banner ────────────────────────────────────────────────────────────────────
NODE_IP=$(hostname -i 2>/dev/null | awk '{print $1}' || echo "unknown")
NODE_NAME=$(hostname -s)

echo "=============================================="
echo "  DeSTA vLLM Benchmark"
echo "=============================================="
echo "  Mode:       ${MODE}"
echo "  Data:       ${DATA_FILE}"
echo "  Embeds:     ${EMBED_DIR}"
echo "  Model:      DeSTA-ntu/Llama-3.1-8B-Instruct"
echo "  LoRA:       ${LORA_PATH:-none}"
echo "  Max len:    ${MAX_MODEL_LEN}"
echo "  GPU mem:    ${GPU_MEM}"
echo "  Job:        ${SLURM_JOB_ID:-local}"
echo "  Node:       ${NODE_NAME} (${NODE_IP})"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Output:     ${OUTPUT_FILE}"
echo "=============================================="
echo ""

# ── Build the Python command ──────────────────────────────────────────────────
PYTHON_CMD="python3 -m desta_vllm.run_benchmark \
    --data '${DATA_FILE}' \
    --embed-dir '${EMBED_DIR}' \
    --output '${OUTPUT_FILE}' \
    --model DeSTA-ntu/Llama-3.1-8B-Instruct \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --gpu-memory ${GPU_MEM} \
    --tensor-parallel 1 \
    --batch-size ${BATCH_SIZE}"

if [ "$MODE" = "direct" ]; then
    PYTHON_CMD="${PYTHON_CMD} --direct"
fi

if [ -n "$LORA_PATH" ]; then
    PYTHON_CMD="${PYTHON_CMD} --lora-path '${LORA_PATH}'"
fi

echo ">>> Command: ${PYTHON_CMD}"
echo ""

# ── Run inside Apptainer ─────────────────────────────────────────────────────
# The gemma4.sif container has vLLM + PyTorch + transformers pre-installed.
# We bind-mount:
#   - /dev/shm         (shared memory for NCCL)
#   - HF_HOME          (model cache)
#   - PROJECT_DIR       (our code + data + embeds)
#   - HOME              (for any stray config reads)
# A writable tmpdir is created for vLLM's cache and IPC files.

echo "Starting benchmark at $(date)"

# Create a writable scratch dir for the container
SCRATCH_DIR="${RESULTS_DIR}/scratch_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH_DIR"
trap "rm -rf '$SCRATCH_DIR'" EXIT

apptainer exec --cleanenv --nv \
    -B /dev/shm \
    -B "${HF_HOME}:${HF_HOME}" \
    -B "${PROJECT_DIR}:${PROJECT_DIR}" \
    -B "${HOME}:${HOME}" \
    -B "${SCRATCH_DIR}:/scratch" \
    --env HOME="/scratch" \
    --env TMPDIR="/scratch" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_OFFLINE_MODE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env PYTHONPATH="${PROJECT_DIR}" \
    --env PYTHONUNBUFFERED=1 \
    --env USER=user \
    --env VLLM_LOGGING_LEVEL=INFO \
    --env VLLM_CONFIGURE_LOGGING=1 \
    --env VLLM_CACHE_ROOT="/scratch/.cache/vllm" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$IFILE" \
    bash -c "mkdir -p /scratch/.cache/vllm && cd '${PROJECT_DIR}' && stdbuf -oL -eL ${PYTHON_CMD}"

EXIT_CODE=$?

echo ""
echo "Benchmark finished at $(date) (exit code: ${EXIT_CODE})"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Results: ${OUTPUT_FILE}"
    # Print summary from the JSON
    if [ -f "${OUTPUT_FILE}" ]; then
        python3 -c "
import json, sys
with open('${OUTPUT_FILE}') as f:
    data = json.load(f)
total = len(data)
correct = sum(1 for d in data if d.get('is_correct', False))
print(f'  Total: {total}, Correct: {correct}, Accuracy: {correct/total*100:.1f}%' if total else '  No results')
tasks = {}
for d in data:
    t = d.get('task', 'unknown')
    tasks.setdefault(t, [0, 0])
    tasks[t][1] += 1
    if d.get('is_correct'):
        tasks[t][0] += 1
for t in sorted(tasks):
    c, n = tasks[t]
    print(f'  {t:30s}: {c:3d}/{n:3d} = {c/n*100:5.1f}%')
" 2>/dev/null || true
    fi
fi

exit $EXIT_CODE
