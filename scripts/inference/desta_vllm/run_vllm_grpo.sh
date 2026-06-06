#!/bin/bash
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --exclude=cn14-dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --job-name=grpo-vllm
#SBATCH --output=expts/%j/grpo_vllm_slurm_%j.log

# =============================================================================
# GRPO Training with vLLM-Accelerated Rollouts
#
# Runs inside gemma4.sif Apptainer container (vLLM 0.19+) with the gemma
# conda env's site-packages for TRL/PEFT/accelerate (both Python 3.12).
# Judge server is managed externally (SGLang via bench_judge.sh).
# =============================================================================

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi
conda activate gemma

# --- Paths ---
CONDA_ENV_DIR="$(conda info --base)/envs/gemma"
CONDA_SITE_PKGS="$CONDA_ENV_DIR/lib/python3.12/site-packages"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
fi

PROJECT_DIR="$PWD"
DESTA_DIR="$(dirname "$PROJECT_DIR")/DeSTA2.5-Audio"
echo "Working dir: $PROJECT_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PROJECT_DIR/.cache/vllm}"

IFILE="./gemma4.sif"
if [ ! -f "$IFILE" ]; then
    echo "ERROR: $IFILE not found!"
    exit 1
fi

# --- Config ---
CONFIG="${1:-desta_vllm/training/configs/vllm_grpo_v1.yaml}"

# --- Create experiment directory and snapshot code ---
EXPT_DIR="expts/${SLURM_JOB_ID:-$$}"
if [ -f scripts/snapshot_expt.sh ]; then
    source scripts/snapshot_expt.sh
    snapshot_expt "$EXPT_DIR" \
        --config   "$CONFIG" \
        --code-dir desta_vllm \
        --code-dir grpo_single_phase \
        --code-dir grpo_single_phase_llm \
        --launcher "${BASH_SOURCE[0]}"
fi

# Read config values (using conda env python, before entering container)
# Colocate mode: one accelerate process per visible GPU, each runs both
# the HF training model and its own vLLM rollout engine on the same GPU.
NUM_TRAIN_GPUS="${SLURM_GPUS_ON_NODE:-4}"

JUDGE_HOST=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(cfg.get('judge_host', 'localhost'))
")
JUDGE_PORT=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(cfg.get('judge_port', 8000))
")

MIXED_PRECISION=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(cfg.get('mixed_precision', 'bf16'))
")

echo "=============================================="
echo "  GRPO Training — vLLM-Accelerated Rollouts"
echo "=============================================="
echo "  Config            : $CONFIG"
echo "  Training GPUs     : $NUM_TRAIN_GPUS"
echo "  Judge endpoint    : http://${JUDGE_HOST}:${JUDGE_PORT}"
echo "  Container         : $IFILE"
echo "  Conda site-pkgs   : $CONDA_SITE_PKGS"
echo "=============================================="

JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}"
if curl --noproxy "*" -s --max-time 5 "$JUDGE_URL/health" > /dev/null 2>&1; then
    echo "Judge server reachable at $JUDGE_URL"
else
    echo "WARNING: Judge server NOT reachable at $JUDGE_URL — will use fallback scores."
fi

# --- Distributed training setup ---
MASTER_ADDR="127.0.0.1"
MASTER_PORT="$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")"

# --- Scratch dir for container ---
SCRATCH_DIR="expts/${SLURM_JOB_ID:-$$}/scratch"
mkdir -p "$SCRATCH_DIR"
trap "rm -rf '$SCRATCH_DIR'" EXIT

START_TIME=$(date +%s)

# --- Build the accelerate command ---
# Colocate mode: each accelerate process binds to one GPU (via LOCAL_RANK)
# and runs both a vLLM rollout engine and the HF training model on that GPU.
# vLLM uses distributed_executor_backend="external_launcher" so it does not
# spawn its own subprocess — it shares the process's CUDA context.
TOTAL_GPUS=${SLURM_GPUS_ON_NODE:-4}
CUDA_DEVICES=$(seq -s, 0 $((TOTAL_GPUS-1)))

TRAIN_CMD="accelerate launch \
    --num_processes=$NUM_TRAIN_GPUS \
    --num_machines=1 \
    --mixed_precision=$MIXED_PRECISION \
    --dynamo_backend=no \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    desta_vllm/training/train_trl.py $CONFIG"

echo ">>> Command: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${TRAIN_CMD}"
echo ""

# --- Run inside Apptainer container ---
# Container provides: vLLM 0.19+, PyTorch, transformers
# Conda env provides (via PYTHONPATH): TRL, PEFT, accelerate, omegaconf, etc.
# Project provides (via PYTHONPATH): desta_vllm, grpo_single_phase, DeSTA source

apptainer exec --cleanenv --nv \
    -B /dev/shm \
    -B "${HF_HOME}:${HF_HOME}" \
    -B "${PROJECT_DIR}:${PROJECT_DIR}" \
    -B "${DESTA_DIR}:${DESTA_DIR}" \
    -B "${HOME}:${HOME}" \
    -B "${CONDA_ENV_DIR}:${CONDA_ENV_DIR}:ro" \
    -B "${SCRATCH_DIR}:/scratch" \
    --env TMPDIR="/scratch" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONPATH="${PROJECT_DIR}:${DESTA_DIR}" \
    --env CONDA_SITE_PKGS="${CONDA_SITE_PKGS}" \
    --env PYTHONUNBUFFERED=1 \
    --env USER=user \
    --env WANDB_DISABLED=true \
    --env WANDB_MODE=disabled \
    --env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env NCCL_SOCKET_IFNAME=lo \
    --env GLOO_SOCKET_IFNAME=lo \
    --env NCCL_TIMEOUT=3600 \
    --env TORCH_NCCL_BLOCKING_WAIT=0 \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --env VLLM_LOGGING_LEVEL=INFO \
    --env VLLM_CONFIGURE_LOGGING=1 \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env VLLM_HOST_IP=127.0.0.1 \
    "$IFILE" \
    bash -c "mkdir -p '${VLLM_CACHE_ROOT}' && cd '${PROJECT_DIR}' && stdbuf -oL -eL ${TRAIN_CMD}"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
printf "Training time: %02dh %02dm %02ds\n" $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))
echo "Training finished with exit code $EXIT_CODE."
