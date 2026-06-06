#!/bin/bash
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --exclude=cn15-dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --job-name=grpo-trl-v4-llm
#SBATCH --output=grpo_trl_v4_slurm_%j.log

# =============================================================================
# GRPO Training v4 — Decoupled Heuristic Rewards
# =============================================================================

eval "$(/home/speech-nlp-cse/24m0756/anaconda3/bin/conda shell.bash hook)"
conda activate slm

# --- Set CUDA_HOME from conda env ---
CONDA_ENV_DIR="$(conda info --base)/envs/slm"
export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_DIR}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=15
echo "CUDA_HOME: $CUDA_HOME"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- WandB offline mode ---
export WANDB_MODE=offline
export WANDB_DISABLE_STATS=true
export WANDB_DISABLE_SYSTEM_REPORTS=true
export WANDB_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-desta-audio-v4-decoupled}"
export WANDB_RUN_ID="${SLURM_JOB_ID:-}"
export WANDB_ERROR_REPORTING=false
export WANDB_START_METHOD=thread

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$(dirname "$SCRIPT_DIR")"
fi

echo "Working dir: $PWD"

# --- Config ---
CONFIG="${1:-grpo_single_phase_llm_decoupled/configs/optimized_v4.yaml}"

# Read number of training GPUs from config (default: all visible GPUs)
NUM_TRAIN_GPUS=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(cfg.get('num_gpus', ${SLURM_GPUS_ON_NODE:-1}))
")

echo "=============================================="
echo "  GRPO TRL Training v4 — Decoupled Heuristic"
echo "=============================================="
echo "  Working directory : $PWD"
echo "  Config            : $CONFIG"
echo "  Training GPUs     : $NUM_TRAIN_GPUS (GPU 0-$((NUM_TRAIN_GPUS-1)))"
echo "=============================================="

# --- Distributed training setup ---
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
# Increase NCCL timeout to 60 min (default 30 min) to tolerate slow judge calls
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

MIXED_PRECISION=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(cfg.get('mixed_precision', 'bf16'))
")

START_TIME=$(date +%s)

# --- Launch training on GPUs 0-(N-1) ---
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_TRAIN_GPUS-1))) \
accelerate launch \
    --num_processes="$NUM_TRAIN_GPUS" \
    --num_machines=1 \
    --mixed_precision="$MIXED_PRECISION" \
    --dynamo_backend=no \
    --main_process_ip="$MASTER_ADDR" \
    --main_process_port="$MASTER_PORT" \
    grpo_single_phase_llm_decoupled/train_trl.py "$CONFIG"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
printf "Training time: %02dh %02dm %02ds\n" $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))
echo "Training finished with exit code $EXIT_CODE."
