#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --exclude=cn15-dgx
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --job-name=grpo-trl-v3
#SBATCH --output=grpo_trl_v3_slurm_%j.log

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi
conda activate slm

# --- Set CUDA_HOME from conda env (nvcc installed via cuda-toolkit) ---
CONDA_ENV_DIR="$(conda info --base)/envs/slm"
export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_DIR}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=15
echo "CUDA_HOME: $CUDA_HOME"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- WandB offline mode ---
export WANDB_MODE=offline
export WANDB_DISABLE_STATS=true
export WANDB_DISABLE_SYSTEM_REPORTS=true
export WANDB_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-desta-audio-v3}"
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

# --- Parameters ---
NUM_GPUS=4
CONFIG="${1:-grpo_single_phase/configs/optimized_v3.yaml}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

echo "=============================================="
echo "  GRPO TRL Training v3 — DeSTA2.5-Audio"
echo "=============================================="
echo "  Working directory : $PWD"
echo "  Config            : $CONFIG"
echo "  GPUs              : $NUM_GPUS"
echo "  MASTER_ADDR       : $MASTER_ADDR"
echo "  MASTER_PORT       : $MASTER_PORT"
echo "=============================================="

START_TIME=$(date +%s)

MIXED_PRECISION=$(python3 -c "
import sys; from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(cfg.get('mixed_precision', 'bf16'))
")

accelerate launch \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision="$MIXED_PRECISION" \
    --dynamo_backend=no \
    --main_process_ip="$MASTER_ADDR" \
    --main_process_port="$MASTER_PORT" \
    grpo_single_phase/train_trl.py "$CONFIG"
