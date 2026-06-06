#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi
conda activate slm

export WANDB_MODE=offline
export WANDB_DISABLE_STATS=true
export WANDB_DISABLE_SYSTEM_REPORTS=true
export WANDB_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-desta-audio}"
export WANDB_RUN_ID="${SLURM_JOB_ID:-}"
export WANDB_ERROR_REPORTING=false
export WANDB_START_METHOD=thread

CONDA_ENV_DIR="$(conda info --base)/envs/slm"
export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_DIR}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=15
echo "CUDA_HOME: $CUDA_HOME"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    grpo/train_trl.py > trl_debug.log 2>&1
