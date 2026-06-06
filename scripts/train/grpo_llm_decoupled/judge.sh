#!/bin/bash
#SBATCH --job-name=apptainer-gpu
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --output=judge_logs/judge_gemma4_job_%j.log


# 1. Initialize Conda
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi
conda activate gemma

# 2. Check if the image file exists
# IFILE="./gemma.sif"
# if [ ! -f "$IFILE" ]; then
#     echo "ERROR: $IFILE not found! Did you run 'apptainer pull' on the login node?"
#     exit 1
# fi

# echo "Starting Apptainer execution at $(date)"

# # 3. Run the local SIF file
# # -d: Debug mode (prints everything to the log)
# # --nv: Enable GPU
# apptainer -d exec --nv "$IFILE" nvidia-smi

# echo "Finished at $(date)"

IFILE="./gemma4.sif"
if [ ! -f "$IFILE" ]; then
    echo "ERROR: $IFILE not found! Ensure you pulled 'vllm/vllm-openai:gemma4'"
    exit 1
fi

export HF_HOME=$HOME/.cache/huggingface

echo "Starting vLLM Gemma 4 at $(date)"

# 3. Run the Command
# We use 'vllm serve' which is the entrypoint for that container
# apptainer exec  --cleanenv --nv \
#     -B /dev/shm:/dev/shm \
#     -B $HF_HOME:$HF_HOME \
#     -B $HOME:$HOME \
#     --env HF_HOME=$HF_HOME \
#     --env HF_HUB_OFFLINE_MODE=1 \
#     --env TRANSFORMERS_OFFLINE=1 \
#     "$IFILE" \
#     vllm serve google/gemma-4-26B-A4B-it \
#         --tensor-parallel-size 2 \
#         --max-model-len 8192 \
#         --gpu-memory-utilization 0.90 \
#         --host 0.0.0.0 \
#         --port 8000 \
#         --trust-remote-code


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

# 1. Manually set identity variables that the system is failing to provide
export USER=user
export LOGNAME=user
export HOME_FOLDER=$(pwd) # Use current directory as a reference


# 2. Run Apptainer with absolute isolation
# --contain: Isolates the container from the host's broken environment
# --no-home: CRITICAL. Tells Apptainer NOT to look up your user info to find a home dir
# --home /tmp: Gives the container a generic 'working' home directory
apptainer exec --cleanenv --nv \
    --contain \
    --no-home \
    --home /tmp \
    --workdir /tmp \
    -B /dev/shm \
    -B "$HF_HOME:$HF_HOME" \
    -B "$HOME:$HOME" \
    --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE_MODE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env USER=user \
    "$IFILE" \
    vllm serve google/gemma-4-26B-A4B-it \
        --tensor-parallel-size 2 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.90 \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code

echo "Job finished at $(date)"
