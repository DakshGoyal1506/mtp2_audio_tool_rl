#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=cn15-dgx
#SBATCH --job-name=test-vllm
#SBATCH --output=logs/test_vllm_slurm_%j.log

# =============================================================================
# Quick smoke test: try to serve a small model with vLLM on DGX
# =============================================================================

eval "$(/home/speech-nlp-cse/24m0756/anaconda3/bin/conda shell.bash hook)"
conda activate vllm-env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(dirname $(realpath $0))))}"
echo "Working dir: $PWD"
mkdir -p logs

echo "=============================================="
echo "  vLLM Smoke Test"
echo "=============================================="
echo "  Model: DeSTA-ntu/Llama-3.1-8B-Instruct"
echo "  Job:   ${SLURM_JOB_ID:-local}"
echo "  CPU:   $(cat /proc/cpuinfo | grep 'model name' | head -1)"
echo "  AVX512: $(grep -o 'avx512f' /proc/cpuinfo | head -1 || echo 'NOT FOUND')"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
echo "  vllm:  $(pip show vllm 2>/dev/null | grep Version)"
echo "  torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "=============================================="

echo ""
echo "--- Step 1: Quick import test ---"
python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
" 2>&1

echo ""
echo "--- Step 2: vLLM import test ---"
python -c "
import vllm
print(f'vllm {vllm.__version__} imported OK')
" 2>&1

echo ""
echo "--- Step 3: Trying vllm serve (Llama-3.1-8B-Instruct, port 8111) ---"
timeout 120 vllm serve DeSTA-ntu/Llama-3.1-8B-Instruct \
  --port 8111 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 512 \
  --dtype bfloat16 2>&1

EXIT_CODE=$?
echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 132 ]; then
    echo "RESULT: SIGILL (Illegal instruction) - AVX512 issue confirmed!"
    echo "FIX: Need to rebuild vllm from source or downgrade."
elif [ $EXIT_CODE -eq 124 ]; then
    echo "RESULT: Server started successfully (timed out after 120s = working!)"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: Server exited cleanly."
else
    echo "RESULT: Exited with code $EXIT_CODE"
fi
echo "=============================================="
