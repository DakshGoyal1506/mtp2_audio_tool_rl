#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --exclude=cn15-dgx
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diag-embed
#SBATCH --output=diag_embed_%j.log

eval "$(/home/speech-nlp-cse/24m0756/anaconda3/bin/conda shell.bash hook)"
conda activate slm

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
echo "Working dir: $PWD"

python3 scripts/diag_embed_gpu.py 2>&1
echo "Exit code: $?"
