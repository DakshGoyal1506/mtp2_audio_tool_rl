#!/bin/bash
#SBATCH --job-name=wavecaps_audio
#SBATCH --output=cache_wavecaps_audio_job_%j.log
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

source /home/speech-nlp-cse/24m0756/anaconda3/etc/profile.d/conda.sh

cd /home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/

# Force offline mode — all models must be pre-cached (run download_models.py on login node first)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

activate_env() {
    local env_name="$1"
    conda activate "$env_name"
    CONDA_ENV_DIR="$(conda info --base)/envs/$env_name"
    export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_DIR}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

# --- Step 1: main audio processing (no autochord needed) ---
echo "=== Step 1: Running main audio processing tools ==="
activate_env audio
python process_audio_wavecaps.py
echo "=== Step 1 Done ==="
conda deactivate

# --- Steps 2 & 3: stress analysis + chord recognition (need legacy keras for autochord) ---
activate_env autochord
export TF_USE_LEGACY_KERAS=1

echo "=== Step 2: Running stress analysis ==="
python process_stress_wavecaps.py
echo "=== Step 2 Done ==="

echo "=== Step 3: Running chord recognition ==="
python process_chord_wavecaps.py
echo "=== Step 3 Done ==="

echo "All processing complete!"
