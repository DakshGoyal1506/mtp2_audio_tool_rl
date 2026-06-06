#!/bin/bash
#SBATCH --job-name=synth_cache
#SBATCH --output=synthetic_dataset/cache_tools_%j.log
#SBATCH --error=synthetic_dataset/cache_tools_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /home/speech-nlp-cse/24m0756/anaconda3/etc/profile.d/conda.sh

cd /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo

activate_env() {
    local env_name="$1"
    conda activate "$env_name"
    CONDA_ENV_DIR="$(conda info --base)/envs/$env_name"
    export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_DIR}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

echo "=== Step 0: Audio checks ==="
python3 synthetic_dataset/check_audio.py

# --- Step 1: Main audio processing (speech_recognition, diarization, etc.) ---
echo ""
echo "=== Step 1: Running main audio processing tools ==="
activate_env audio
python3 synthetic_dataset/process_audio.py
echo "=== Step 1 Done ==="
conda deactivate

# --- Steps 2 & 3: stress analysis + chord recognition (need legacy keras for autochord) ---
activate_env autochord
export TF_USE_LEGACY_KERAS=1

echo ""
echo "=== Step 2: Running stress analysis ==="
python3 synthetic_dataset/process_stress.py
echo "=== Step 2 Done ==="

echo ""
echo "=== Step 3: Running chord recognition ==="
python3 synthetic_dataset/process_chord.py
echo "=== Step 3 Done ==="
conda deactivate

# --- Step 4: Merge all tool outputs ---
echo ""
echo "=== Step 4: Merging tool outputs ==="
python3 synthetic_dataset/merge_tool_outputs.py
echo "=== Step 4 Done ==="

# --- Step 5: Extract Qformer embeddings ---
echo ""
echo "=== Step 5: Extracting embeddings ==="
activate_env slm
python3 synthetic_dataset/extract_embeddings.py \
    --model_path "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B" \
    --audio_dir "dataset/audio" \
    --data_jsonl "dataset/grpo_tool_dataset.with_relative_audio.jsonl" \
    --out_dir "synthetic_dataset/precomputed_embeds"
echo "=== Step 5 Done ==="

# --- Step 6: Final verification ---
echo ""
echo "=== Step 6: Verify embeds ==="
python3 synthetic_dataset/check_audio.py

echo "All done!"
