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

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi

MTP2_REPO_ROOT="${MTP2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MTP2_DATA_ROOT="${MTP2_DATA_ROOT:?Set MTP2_DATA_ROOT to the external dataset root.}"
MTP2_EMBED_ROOT="${MTP2_EMBED_ROOT:-${MTP2_DATA_ROOT}/precomputed_embeds}"
MTP2_AUDIO_ROOT="${MTP2_AUDIO_ROOT:-${MTP2_DATA_ROOT}/audio}"
MTP2_MANIFEST_ROOT="${MTP2_MANIFEST_ROOT:-${MTP2_DATA_ROOT}}"

cd "$MTP2_REPO_ROOT"

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
python3 scripts/data/synthetic/check_audio.py

# --- Step 1: Main audio processing (speech_recognition, diarization, etc.) ---
echo ""
echo "=== Step 1: Running main audio processing tools ==="
activate_env audio
python3 scripts/data/synthetic/process_audio.py
echo "=== Step 1 Done ==="
conda deactivate

# --- Steps 2 & 3: stress analysis + chord recognition (need legacy keras for autochord) ---
activate_env autochord
export TF_USE_LEGACY_KERAS=1

echo ""
echo "=== Step 2: Running stress analysis ==="
python3 scripts/data/synthetic/process_stress.py
echo "=== Step 2 Done ==="

echo ""
echo "=== Step 3: Running chord recognition ==="
python3 scripts/data/synthetic/process_chord.py
echo "=== Step 3 Done ==="
conda deactivate

# --- Step 4: Merge all tool outputs ---
echo ""
echo "=== Step 4: Merging tool outputs ==="
python3 scripts/data/synthetic/merge_tool_outputs.py
echo "=== Step 4 Done ==="

# --- Step 5: Extract Qformer embeddings ---
echo ""
echo "=== Step 5: Extracting embeddings ==="
activate_env slm
python3 scripts/data/synthetic/extract_embeddings.py \
    --model_path "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B" \
    --audio_dir "$MTP2_AUDIO_ROOT" \
    --data_jsonl "${MTP2_MANIFEST_ROOT}/grpo_tool_dataset.with_relative_audio.jsonl" \
    --out_dir "$MTP2_EMBED_ROOT"
echo "=== Step 5 Done ==="

# --- Step 6: Final verification ---
echo ""
echo "=== Step 6: Verify embeds ==="
python3 scripts/data/synthetic/check_audio.py

echo "All done!"
