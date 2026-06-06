#!/bin/bash
#SBATCH --job-name=synth_embed
#SBATCH --output=synthetic_dataset/extract_embeds_%j.log
#SBATCH --error=synthetic_dataset/extract_embeds_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Load environment
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi

conda activate slm

# Ensure ffmpeg backend for torchaudio (avoid torchcodec dependency)
export TORCHAUDIO_BACKEND=ffmpeg

MTP2_REPO_ROOT="${MTP2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MTP2_DATA_ROOT="${MTP2_DATA_ROOT:?Set MTP2_DATA_ROOT to the external dataset root.}"
MTP2_EMBED_ROOT="${MTP2_EMBED_ROOT:-${MTP2_DATA_ROOT}/precomputed_embeds}"
MTP2_AUDIO_ROOT="${MTP2_AUDIO_ROOT:-${MTP2_DATA_ROOT}/audio}"
MTP2_MANIFEST_ROOT="${MTP2_MANIFEST_ROOT:-${MTP2_DATA_ROOT}}"

cd "$MTP2_REPO_ROOT"

echo "=== Extracting embeddings ==="
python3 scripts/data/synthetic/extract_embeddings.py \
    --model_path "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B" \
    --audio_dir "$MTP2_AUDIO_ROOT" \
    --data_jsonl "${MTP2_MANIFEST_ROOT}/grpo_tool_dataset.with_relative_audio.jsonl" \
    --out_dir "$MTP2_EMBED_ROOT"

echo ""
echo "=== Verify embeds ==="
python3 scripts/data/synthetic/check_audio.py

echo "All done!"
