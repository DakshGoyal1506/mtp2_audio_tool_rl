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
source /home/speech-nlp-cse/24m0756/anaconda3/bin/activate slm

# Ensure ffmpeg backend for torchaudio (avoid torchcodec dependency)
export TORCHAUDIO_BACKEND=ffmpeg

cd /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo

echo "=== Extracting embeddings ==="
python3 synthetic_dataset/extract_embeddings.py \
    --model_path "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B" \
    --audio_dir "dataset/audio" \
    --data_jsonl "dataset/grpo_tool_dataset.with_relative_audio.jsonl" \
    --out_dir "synthetic_dataset/precomputed_embeds"

echo ""
echo "=== Verify embeds ==="
python3 synthetic_dataset/check_audio.py

echo "All done!"
