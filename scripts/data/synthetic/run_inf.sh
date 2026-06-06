#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=infer-synth-tool
#SBATCH --output=logs/infer_synth_tool_slurm_%j.log

eval "$(/home/speech-nlp-cse/24m0756/anaconda3/bin/conda shell.bash hook)"
conda activate slm
export TRANSFORMERS_OFFLINE=1

PARENT_DIR="/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo"
PROJECT_DIR="/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset"
OUTPUT_DIR="/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/tool_base_output.json"
EVAL_LOG="/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/tool_base_output.eval.log"

cd "$PROJECT_DIR"

echo "=============================================="

python "$PARENT_DIR/inference/tool_execute.py" \
    --model desta-8b \
    --data synth_cached_train_verified_gold.json \
    --precomputed-embed-dir precomputed_embeds \
    --output "$OUTPUT_DIR"

echo "=============================================="
echo "Post-processing: filling empty metadata fields"
echo "=============================================="
python -c "
import json, sys
with open('$OUTPUT_DIR') as f:
    results = json.load(f)
for r in results:
    task = r.get('task', 'speech')
    if r.get('difficulty', '') not in ('easy', 'medium', 'hard'):
        r['difficulty'] = 'medium'
    if not r.get('category'):
        r['category'] = task
    if not r.get('sub-category'):
        r['sub-category'] = task
with open('$OUTPUT_DIR', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Filled metadata for {len(results)} entries')
"

echo "=============================================="
echo "Evaluation: $OUTPUT_DIR"
echo "=============================================="
python "$PARENT_DIR/evaluation.py" --input "$OUTPUT_DIR" | tee "$EVAL_LOG"