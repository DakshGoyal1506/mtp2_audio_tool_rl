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

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi

conda activate slm
export TRANSFORMERS_OFFLINE=1

MTP2_REPO_ROOT="${MTP2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MTP2_DATA_ROOT="${MTP2_DATA_ROOT:?Set MTP2_DATA_ROOT to the external dataset root.}"
MTP2_OUTPUT_ROOT="${MTP2_OUTPUT_ROOT:?Set MTP2_OUTPUT_ROOT to an external output directory.}"
MTP2_EMBED_ROOT="${MTP2_EMBED_ROOT:-${MTP2_DATA_ROOT}/precomputed_embeds}"
MTP2_MANIFEST_ROOT="${MTP2_MANIFEST_ROOT:-${MTP2_DATA_ROOT}}"
MTP2_WORKSPACE_ROOT="${MTP2_WORKSPACE_ROOT:?Set MTP2_WORKSPACE_ROOT to the external workspace containing inference/tool_execute.py.}"

PROJECT_DIR="${MTP2_REPO_ROOT}/scripts/data/synthetic"
OUTPUT_DIR="${MTP2_OUTPUT_ROOT}/tool_base_output.json"
EVAL_LOG="${MTP2_OUTPUT_ROOT}/tool_base_output.eval.log"
TOOL_EXECUTE_SCRIPT="${MTP2_WORKSPACE_ROOT}/inference/tool_execute.py"
EVALUATION_SCRIPT="${MTP2_REPO_ROOT}/src/mtp2_audio_tool_rl/evaluation/evaluation.py"

mkdir -p "$MTP2_OUTPUT_ROOT"

cd "$PROJECT_DIR"

echo "=============================================="

python "$TOOL_EXECUTE_SCRIPT" \
    --model desta-8b \
    --data "${MTP2_MANIFEST_ROOT}/synth_cached_train_verified_gold.json" \
    --precomputed-embed-dir "$MTP2_EMBED_ROOT" \
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
python "$EVALUATION_SCRIPT" --input "$OUTPUT_DIR" | tee "$EVAL_LOG"
