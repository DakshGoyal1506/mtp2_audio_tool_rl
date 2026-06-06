#!/bin/bash
#SBATCH --job-name=filter_job
#SBATCH --output=filtering/job_%j.out
#SBATCH --error=filtering/job_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

# Do not request GPUs
# #SBATCH --gres=gpu:0

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda is not available. Load conda or install it under \$HOME/anaconda3." >&2
    exit 1
fi
conda activate slm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MTP2_WORKSPACE_ROOT="${MTP2_WORKSPACE_ROOT:-}"
MTP2_DATA_ROOT="${MTP2_DATA_ROOT:-}"

if [ -z "$MTP2_DATA_ROOT" ] && [ -n "$MTP2_WORKSPACE_ROOT" ]; then
    MTP2_DATA_ROOT="$MTP2_WORKSPACE_ROOT/grpo_dataset"
fi
if [ -z "$MTP2_DATA_ROOT" ]; then
    MTP2_DATA_ROOT="$PROJECT_DIR/scripts/data"
fi

cd "$MTP2_DATA_ROOT"

echo "Starting dataset filtering job..."
cd "${MTP2_FILTER_SUBDIR:-filtering}"
python filter_dataset.py
echo "Done!"
