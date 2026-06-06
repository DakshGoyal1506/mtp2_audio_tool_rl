#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=generate_job_%j.log
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

MTP2_REPO_ROOT="${MTP2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$MTP2_REPO_ROOT"
export PYTHONPATH="$MTP2_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting dataset generation job..."
python -m mtp2_audio_tool_rl.datasets.generate --samples-per-tool 100 --resume
echo "Done!"
