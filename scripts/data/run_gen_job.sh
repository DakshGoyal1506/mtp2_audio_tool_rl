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

source /home/speech-nlp-cse/24m0756/anaconda3/etc/profile.d/conda.sh
conda activate slm
cd /home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/

echo "Starting dataset generation job..."
python generate.py --samples-per-tool 100 --resume
echo "Done!"
