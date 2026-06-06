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

source /home/speech-nlp-cse/24m0756/anaconda3/etc/profile.d/conda.sh
conda activate slm
cd /home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/

echo "Starting dataset filtering job..."
cd filtering
python filter_dataset.py
echo "Done!"
