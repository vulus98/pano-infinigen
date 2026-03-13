#!/usr/bin/env bash
#SBATCH --job-name=hf-dataset-upload
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=150:00:00
#SBATCH -o logs/upload_%j.out

# Standard environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen

# Essential for ETH Zurich clusters to reach Hugging Face
module load eth_proxy

export HF_DATASETS_CACHE="/cluster/scratch/vbozic"

echo "$(date) Starting upload to Hugging Face..."

# Run the python script
# Make sure the script name matches your file name (e.g., upload_to_hf.py)
python create_HF_dataset.py

echo "$(date) Upload finished with exit code $?"
