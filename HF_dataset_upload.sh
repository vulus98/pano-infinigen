#!/usr/bin/env bash
#SBATCH --job-name=hf-dataset-upload
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=150:00:00
#SBATCH -o logs/upload_%j.out

# Resources:
#   The upload path is now single-process streaming (no Dataset.map workers).
#   Peak RAM ≈ --mini-batch × sample_size. With mini_batch=8 and ~500 MB/sample
#   that's ~4 GB, so 4×8 = 32 GB is plenty of headroom. Increase --mini-batch
#   if you want slightly higher throughput and have RAM to spare; decrease to
#   4 or 2 if you still see OOMs.
#   CPUs are only used for PNG/Arrow/parquet compression by the single process;
#   4 cores is more than enough.

# Standard environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen

# Essential for ETH Zurich clusters to reach Hugging Face
module load eth_proxy

export HF_DATASETS_CACHE="/cluster/scratch/vbozic"

echo "$(date) Starting upload to Hugging Face..."

python create_HF_dataset.py --mini-batch 8

echo "$(date) Upload finished with exit code $?"