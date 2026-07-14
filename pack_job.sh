#!/bin/bash
#SBATCH --job-name=pack-dataset
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=16:00:00
#SBATCH -o logs/pack_%j.out
# Package rendered outputs into the compact tar-per-scene dataset (see pack_dataset.py).
# Usage: sbatch pack_job.sh [domains...]   e.g. sbatch pack_job.sh indoor urban
#        (defaults to all three domains)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen
cd /cluster/work/igp_psr/vbozic/pano-infinigen
if [ "$1" = "meta" ]; then
    echo "rebuilding dataset metadata only (dataset_info.json + index.jsonl)"
    python pack_dataset.py --dst dataset --metadata-only --workers 16
else
    DOMAINS="${*:-indoor outdoor urban}"
    echo "packing domains: $DOMAINS"
    python pack_dataset.py --src outputs --dst dataset --domains $DOMAINS --workers 16
fi
