#!/usr/bin/env bash
#SBATCH --job-name=panoinf-backup
#SBATCH --array=0-31%32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH -o logs/backup_%A_%a.out
#
# Phase 1: mirror every PanoInfinigen parquet shard to local scratch.
#
# Tunables (override via `sbatch --export=ALL,NUM_TASKS=64 ...`):
#   NUM_TASKS    : total parallel array tasks (must match `--array=...`)
#   MANIFEST     : path to the precomputed shard manifest
#   BACKUP_DIR   : local target dir
#
set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen
module load eth_proxy 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

: "${NUM_TASKS:=32}"
: "${MANIFEST:=/cluster/scratch/$USER/panoinfinigen_manifest.json}"
: "${BACKUP_DIR:=/cluster/scratch/$USER/panoinfinigen_backup}"

# Generate manifest on first task only; others wait for it to appear.
if [[ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 && ! -f "$MANIFEST" ]]; then
    echo "[task 0] generating manifest -> $MANIFEST"
    python -m scripts.option_c.manifest --out "$MANIFEST"
else
    until [[ -f "$MANIFEST" ]]; do sleep 5; done
fi

echo "$(date) task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS} starting"
python -m scripts.option_c.process_batch \
    --manifest "$MANIFEST" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${NUM_TASKS}" \
    --backup-dir "$BACKUP_DIR" \
    --do backup
echo "$(date) task ${SLURM_ARRAY_TASK_ID} done"
