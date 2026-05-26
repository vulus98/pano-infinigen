#!/usr/bin/env bash
#SBATCH --job-name=panoinf-reencode
#SBATCH --array=0-63%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH -o logs/reencode_%A_%a.out
#
# Phase 2: re-encode + upload. Run AFTER option_c_backup.sh has finished
# successfully (i.e. every shard mirrored to $BACKUP_DIR).
#
# Bump --array=0-N%K to parallelize harder. Per-task wall time scales with
# (num_shards / NUM_TASKS) * (~80s/shard for indoor/train).
#
# Tunables (override via `sbatch --export=ALL,NUM_TASKS=128 ...`):
#   NUM_TASKS, MANIFEST, BACKUP_DIR, NEW_DIR, TARGET_REPO, TARGET_PREFIX
#
set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen
module load eth_proxy 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

: "${NUM_TASKS:=64}"
: "${MANIFEST:=/cluster/scratch/$USER/panoinfinigen_manifest.json}"
: "${BACKUP_DIR:=/cluster/scratch/$USER/panoinfinigen_backup}"
: "${NEW_DIR:=/cluster/scratch/$USER/panoinfinigen_new}"
: "${TARGET_REPO:=prs-eth/PanoInfinigen}"
: "${TARGET_PREFIX:=data/}"
: "${HF_TOKEN:?HF_TOKEN must be set for upload}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "Run option_c_backup.sh first, or run:"
    echo "  python -m scripts.option_c.manifest --out $MANIFEST"
    exit 1
fi

echo "$(date) task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS} starting"
echo "  target_repo=$TARGET_REPO  target_prefix=$TARGET_PREFIX"
python -m scripts.option_c.process_batch \
    --manifest "$MANIFEST" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${NUM_TASKS}" \
    --backup-dir "$BACKUP_DIR" \
    --new-dir "$NEW_DIR" \
    --target-repo "$TARGET_REPO" \
    --target-prefix "$TARGET_PREFIX" \
    --cpus "${SLURM_CPUS_PER_TASK:-8}" \
    --do reencode --do upload
echo "$(date) task ${SLURM_ARRAY_TASK_ID} done"
