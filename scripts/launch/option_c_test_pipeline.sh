#!/usr/bin/env bash
#SBATCH --job-name=panoinf-optc-test
#SBATCH --array=0-3%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH -o logs/optc_test_%A_%a.out
#
# End-to-end small-scale test on cluster.
#
# Processes only urban/val (18 shards) and uploads under a separate test
# repo, leaving the production PanoInfinigen dataset untouched. Verify the
# viewer there before kicking off the real backup+reencode jobs.
#
# Run this AFTER the test repo exists. Recommended workflow:
#
#   # 1. Create the test repo (once, manually, public so the viewer works):
#   python -c "from huggingface_hub import HfApi, login; import os; \
#       login(os.environ['HF_TOKEN']); \
#       HfApi().create_repo('vulus98/panoinfinigen-option-c-validation', \
#                           repo_type='dataset', private=False, exist_ok=True)"
#
#   # 2. Push a stub README that registers the urban-val config:
#   python -m scripts.option_c.update_readme \
#       --repo vulus98/panoinfinigen-option-c-validation \
#       --input scripts/option_c/test_repo_stub_README.md
#
#   # 3. sbatch this file. With --array=0-3%4 it processes urban/val in 4
#   #    parallel slices (each handles ~5 shards).
#
set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen
module load eth_proxy 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

: "${NUM_TASKS:=4}"
: "${MANIFEST:=/cluster/scratch/$USER/panoinfinigen_manifest.json}"
: "${BACKUP_DIR:=/cluster/scratch/$USER/panoinfinigen_backup_test}"
: "${NEW_DIR:=/cluster/scratch/$USER/panoinfinigen_new_test}"
: "${TARGET_REPO:=vulus98/panoinfinigen-option-c-validation}"
: "${TARGET_PREFIX:=data/}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

# Auto-generate manifest on task 0; others wait.
if [[ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 && ! -f "$MANIFEST" ]]; then
    python -m scripts.option_c.manifest --out "$MANIFEST"
else
    until [[ -f "$MANIFEST" ]]; do sleep 5; done
fi

echo "$(date) test task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS} starting (urban/val only)"
python -m scripts.option_c.process_batch \
    --manifest "$MANIFEST" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${NUM_TASKS}" \
    --config urban --split val \
    --backup-dir "$BACKUP_DIR" \
    --new-dir "$NEW_DIR" \
    --target-repo "$TARGET_REPO" \
    --target-prefix "$TARGET_PREFIX" \
    --cpus "${SLURM_CPUS_PER_TASK:-8}" \
    --do backup --do reencode --do upload
echo "$(date) test task ${SLURM_ARRAY_TASK_ID} done"
