#!/usr/bin/env bash
#SBATCH --job-name=panoinf-optc-finalize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
#SBATCH -o logs/optc_finalize_%j.out
#
# Phase 3 — rewrites the dataset card on prs-eth/PanoInfinigen *after* every
# shard has been re-encoded and uploaded.
#
# Drops the `dataset_info` block from the YAML (forces HF to auto-derive
# features from the new parquet shards — this is what worked around the
# CastError we hit on ZuriPano), and rewrites the Data Structure / How to Use
# sections to match the new column layout.
#
# Run this once, after `option_c_reencode.sh` finishes successfully.
#
set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen
module load eth_proxy 2>/dev/null || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
if [[ ! -d scripts/option_c ]]; then
    echo "ERROR: scripts/option_c not found under $REPO_ROOT — call \`sbatch\` from the repo root." >&2
    exit 1
fi

: "${HF_TOKEN:?HF_TOKEN must be set (write scope on prs-eth/PanoInfinigen)}"
: "${TARGET_REPO:=prs-eth/PanoInfinigen}"

echo "$(date) rewriting dataset card on $TARGET_REPO"
python -m scripts.option_c.update_readme --repo "$TARGET_REPO"
echo "$(date) finalize complete"
