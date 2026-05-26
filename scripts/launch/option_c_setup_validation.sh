#!/usr/bin/env bash
#SBATCH --job-name=panoinf-optc-setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
#SBATCH -o logs/optc_setup_%j.out
#
# One-shot setup for the Option-C validation run.
#
#   1. Generates the shard manifest if it doesn't exist.
#   2. Creates vulus98/panoinfinigen-option-c-validation as a *public*
#      dataset repo on the Hub (no-op if already exists).
#   3. Uploads the stub README that registers the urban-val config.
#
# Run this once before submitting `option_c_test_pipeline.sh`.
#
set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen
module load eth_proxy 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

: "${HF_TOKEN:?HF_TOKEN must be set (write scope on the target test repo)}"
: "${VALIDATION_REPO:=vulus98/panoinfinigen-option-c-validation}"
: "${MANIFEST:=/cluster/scratch/$USER/panoinfinigen_manifest.json}"

# (1) Manifest --------------------------------------------------------------
if [[ ! -f "$MANIFEST" ]]; then
    echo "$(date) generating manifest -> $MANIFEST"
    python -m scripts.option_c.manifest --out "$MANIFEST"
else
    echo "$(date) manifest already at $MANIFEST"
fi

# (2) Create the validation repo (idempotent: exist_ok=True) ----------------
echo "$(date) creating/ensuring $VALIDATION_REPO (public)"
python -c "
import os
from huggingface_hub import HfApi, login
login(os.environ['HF_TOKEN'])
HfApi().create_repo(
    repo_id='${VALIDATION_REPO}',
    repo_type='dataset',
    private=False,
    exist_ok=True,
)
print('repo ready: ${VALIDATION_REPO}')
"

# (3) Push the stub README --------------------------------------------------
echo "$(date) pushing stub README to $VALIDATION_REPO"
python -m scripts.option_c.update_readme \
    --repo "$VALIDATION_REPO" \
    --input scripts/option_c/test_repo_stub_README.md

echo "$(date) setup complete"
