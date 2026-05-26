#!/usr/bin/env bash
#SBATCH --job-name=panoinf-optc-teardown
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00
#SBATCH -o logs/optc_teardown_%j.out
#
# Final cleanup: delete the temporary validation repo (after you've signed
# off on the test run and finished the production re-encode). Optional.
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

: "${HF_TOKEN:?HF_TOKEN must be set (write scope on the validation repo)}"
: "${VALIDATION_REPO:=vulus98/panoinfinigen-option-c-validation}"

echo "$(date) deleting $VALIDATION_REPO"
python -c "
import os
from huggingface_hub import delete_repo
delete_repo('${VALIDATION_REPO}', repo_type='dataset', token=os.environ['HF_TOKEN'])
print('deleted ${VALIDATION_REPO}')
"
echo "$(date) teardown complete"
