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

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

: "${HF_TOKEN:?HF_TOKEN must be set (write scope on the validation repo)}"
: "${VALIDATION_REPO:=vulus98/panoinfinigen-option-c-validation}"

echo "$(date) deleting $VALIDATION_REPO"
python -c "
import os
from huggingface_hub import delete_repo, login
login(os.environ['HF_TOKEN'])
delete_repo('${VALIDATION_REPO}', repo_type='dataset')
print('deleted ${VALIDATION_REPO}')
"
echo "$(date) teardown complete"
