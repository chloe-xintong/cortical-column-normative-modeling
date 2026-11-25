#!/bin/bash
#SBATCH --job-name=fig6_reg186_all
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=analysis_results/figure6/regional_warped_m2_186/logs/summarize_all_%j.out
#SBATCH --error=analysis_results/figure6/regional_warped_m2_186/logs/summarize_all_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
RESULT_ROOT="${REPO_ROOT}/analysis_results/figure6/regional_warped_m2_186"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fig6_reg186_all_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fig6_reg186_all_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/summarize_regional_models_186.py" \
  --result-root "${RESULT_ROOT}" --n-bootstrap 10000 --seed 20261108
