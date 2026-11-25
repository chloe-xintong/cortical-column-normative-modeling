#!/bin/bash
#SBATCH --job-name=fig6_full_val
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=analysis_results/figure6/ico4_warped_m2_186/logs/full_validate_%j.out
#SBATCH --error=analysis_results/figure6/ico4_warped_m2_186/logs/full_validate_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fig6_full_validate_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fig6_full_validate_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/validate_summarize_ico4_full_cohort.py" \
  --result-root "${REPO_ROOT}/analysis_results/figure6/ico4_warped_m2_186"
