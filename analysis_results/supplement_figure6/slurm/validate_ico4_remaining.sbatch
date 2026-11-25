#!/bin/bash
#SBATCH --job-name=fig6_i4_val15
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=analysis_results/figure6/ico4_warped_m2_186/logs/validate_remaining_%j.out
#SBATCH --error=analysis_results/figure6/ico4_warped_m2_186/logs/validate_remaining_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
RESULT_ROOT="${REPO_ROOT}/analysis_results/figure6/ico4_warped_m2_186"
FEATURE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/features"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fig6_i4_val15_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fig6_i4_val15_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/validate_ico4_stage.py" \
  --result-root "${RESULT_ROOT}" \
  --subject-order "${FEATURE_ROOT}/subject_order.csv" \
  --ico4-md "${FEATURE_ROOT}/matrices/X_ico4_md.npy" \
  --stage remaining \
  --depth-indices 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19 \
  --output "${RESULT_ROOT}/remaining/remaining_validation.json"
