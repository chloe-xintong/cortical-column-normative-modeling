#!/bin/bash
#SBATCH --job-name=fig6_reg187_g
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=analysis_results/figure6/regional_pilot_exclude_D0271/logs/gaussian_%j.out
#SBATCH --error=analysis_results/figure6/regional_pilot_exclude_D0271/logs/gaussian_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
RESULT_ROOT="${REPO_ROOT}/analysis_results/figure6/regional_pilot_exclude_D0271"
FIGURE5_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"

export XDG_CACHE_HOME="${TMPDIR:-/tmp}/figure6_reg187_g_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/figure6_reg187_g_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/run_gaussian_models.py" \
  --subject-order "${FIGURE5_ROOT}/features/subject_order.csv" \
  --cohort-audit "${RESULT_ROOT}/cohort_audit.csv" \
  --cohort-manifest "${REPO_ROOT}/analysis_results/figure5/cohort_selection/outputs/cohort_manifest.csv" \
  --regional-profile "${FIGURE5_ROOT}/features/matrices/X_regional_profile.npy" \
  --regional-thickness "${FIGURE5_ROOT}/features/matrices/X_regional_thickness.npy" \
  --profile-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_profile.csv" \
  --thickness-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_thickness.csv" \
  --folds "${RESULT_ROOT}/fold_assignments.json" \
  --output-dir "${RESULT_ROOT}" --regions all --n-jobs "${SLURM_CPUS_PER_TASK}"
