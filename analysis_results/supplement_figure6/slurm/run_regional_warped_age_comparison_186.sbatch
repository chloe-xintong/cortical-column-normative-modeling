#!/bin/bash
#SBATCH --job-name=fig6_reg_age
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=analysis_results/figure6/regional_age_function_comparison_186/logs/run_%j.out
#SBATCH --error=analysis_results/figure6/regional_age_function_comparison_186/logs/run_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
BASE_ROOT="${REPO_ROOT}/analysis_results/figure6/regional_warped_m2_186"
OUT_ROOT="${REPO_ROOT}/analysis_results/figure6/regional_age_function_comparison_186"
FEATURE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/features"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fig6_reg_age_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fig6_reg_age_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "${OUT_ROOT}/logs"

COMMON=(
  --subject-order "${FEATURE_ROOT}/subject_order.csv"
  --cohort-audit "${BASE_ROOT}/cohort_audit.csv"
  --cohort-manifest "${REPO_ROOT}/analysis_results/figure5/cohort_selection/outputs/cohort_manifest.csv"
  --regional-profile "${FEATURE_ROOT}/matrices/X_regional_profile.npy"
  --regional-thickness "${FEATURE_ROOT}/matrices/X_regional_thickness.npy"
  --profile-metadata "${FEATURE_ROOT}/feature_metadata/features_regional_profile.csv"
  --thickness-metadata "${FEATURE_ROOT}/feature_metadata/features_regional_thickness.csv"
  --folds "${BASE_ROOT}/fold_assignments.json"
)

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/run_warped_age_candidates.py" \
  --candidate spline "${COMMON[@]}" --output-dir "${OUT_ROOT}/warped_spline" \
  --regions all --depths all --n-jobs "${SLURM_CPUS_PER_TASK}"

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/run_warped_age_candidates.py" \
  --candidate gompertz "${COMMON[@]}" --output-dir "${OUT_ROOT}/warped_gompertz_pilot" \
  --n-jobs "${SLURM_CPUS_PER_TASK}"

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/summarize_warped_age_comparison.py" \
  --baseline "${BASE_ROOT}/oof_predictions_warped_M2_long.parquet" \
  --baseline-diagnostics "${BASE_ROOT}/warped_m2/shards/model_diagnostics_all.csv" \
  --spline-dir "${OUT_ROOT}/warped_spline" \
  --gompertz-dir "${OUT_ROOT}/warped_gompertz_pilot" \
  --subject-order "${FEATURE_ROOT}/subject_order.csv" \
  --cohort-audit "${BASE_ROOT}/cohort_audit.csv" \
  --regional-profile "${FEATURE_ROOT}/matrices/X_regional_profile.npy" \
  --folds "${BASE_ROOT}/fold_assignments.json" \
  --output-dir "${OUT_ROOT}/comparison" --n-bootstrap 10000 --seed 20261110
