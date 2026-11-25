#!/bin/bash
#SBATCH --job-name=fig6_reg186_w
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=analysis_results/figure6/regional_warped_m2_186/logs/warped_%j.out
#SBATCH --error=analysis_results/figure6/regional_warped_m2_186/logs/warped_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
RESULT_ROOT="${REPO_ROOT}/analysis_results/figure6/regional_warped_m2_186"
FEATURE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/features"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fig6_reg186_w_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fig6_reg186_w_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/run_warped_m2.py" \
  --subject-order "${FEATURE_ROOT}/subject_order.csv" \
  --cohort-audit "${RESULT_ROOT}/cohort_audit.csv" \
  --cohort-manifest "${REPO_ROOT}/analysis_results/figure5/cohort_selection/outputs/cohort_manifest.csv" \
  --regional-profile "${FEATURE_ROOT}/matrices/X_regional_profile.npy" \
  --regional-thickness "${FEATURE_ROOT}/matrices/X_regional_thickness.npy" \
  --profile-metadata "${FEATURE_ROOT}/feature_metadata/features_regional_profile.csv" \
  --thickness-metadata "${FEATURE_ROOT}/feature_metadata/features_regional_thickness.csv" \
  --folds "${RESULT_ROOT}/fold_assignments.json" \
  --output-dir "${RESULT_ROOT}/warped_m2" --regions all --n-jobs "${SLURM_CPUS_PER_TASK}"
