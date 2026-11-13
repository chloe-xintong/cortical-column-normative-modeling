#!/usr/bin/env bash
#SBATCH --job-name=F4Fwhm0Sum
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT}"
REAL_RESULTS_ROOT="${REAL_RESULTS_ROOT:?Set REAL_RESULTS_ROOT}"
MANIFEST="${MANIFEST:?Set MANIFEST}"

"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/validate_acquisition_sampling_null.py" \
  --condition "0=${RESULT_ROOT:?Set RESULT_ROOT}" \
  --manifest "$MANIFEST" \
  --output-dir "${RESULT_ROOT}/cohort"

"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/plot_profile_mae_curves.py" \
  --manifest "$MANIFEST" \
  --real-results-root "$REAL_RESULTS_ROOT" \
  --null-results-root "$RESULT_ROOT" \
  --output-dir "${RESULT_ROOT}/cohort"
