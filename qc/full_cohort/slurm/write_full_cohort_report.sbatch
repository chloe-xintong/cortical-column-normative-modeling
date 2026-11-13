#!/usr/bin/env bash
#SBATCH --job-name=QCFinalStatus
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/full_cohort/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/full_cohort/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:?Set MANIFEST}"
OPTION3_ROOT="${OPTION3_ROOT:?Set OPTION3_ROOT}"
FIGURE4_REAL_ROOT="${FIGURE4_REAL_ROOT:?Set FIGURE4_REAL_ROOT}"
FIGURE4_NULL_ROOT="${FIGURE4_NULL_ROOT:?Set FIGURE4_NULL_ROOT}"
STATUS_ROOT="${STATUS_ROOT:?Set STATUS_ROOT}"
METHOD_COMPARISON_MANIFEST="${METHOD_COMPARISON_MANIFEST:?Set METHOD_COMPARISON_MANIFEST}"

mkdir -p "$PROJECT_DIR/logs/full_cohort" "$STATUS_ROOT"

"$PYTHON" "$PROJECT_DIR/qc/full_cohort/scripts/write_full_cohort_report.py" \
    --manifest "$MANIFEST" \
    --option3-summary-dir "$OPTION3_ROOT/cohort" \
    --figure4-real-summary-dir "$FIGURE4_REAL_ROOT/cohort" \
    --figure4-null-summary-dir "$FIGURE4_NULL_ROOT/cohort" \
    --method-comparison-manifest "$METHOD_COMPARISON_MANIFEST" \
    --output-dir "$STATUS_ROOT"

echo "Notebooks were not executed. Run them manually after reviewing this report."
