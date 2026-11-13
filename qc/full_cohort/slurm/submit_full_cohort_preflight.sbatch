#!/usr/bin/env bash
#SBATCH --job-name=QCPreflight
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/full_cohort/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/full_cohort/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:?Set MANIFEST}"
SUBJECTS_DIR="${SUBJECTS_DIR:?Set SUBJECTS_DIR}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:?Set PREFLIGHT_ROOT}"

mkdir -p "$PROJECT_DIR/logs/full_cohort" "$PREFLIGHT_ROOT"

"$PYTHON" "$PROJECT_DIR/qc/full_cohort/scripts/run_full_cohort_preflight.py" \
    --manifest "$MANIFEST" \
    --subjects-dir "$SUBJECTS_DIR" \
    --output-root "$PREFLIGHT_ROOT"
