#!/usr/bin/env bash
#SBATCH --job-name=F4RealSummary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:?Set MANIFEST}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_ROOT/cohort"

"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/fit_correlation_lengths.py" \
    --manifest "$MANIFEST" \
    --results-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/cohort"
