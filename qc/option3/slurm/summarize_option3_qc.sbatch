#!/usr/bin/env bash
#SBATCH --job-name=O3QCSummary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/option3_qc/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/option3_qc/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:?Set MANIFEST}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

mkdir -p "$PROJECT_DIR/logs/option3_qc" "$OUTPUT_ROOT/cohort"

"$PYTHON" "$PROJECT_DIR/qc/option3/scripts/summarize_option3_qc.py" \
    --manifest "$MANIFEST" \
    --results-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/cohort"
