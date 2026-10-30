#!/usr/bin/env bash
#SBATCH --job-name=O3QC
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/option3_qc/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/option3_qc/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:?Set MANIFEST to the full-cohort CSV}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for O3 QC caches}"
SUBJECTS_DIR="${SUBJECTS_DIR:?Set SUBJECTS_DIR}"

mkdir -p "$PROJECT_DIR/logs/option3_qc" "$OUTPUT_ROOT"

echo "[$(date --iso-8601=seconds)] START O3 QC index=$SLURM_ARRAY_TASK_ID node=$(hostname)"

"$PYTHON" "$PROJECT_DIR/qc/option3/scripts/compute_option3_qc.py" \
    --manifest "$MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output-root "$OUTPUT_ROOT" \
    --subjects-dir "$SUBJECTS_DIR" \
    --template fsaverage

echo "[$(date --iso-8601=seconds)] DONE O3 QC index=$SLURM_ARRAY_TASK_ID"
