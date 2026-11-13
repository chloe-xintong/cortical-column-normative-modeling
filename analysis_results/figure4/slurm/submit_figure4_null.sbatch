#!/usr/bin/env bash
#SBATCH --job-name=F4Null
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:?Set MANIFEST}"
PAIR_DIR="${PAIR_DIR:?Set PAIR_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_ROOT"
echo "[$(date --iso-8601=seconds)] START Figure4 null index=$SLURM_ARRAY_TASK_ID"

"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/compute_spatial_null.py" \
    --manifest "$MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --pair-dir "$PAIR_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --null-method "${NULL_METHOD:-acquisition_sampling}" \
    --replicates "${N_NULL_REPLICATES:-5}" \
    --psf-fwhm-mm "${PSF_FWHM_MM:-0}" \
    --match-laminar-statistics \
    --seed "${RANDOM_SEED:-20261020}"

echo "[$(date --iso-8601=seconds)] DONE Figure4 null index=$SLURM_ARRAY_TASK_ID"
