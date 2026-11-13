#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5Null
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
MANIFEST="${MANIFEST:?Set MANIFEST}"
PAIR_DIR="${PAIR_DIR:?Set PAIR_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
REPLICATES="${REPLICATES:-100}"
mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_ROOT"
export PYTHONPATH="$PROJECT_DIR/analysis_results/figure4/scripts:$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/compute_spatial_null.py" \
  --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
  --pair-dir "$PAIR_DIR" --output-root "$OUTPUT_ROOT" \
  --null-method acquisition_sampling --replicates "$REPLICATES" \
  --psf-fwhm-mm 0 --match-laminar-statistics --no-preliminary \
  --seed 20261020
