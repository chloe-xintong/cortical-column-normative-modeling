#!/usr/bin/env bash
#SBATCH --job-name=F5Ico4PatchQC
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
SUPPLEMENT_ROOT="${SUPPLEMENT_ROOT:-$OUTPUT_ROOT/supplement/patch_test}"

cd "$PROJECT_DIR"
"$PYTHON" analysis_results/figure5/supplement/patch_test/scripts/qc_ico4_patch_mean.py \
  --feature-dir "$SUPPLEMENT_ROOT/features" \
  --vertex-order-dir "$OUTPUT_ROOT/template" \
  --output-dir "$SUPPLEMENT_ROOT/qc"
