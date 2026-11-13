#!/usr/bin/env bash
#SBATCH --job-name=F5Ico4PatchCompare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
SUPPLEMENT_ROOT="${SUPPLEMENT_ROOT:-$OUTPUT_ROOT/supplement/patch_test}"

cd "$PROJECT_DIR"
"$PYTHON" analysis_results/figure5/supplement/patch_test/scripts/compare_ico4_patch_mean.py \
  --baseline-model-root "$OUTPUT_ROOT/models_lsqr" \
  --patch-model-root "$SUPPLEMENT_ROOT/models_lsqr" \
  --output-dir "$SUPPLEMENT_ROOT/analysis" \
  --bootstrap-replicates 10000 \
  --seed 20261028
