#!/usr/bin/env bash
#SBATCH --job-name=F5LSQRAnalysis
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
LSQR_MODEL_ROOT="${LSQR_MODEL_ROOT:-$OUTPUT_ROOT/models_lsqr}"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/summary_lsqr" "$OUTPUT_ROOT/stability_lsqr"
"$PYTHON" analysis_results/figure5/scripts/summarize_models.py \
  --model-root "$LSQR_MODEL_ROOT" \
  --output-dir "$OUTPUT_ROOT/summary_lsqr"
"$PYTHON" analysis_results/figure5/scripts/analyze_lsqr_stability.py \
  --auto-model-root "$OUTPUT_ROOT/models" \
  --lsqr-model-root "$LSQR_MODEL_ROOT" \
  --output-dir "$OUTPUT_ROOT/stability_lsqr" \
  --bootstrap-replicates 10000 \
  --seed 20261028
