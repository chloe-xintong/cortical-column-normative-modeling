#!/usr/bin/env bash
#SBATCH --job-name=F5Ico4Analysis
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
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/ico4_geometry_analysis"
"$PYTHON" analysis_results/figure5/scripts/analyze_ico4_geometry.py \
  --model-root "$OUTPUT_ROOT/models_lsqr" \
  --cbag-root "$OUTPUT_ROOT/cbag_crossfit_lsqr" \
  --output-dir "$OUTPUT_ROOT/ico4_geometry_analysis" \
  --bootstrap-replicates 10000 \
  --seed 20261028
