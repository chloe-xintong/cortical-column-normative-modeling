#!/usr/bin/env bash
#SBATCH --job-name=F4RelMAESum
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/summarize_relative_mae.py" \
  --manifest "${MANIFEST:?Set MANIFEST}" \
  --results-root "${OUTPUT_ROOT:?Set OUTPUT_ROOT}" \
  --output-dir "$OUTPUT_ROOT/cohort" \
  --repo-root "$PROJECT_DIR"
