#!/usr/bin/env bash
#SBATCH --job-name=F4AWithRelMAE
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/plot_ico5_formal_a_with_relative_mae.py" \
  --real-summary "${REAL_SUMMARY:?Set REAL_SUMMARY}" \
  --relative-summary "${RELATIVE_SUMMARY:?Set RELATIVE_SUMMARY}" \
  --output-dir "${OUTPUT_DIR:?Set OUTPUT_DIR}" \
  --repo-root "$PROJECT_DIR"
