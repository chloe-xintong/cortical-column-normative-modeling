#!/usr/bin/env bash
#SBATCH --job-name=F4WithinWB
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
python "$PROJECT_DIR/analysis_results/figure4/scripts/summarize_within_region_whole_cortex.py" \
  --manifest "${MANIFEST:?Set MANIFEST}" \
  --results-root "${RESULTS_ROOT:?Set RESULTS_ROOT}" \
  --pair-table "${PAIR_TABLE:?Set PAIR_TABLE}" \
  --output-dir "${OUTPUT_DIR:?Set OUTPUT_DIR}" \
  --figure-dir "${FIGURE_DIR:?Set FIGURE_DIR}" \
  --repo-root "$PROJECT_DIR"
