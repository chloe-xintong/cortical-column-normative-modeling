#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5RegionSum
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
MANIFEST="${MANIFEST:?Set MANIFEST}"
PAIR_FILE="${PAIR_FILE:?Set PAIR_FILE}"
RESULTS_ROOT="${RESULTS_ROOT:?Set RESULTS_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR/analysis_results/figure4/scripts:$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/summarize_ico5_region_real.py" \
  --manifest "$MANIFEST" --pair-file "$PAIR_FILE" \
  --results-root "$RESULTS_ROOT" --output-dir "$OUTPUT_DIR"
