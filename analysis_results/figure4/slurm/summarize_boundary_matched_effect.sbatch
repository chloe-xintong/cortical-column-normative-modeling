#!/usr/bin/env bash
#SBATCH --job-name=F4DSummary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
mkdir -p "$PROJECT_DIR/logs/figure4" "${OUTPUT_DIR:?Set OUTPUT_DIR}"
export PYTHONPATH="$PROJECT_DIR/analysis_results/figure4/scripts:$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/summarize_boundary_matched_effect.py" \
  --manifest "${MANIFEST:?Set MANIFEST}" \
  --pair-root "${PAIR_ROOT:?Set PAIR_ROOT}" \
  --effect-root "${EFFECT_ROOT:?Set EFFECT_ROOT}" \
  --output-dir "$OUTPUT_DIR" --repo-root "$PROJECT_DIR"
