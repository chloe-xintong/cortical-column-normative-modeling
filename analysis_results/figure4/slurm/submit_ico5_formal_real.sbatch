#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5Real
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
MANIFEST="${MANIFEST:?Set MANIFEST}"
PAIR_FILE="${PAIR_FILE:?Set PAIR_FILE}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_ROOT"
export PYTHONPATH="$PROJECT_DIR/analysis_results/figure4/scripts:$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/compute_ico5_formal_real.py" \
  --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
  --pair-file "$PAIR_FILE" --output-root "$OUTPUT_ROOT"
