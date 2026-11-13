#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5NullSum
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
MANIFEST="${MANIFEST:?Set MANIFEST}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT}"
export PYTHONPATH="$PROJECT_DIR/analysis_results/figure4/scripts:$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/validate_acquisition_sampling_null.py" \
  --condition "0=$RESULT_ROOT" --manifest "$MANIFEST" \
  --output-dir "$RESULT_ROOT/cohort"
