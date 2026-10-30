#!/usr/bin/env bash
#SBATCH --job-name=F4DMatch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
mkdir -p "$PROJECT_DIR/logs/figure4" "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
export PYTHONPATH="$PROJECT_DIR/analysis_results/figure4/scripts:$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/generate_boundary_matched_pairs.py" \
  --manifest "${MANIFEST:?Set MANIFEST}" --index "$SLURM_ARRAY_TASK_ID" \
  --candidate-dir "${CANDIDATE_DIR:?Set CANDIDATE_DIR}" \
  --output-root "$OUTPUT_ROOT" --k "${K:-15}" \
  --caliper-mm "${CALIPER_MM:-0.1}"
