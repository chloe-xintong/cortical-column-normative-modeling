#!/usr/bin/env bash
#SBATCH --job-name=F4DBoundaryGeom
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
mkdir -p "$PROJECT_DIR/logs/figure4" "${OUTPUT_DIR:?Set OUTPUT_DIR}"
export PYTHONPATH="$PROJECT_DIR/analysis_results/figure4/scripts:$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/build_boundary_candidate_neighborhoods.py" \
  --subjects-dir "${SUBJECTS_DIR:?Set SUBJECTS_DIR}" \
  --formal-pair-file "${FORMAL_PAIR_FILE:?Set FORMAL_PAIR_FILE}" \
  --output-dir "$OUTPUT_DIR" --template fsaverage --batch-size 24
