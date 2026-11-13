#!/usr/bin/env bash
#SBATCH --job-name=F5MatrixQC
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/matrix_qc"
"$PYTHON" analysis_results/figure5/scripts/qc_matrices.py \
  --feature-dir "$OUTPUT_ROOT/features" \
  --output-dir "$OUTPUT_ROOT/matrix_qc"
