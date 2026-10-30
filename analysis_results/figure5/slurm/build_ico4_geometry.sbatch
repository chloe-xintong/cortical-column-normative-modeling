#!/usr/bin/env bash
#SBATCH --job-name=F5Ico4Geom
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
"$PYTHON" analysis_results/figure5/scripts/build_ico4_geometry_features.py \
  --subject-order "$OUTPUT_ROOT/features/subject_order.csv" \
  --profile-root /mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns \
  --native-freesurfer-root /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer \
  --vertex-order "$OUTPUT_ROOT/template/ico4_vertex_order.csv" \
  --feature-dir "$OUTPUT_ROOT/features"
"$PYTHON" analysis_results/figure5/scripts/qc_ico4_geometry.py \
  --feature-dir "$OUTPUT_ROOT/features" \
  --vertex-order "$OUTPUT_ROOT/template/ico4_vertex_order.csv" \
  --output-dir "$OUTPUT_ROOT/matrix_qc"
