#!/usr/bin/env bash
#SBATCH --job-name=F5Ico4PatchBuild
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
SUPPLEMENT_ROOT="${SUPPLEMENT_ROOT:-$OUTPUT_ROOT/supplement/patch_test}"
PROFILE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns"
TEMPLATE_FS_ROOT="/home/apps/freesurfer/subjects"

cd "$PROJECT_DIR"
"$PYTHON" analysis_results/figure5/supplement/patch_test/scripts/build_ico4_patch_mean_features.py \
  --base-feature-dir "$OUTPUT_ROOT/features" \
  --output-dir "$SUPPLEMENT_ROOT/features" \
  --profile-root "$PROFILE_ROOT" \
  --vertex-order-dir "$OUTPUT_ROOT/template" \
  --subjects-dir "$TEMPLATE_FS_ROOT"
