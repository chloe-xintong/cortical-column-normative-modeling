#!/usr/bin/env bash
# Submit the pre-specified ico4 geodesic Voronoi patch-mean supplement.

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
SUPPLEMENT_ROOT="$OUTPUT_ROOT/supplement/patch_test"
SLURM_DIR="$PROJECT_DIR/analysis_results/figure5/supplement/patch_test/slurm"
LOG_DIR="$PROJECT_DIR/logs/figure5"
mkdir -p "$LOG_DIR"
COMMON_EXPORT="ALL,OUTPUT_ROOT=$OUTPUT_ROOT,SUPPLEMENT_ROOT=$SUPPLEMENT_ROOT"

J1=$(sbatch --parsable --export="$COMMON_EXPORT" "$SLURM_DIR/build_features.sbatch")
J2=$(sbatch --parsable --dependency="afterok:$J1" --export="$COMMON_EXPORT" "$SLURM_DIR/qc_features.sbatch")
J3=$(sbatch --parsable --dependency="afterok:$J2" --export="$COMMON_EXPORT" "$SLURM_DIR/run_lsqr.sbatch")
J4=$(sbatch --parsable --dependency="afterok:$J3" --export="$COMMON_EXPORT" "$SLURM_DIR/compare.sbatch")

printf 'patch_feature_job=%s\npatch_qc_job=%s\npatch_lsqr_job=%s\npatch_comparison_job=%s\n' \
  "$J1" "$J2" "$J3" "$J4"
