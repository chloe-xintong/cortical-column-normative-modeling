#!/usr/bin/env bash
# Submit the Figure 5 feature, QC, fold, model, and summary dependency graph.

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
COHORT="$PROJECT_DIR/analysis_results/figure5/cohort_selection/outputs/figure5_primary_metadata_cn.csv"
SLURM_DIR="$PROJECT_DIR/analysis_results/figure5/slurm"
LOG_DIR="$PROJECT_DIR/logs/figure5"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
COMMON_EXPORT="ALL,OUTPUT_ROOT=$OUTPUT_ROOT,COHORT=$COHORT"

J1=$(sbatch --parsable --export="$COMMON_EXPORT" "$SLURM_DIR/build_features.sbatch")
J2=$(sbatch --parsable --dependency="afterok:$J1" --export="$COMMON_EXPORT" "$SLURM_DIR/qc_matrices.sbatch")
J3=$(sbatch --parsable --dependency="afterok:$J2" --export="$COMMON_EXPORT" "$SLURM_DIR/build_folds.sbatch")
J4=$(sbatch --parsable --dependency="afterok:$J3" --array="0-7%4" --export="$COMMON_EXPORT" "$SLURM_DIR/run_ridge_array.sbatch")
J5=$(sbatch --parsable --dependency="afterok:$J3" --export="$COMMON_EXPORT" "$SLURM_DIR/run_ridge_ico5.sbatch")
J6=$(sbatch --parsable --dependency="afterok:$J4:$J5" --export="$COMMON_EXPORT" "$SLURM_DIR/summarize_models.sbatch")

printf 'feature_job=%s\nmatrix_qc_job=%s\nfold_job=%s\nridge_array_job=%s\nico5_job=%s\nsummary_job=%s\n' \
  "$J1" "$J2" "$J3" "$J4" "$J5" "$J6"
