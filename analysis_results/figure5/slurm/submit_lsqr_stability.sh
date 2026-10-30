#!/usr/bin/env bash
# Submit fixed-LSQR reruns followed by stability, bootstrap, and cBAG analysis.

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
MODEL_ROOT="$OUTPUT_ROOT/models_lsqr"
SLURM_DIR="$PROJECT_DIR/analysis_results/figure5/slurm"
LOG_DIR="$PROJECT_DIR/logs/figure5"
mkdir -p "$MODEL_ROOT" "$LOG_DIR"
COMMON_EXPORT="ALL,OUTPUT_ROOT=$OUTPUT_ROOT,MODEL_ROOT=$MODEL_ROOT,LSQR_MODEL_ROOT=$MODEL_ROOT,RIDGE_SOLVER=lsqr"

J1=$(sbatch --parsable --array="0-7%4" --export="$COMMON_EXPORT" "$SLURM_DIR/run_ridge_array.sbatch")
J2=$(sbatch --parsable --export="$COMMON_EXPORT" "$SLURM_DIR/run_ridge_ico5.sbatch")
J3=$(sbatch --parsable --dependency="afterok:$J1:$J2" --export="$COMMON_EXPORT" "$SLURM_DIR/analyze_lsqr.sbatch")

printf 'lsqr_ridge_array_job=%s\nlsqr_ico5_job=%s\nlsqr_analysis_job=%s\n' "$J1" "$J2" "$J3"
