#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
SLURM_DIR="$PROJECT_DIR/analysis_results/figure5/slurm"
mkdir -p "$PROJECT_DIR/logs/figure5"

JOB_ID=$(sbatch --parsable \
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT" \
  "$SLURM_DIR/run_ico4_column_mean_md.sbatch")
printf 'ico4_column_mean_md_job=%s\n' "$JOB_ID"
