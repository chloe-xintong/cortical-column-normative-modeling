#!/usr/bin/env bash
# Submit leakage-resistant foldwise cBAG calculations from completed LSQR models.

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
SLURM_DIR="$PROJECT_DIR/analysis_results/figure5/slurm"
mkdir -p "$OUTPUT_ROOT/cbag_crossfit_lsqr" "$PROJECT_DIR/logs/figure5"
COMMON_EXPORT="ALL,OUTPUT_ROOT=$OUTPUT_ROOT"
J1=$(sbatch --parsable --array="0-7%4" --export="$COMMON_EXPORT" "$SLURM_DIR/compute_crossfit_cbag_array.sbatch")
J2=$(sbatch --parsable --export="$COMMON_EXPORT" "$SLURM_DIR/compute_crossfit_cbag_ico5.sbatch")
J3=$(sbatch --parsable --dependency="afterok:$J1:$J2" --export="$COMMON_EXPORT" "$SLURM_DIR/summarize_crossfit_cbag.sbatch")
printf 'crossfit_cbag_array_job=%s\ncrossfit_cbag_ico5_job=%s\ncrossfit_cbag_summary_job=%s\n' "$J1" "$J2" "$J3"
