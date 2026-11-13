#!/usr/bin/env bash
# Submit ico4 geometry feature, model, cBAG, and paired-inference tasks.

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
SLURM_DIR="$PROJECT_DIR/analysis_results/figure5/slurm"
mkdir -p "$OUTPUT_ROOT/ico4_geometry_analysis" "$PROJECT_DIR/logs/figure5"
COMMON_EXPORT="ALL,OUTPUT_ROOT=$OUTPUT_ROOT"
J1=$(sbatch --parsable --export="$COMMON_EXPORT" "$SLURM_DIR/build_ico4_geometry.sbatch")
J2=$(sbatch --parsable --dependency="afterok:$J1" --array="0-2%3" --export="$COMMON_EXPORT" "$SLURM_DIR/run_ico4_geometry_models.sbatch")
J3=$(sbatch --parsable --dependency="afterok:$J2" --array="0-2%3" --export="$COMMON_EXPORT" "$SLURM_DIR/compute_ico4_geometry_cbag.sbatch")
J4=$(sbatch --parsable --dependency="afterok:$J3" --export="$COMMON_EXPORT" "$SLURM_DIR/analyze_ico4_geometry.sbatch")
printf 'ico4_geometry_features_job=%s\nico4_geometry_models_job=%s\nico4_geometry_cbag_job=%s\nico4_geometry_analysis_job=%s\n' "$J1" "$J2" "$J3" "$J4"
