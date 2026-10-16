#!/bin/bash
#SBATCH --job-name=interp-order-py
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/interpolation_order_py_%j.out
#SBATCH --error=logs/interpolation_order_py_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mkdir -p logs

PROJECT_ROOT=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
RESULTS_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results
OUTPUT_ROOT=${OUTPUT_ROOT:-$RESULTS_ROOT/interpolation_order_comparison_fsaverage_python}
TEMPLATE=${TEMPLATE:-fsaverage}

echo "START $(date -Is)"
echo "HOST $(hostname)"
echo "TEMPLATE ${TEMPLATE}"
echo "OUTPUT_ROOT ${OUTPUT_ROOT}"

python -u surface_resampling/analysis/py/compare_o2_column_o3_coordinate.py \
    --manifest "$PROJECT_ROOT/manifests/method_comparison_common_30_subjects.csv" \
    --o2-root "$RESULTS_ROOT/o2_column_native_profiles" \
    --o3-root "$RESULTS_ROOT/o3_coordinate_profiles" \
    --template "$TEMPLATE" \
    --output-root "$OUTPUT_ROOT" \
    --min-subjects 20 \
    --group-sizes 1 50 100 200 400

echo "COMPLETED $(date -Is)"
