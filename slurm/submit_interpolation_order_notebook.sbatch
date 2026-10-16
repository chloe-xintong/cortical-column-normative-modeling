#!/bin/bash
#SBATCH --job-name=interp-order-nb
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/interpolation_order_notebook_%j.out
#SBATCH --error=logs/interpolation_order_notebook_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
RESULTS_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results
NOTEBOOK_OUTPUT_DIR="$RESULTS_ROOT/interpolation_order_comparison_fsaverage_python/notebook_validation"
mkdir -p "$NOTEBOOK_OUTPUT_DIR"

echo "START $(date -Is)"
echo "HOST $(hostname)"

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output-dir "$NOTEBOOK_OUTPUT_DIR" \
    --output summary_notebook_executed.ipynb \
    surface_resampling/template_registration_comparison/notebooks/compare_md_interpolation_order_fsaverage.ipynb

echo "COMPLETED $(date -Is)"
