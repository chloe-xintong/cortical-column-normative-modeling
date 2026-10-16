#!/bin/bash
#SBATCH --job-name=o2col-o3coord
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/method_comparison_o2col_o3coord_%j.out
#SBATCH --error=logs/method_comparison_o2col_o3coord_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

echo "START $(date -Is)"
echo "HOST $(hostname)"

template="${TEMPLATE:-fsaverage6}"
output_root="${OUTPUT_ROOT:-/mnt/newStor/paros/paros_WORK/xintong/results/method_comparison_o2_column_o3_coordinate_fsaverage6}"

python -u surface_resampling/analysis/py/compare_o2_column_o3_coordinate.py \
    --manifest manifests/method_comparison_common_30_subjects.csv \
    --o2-root /mnt/newStor/paros/paros_WORK/xintong/results/o2_column_native_profiles \
    --o3-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles \
    --template "$template" \
    --output-root "$output_root"

echo "COMPLETED $(date -Is)"
