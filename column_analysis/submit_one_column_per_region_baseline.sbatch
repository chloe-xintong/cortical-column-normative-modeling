#!/bin/bash
#SBATCH --job-name=region-baseline
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/region_baseline_%j.out
#SBATCH --error=logs/region_baseline_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mkdir -p logs

python -u column_analysis/one_column_per_region_baseline.py \
    --analysis-root /mnt/newStor/paros/paros_WORK/xintong/results/native_unmapped_column_results/column_analysis \
    --freesurfer-root /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer \
    --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/native_unmapped_column_results/column_analysis/one_column_per_region_baseline \
    --log-every 25
