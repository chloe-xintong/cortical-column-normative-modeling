#!/bin/bash
#SBATCH --job-name=native-repro-D0050
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/native_repro_D0050_%j.out
#SBATCH --error=logs/native_repro_D0050_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

python -u surface_resampling/validate_native_profile_reproduction.py \
    --subject-id D0050 \
    --profiles-root \
    /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
    --metric-image \
    /mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi/D0050/D0050_md.nii.gz \
    --n-samples 1000 \
    --seed 20261002 \
    --output \
    /mnt/newStor/paros/paros_WORK/xintong/results/option3_sampling_qc/D0050_native_profile_reproduction.json

echo "NATIVE PROFILE REPRODUCTION COMPLETED subject=D0050"
