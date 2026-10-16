#!/bin/bash
#SBATCH --job-name=o3-cohort-qc
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/o3_cohort_qc_%j.out
#SBATCH --error=logs/o3_cohort_qc_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

mkdir -p logs

python -u surface_resampling/analysis/py/cohort_option3_qc.py \
    --manifest manifests/method_comparison_common_30_subjects.csv \
    --o3-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles \
    --freesurfer-subjects-dir /home/apps/freesurfer/subjects \
    --output-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_cohort_qc_fsaverage

python -u surface_resampling/analysis/py/make_o3_registration_overlays.py \
    --qc-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_cohort_qc_fsaverage \
    --o3-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles \
    --output-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_registration_qc_overlays

echo "OPTION3 COHORT QC COMPLETED"
echo "OUTPUT_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/o3_cohort_qc_fsaverage"
echo "REGISTRATION_OVERLAY_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/o3_registration_qc_overlays"
