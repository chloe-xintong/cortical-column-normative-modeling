#!/bin/bash
#SBATCH --job-name=method-regional
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/method_regional_%j.out
#SBATCH --error=logs/method_regional_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"

mkdir -p logs

OUTPUT_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/method_comparison_regional_profiles_fsaverage_common30

python -u qc/archive/template_registration_comparison/regional_profiles/py/compare_method_regional_profiles.py \
    --manifest manifests/method_comparison_common_30_subjects.csv \
    --native-profile-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
    --o2-root /mnt/newStor/paros/paros_WORK/xintong/results/surface_resampling \
    --o2-column-root /mnt/newStor/paros/paros_WORK/xintong/results/o2_column_native_profiles \
    --o3-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles \
    --freesurfer-subjects-dir /home/apps/freesurfer/subjects \
    --output-root "$OUTPUT_ROOT"

echo "FOUR-METHOD REGIONAL COMPARISON COMPLETED"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
