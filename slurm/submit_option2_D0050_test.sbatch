#!/bin/bash
#SBATCH --job-name=opt2-native-test
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/option2_native_D0050_%j.out
#SBATCH --error=logs/option2_native_D0050_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

python -u surface_resampling/run_option2_native_profiles.py \
    --subject-id D0050 \
    --hemisphere lh \
    --template fsaverage6 \
    --mapping-root \
    /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 \
    --option3-root \
    /mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles \
    --output \
    /mnt/newStor/paros/paros_WORK/xintong/results/o2_column_native_profiles/D0050/fsaverage6/o2_column/D0050_lh_profiles.npz

echo "OPTION2 NATIVE PROFILE TEST COMPLETED subject=D0050 template=fsaverage6 hemisphere=lh"
