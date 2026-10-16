#!/bin/bash
#SBATCH --job-name=opt2-native-D0007
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/option2_native_D0007_%j.out
#SBATCH --error=logs/option2_native_D0007_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

mapping_root=/mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6
option3_root=/mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles
output_root=/mnt/newStor/paros/paros_WORK/xintong/results/o2_column_native_profiles
completed=0
total=4

echo "PROGRESS $completed/$total subject=D0007"
for template in fsaverage6 fsaverage; do
    for hemi in lh rh; do
        python -u surface_resampling/run_option2_native_profiles.py \
            --subject-id D0007 \
            --hemisphere "$hemi" \
            --template "$template" \
            --mapping-root "$mapping_root" \
            --option3-root "$option3_root" \
            --output \
            "$output_root/D0007/$template/o2_column/D0007_${hemi}_profiles.npz"
        completed=$((completed + 1))
        echo "PROGRESS $completed/$total subject=D0007 template=$template hemisphere=$hemi"
    done
done

echo "OPTION2 NATIVE PROFILE SUBJECT COMPLETED subject=D0007"
