#!/bin/bash
#SBATCH --job-name=opt3-ray-D0007
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/option3_ray_D0007_%j.out
#SBATCH --error=logs/option3_ray_D0007_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

export SUBJECTS_DIR="${TMPDIR:-/tmp}/option3_ray_subjects"
mkdir -p "$SUBJECTS_DIR"
ln -sfn \
    /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/D0007 \
    "$SUBJECTS_DIR/D0007"
ln -sfn /home/apps/freesurfer/subjects/fsaverage "$SUBJECTS_DIR/fsaverage"
ln -sfn /home/apps/freesurfer/subjects/fsaverage6 "$SUBJECTS_DIR/fsaverage6"

md_path=/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi/D0007/D0007_md.nii.gz
reg_path=/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis/D0007/registration/D0007_DWI2T1.dat
output_root=/mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles
depths=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00

completed=0
total=4
echo "PROGRESS $completed/$total subject=D0007"

for template in fsaverage6 fsaverage; do
    for hemi in lh rh; do
        python -u surface_resampling/run_option3_target_driven.py \
            --subject-id D0007 \
            --subjects-dir "$SUBJECTS_DIR" \
            --metric-image "$md_path" \
            --reg "$reg_path" \
            --template "$template" \
            --hemisphere "$hemi" \
            --depths "$depths" \
            --output \
            "$output_root/D0007/$template/o3_coordinate/D0007_${hemi}_profiles.npz"
        completed=$((completed + 1))
        echo "PROGRESS $completed/$total subject=D0007 template=$template hemisphere=$hemi"
    done
done

echo "OPTION3 CORRECTED SUBJECT COMPLETED subject=D0007"
