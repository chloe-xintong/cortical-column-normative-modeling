#!/bin/bash
#SBATCH --job-name=opt3-ray-rh
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/option3_ray_D0050_rh_%j.out
#SBATCH --error=logs/option3_ray_D0050_rh_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="${TMPDIR:-/tmp}/option3_ray_subjects"

mkdir -p "$SUBJECTS_DIR"
ln -sfn \
    /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/D0050 \
    "$SUBJECTS_DIR/D0050"
ln -sfn /home/apps/freesurfer/subjects/fsaverage6 "$SUBJECTS_DIR/fsaverage6"

MD=/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi/D0050/D0050_md.nii.gz
REG=/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis/D0050/registration/D0050_DWI2T1.dat
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/template_method_test/D0050
DEPTHS=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00

echo "PROGRESS 0/1 template=fsaverage6 hemisphere=rh"
python -u surface_resampling/run_option3_target_driven.py \
    --subject-id D0050 \
    --subjects-dir "$SUBJECTS_DIR" \
    --metric-image "$MD" \
    --reg "$REG" \
    --template fsaverage6 \
    --hemisphere rh \
    --depths "$DEPTHS" \
    --output "$OUT/fsaverage6/o3_coordinate/D0050_rh_profiles.npz"
echo "PROGRESS 1/1 template=fsaverage6 hemisphere=rh"
echo "OPTION3 RAY-TRIANGLE RIGHT-HEMISPHERE TEST COMPLETED"
