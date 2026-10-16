#!/bin/bash
#SBATCH --job-name=o2-fs-d0007
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/o2_fs_d0007_%j.out
#SBATCH --error=logs/o2_fs_d0007_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="${TMPDIR:-/tmp}/o2_fs_d0007_subjects"

mkdir -p "$SUBJECTS_DIR"
ln -sfn \
    /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/D0007 \
    "$SUBJECTS_DIR/D0007"
ln -sfn /home/apps/freesurfer/subjects/fsaverage "$SUBJECTS_DIR/fsaverage"

MD=/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi/D0007/D0007_md.nii.gz
REG=/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis/D0007/registration/D0007_DWI2T1.dat
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/surface_resampling/D0007/fsaverage
DEPTHS=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00

mkdir -p "$OUT/lh" "$OUT/rh"

for hemisphere in lh rh; do
    echo "PROGRESS 0/2 hemisphere=$hemisphere"
    python -u surface_resampling/run_vol2surf.py \
        --subject-id D0007 \
        --srcsubject D0007 \
        --subjects-dir "$SUBJECTS_DIR" \
        --metric-image "$MD" \
        --reg "$REG" \
        --template fsaverage \
        --hemisphere "$hemisphere" \
        --depths "$DEPTHS" \
        --output-dir "$OUT/$hemisphere"
    echo "PROGRESS 1/2 hemisphere=$hemisphere"
done

echo "O2-FS D0007 fsaverage COMPLETED"
