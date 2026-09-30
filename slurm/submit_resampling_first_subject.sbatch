#!/bin/bash
#SBATCH --job-name=resampling-first
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/resampling_first_%j.out
#SBATCH --error=logs/resampling_first_%j.err

set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mkdir -p logs

export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="${TMPDIR:-/tmp}/surface_resampling_subjects"
mkdir -p "$SUBJECTS_DIR"
ln -sfn /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/D0050 "$SUBJECTS_DIR/D0050"
ln -sfn /home/apps/freesurfer/subjects/fsaverage "$SUBJECTS_DIR/fsaverage"
ln -sfn /home/apps/freesurfer/subjects/fsaverage6 "$SUBJECTS_DIR/fsaverage6"

ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/surface_resampling/first_subject_D0050
MD=/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi/D0050/D0050_md.nii.gz
REG=/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis/D0050/registration/D0050_DWI2T1.dat
DEPTHS=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00

TOTAL=84
DONE=0
for template in fsaverage6 fsaverage; do
    for hemi in lh rh; do
        OUT="$ROOT/$template/$hemi"
        mkdir -p "$OUT"
        IFS=',' read -ra DEPTH_ARRAY <<< "$DEPTHS"
        for depth in "${DEPTH_ARRAY[@]}"; do
            python -u surface_resampling/run_vol2surf.py \
                --subject-id D0050 --srcsubject D0050 \
                --subjects-dir "$SUBJECTS_DIR" --metric-image "$MD" \
                --reg "$REG" --template "$template" --hemisphere "$hemi" \
                --depths "$depth" --output-dir "$OUT"
            DONE=$((DONE + 1))
            echo "PROGRESS $DONE/$TOTAL template=$template hemi=$hemi depth=$depth"
        done
        python -u surface_resampling/stack_depth_profiles.py \
            --input-dir "$OUT" --subject-id D0050 --template "$template" \
            --hemisphere "$hemi" --depths "$DEPTHS" \
            --output "$OUT/D0050_${hemi}_md_profiles.npz"
    done
done

python -u qc/surface_resampling/validate_outputs.py \
    --profiles "$ROOT/fsaverage6/lh/D0050_lh_md_profiles.npz" \
    "$ROOT/fsaverage6/rh/D0050_rh_md_profiles.npz" \
    "$ROOT/fsaverage/lh/D0050_lh_md_profiles.npz" \
    "$ROOT/fsaverage/rh/D0050_rh_md_profiles.npz" \
    --template fsaverage6 --output-dir "$ROOT/qc"

echo "FIRST SUBJECT END-TO-END TEST COMPLETED"
