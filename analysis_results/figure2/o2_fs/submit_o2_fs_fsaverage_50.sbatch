#!/bin/bash
#SBATCH --job-name=o2-fs-50
#SBATCH --partition=normal
#SBATCH --array=0-49%4
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/o2_fs_50_%A_%a.out
#SBATCH --error=logs/o2_fs_50_%A_%a.err

set -euo pipefail

PROJECT_DIR=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
SUBJECT_MANIFEST="$PROJECT_DIR/manifests/o2_fs_50_fsaverage_subjects.txt"
FREESURFER_ROOT=/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer
DWI_ROOT=/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi
RESULTS_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/qc/o2_fs

cd "$PROJECT_DIR"
mkdir -p logs

mapfile -t SUBJECTS < <(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$SUBJECT_MANIFEST")
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="${TMPDIR:-/tmp}/o2_fs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$SUBJECTS_DIR"
ln -sfn "$FREESURFER_ROOT/$SUBJECT" "$SUBJECTS_DIR/$SUBJECT"
ln -sfn /home/apps/freesurfer/subjects/fsaverage "$SUBJECTS_DIR/fsaverage"

MD="$DWI_ROOT/$SUBJECT/${SUBJECT}_md.nii.gz"
REG="/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns/$SUBJECT/registration/${SUBJECT}_DWI2T1.dat"
OUTPUT_DIR="$RESULTS_ROOT/$SUBJECT/fsaverage"
DEPTHS=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00

test -s "$MD"
test -s "$REG"
mkdir -p "$OUTPUT_DIR/lh" "$OUTPUT_DIR/rh"

echo "PROGRESS subject=$SUBJECT step=inputs_validated"

for hemisphere in lh rh; do
    echo "PROGRESS subject=$SUBJECT hemisphere=$hemisphere step=0/21_start"
    "$PYTHON" -u analysis_results/figure2/o2_fs/run_vol2surf.py \
        --subject-id "$SUBJECT" \
        --srcsubject "$SUBJECT" \
        --subjects-dir "$SUBJECTS_DIR" \
        --metric-image "$MD" \
        --reg "$REG" \
        --template fsaverage \
        --hemisphere "$hemisphere" \
        --depths "$DEPTHS" \
        --output-dir "$OUTPUT_DIR/$hemisphere"
    echo "PROGRESS subject=$SUBJECT hemisphere=$hemisphere step=21/21_complete"
done

echo "PROGRESS subject=$SUBJECT step=completed"
