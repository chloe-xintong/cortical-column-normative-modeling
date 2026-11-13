#!/bin/bash
#SBATCH --job-name=curvature-50
#SBATCH --partition=normal
#SBATCH --array=0-49%4
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/curvature_50_%A_%a.out
#SBATCH --error=logs/curvature_50_%A_%a.err

set -euo pipefail

PROJECT_DIR=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
SUBJECT_MANIFEST="$PROJECT_DIR/manifests/o2_fs_50_fsaverage_subjects.txt"
FREESURFER_ROOT=/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer
COLUMN_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns
OUTPUT_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/qc/template_registration_comparison/curvature

cd "$PROJECT_DIR"
mkdir -p logs

mapfile -t SUBJECTS < <(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$SUBJECT_MANIFEST")
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="${TMPDIR:-/tmp}/curvature_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$SUBJECTS_DIR"
ln -sfn "$FREESURFER_ROOT/$SUBJECT" "$SUBJECTS_DIR/$SUBJECT"
ln -sfn /home/apps/freesurfer/subjects/fsaverage "$SUBJECTS_DIR/fsaverage"

echo "PROGRESS subject=$SUBJECT step=started"

for hemisphere in lh rh; do
    "$PYTHON" -u analysis_results/figure2/curvature/run_o2_fs_curvature.py \
        --subject-id "$SUBJECT" \
        --hemisphere "$hemisphere" \
        --subjects-dir "$SUBJECTS_DIR" \
        --output-dir "$OUTPUT_ROOT/o2_fs/$SUBJECT/fsaverage/$hemisphere"

    "$PYTHON" -u analysis_results/figure2/curvature/map_barycentric_curvature.py \
        --subject-id "$SUBJECT" \
        --hemisphere "$hemisphere" \
        --subjects-dir "$FREESURFER_ROOT" \
        --mapping-file "$COLUMN_ROOT/$SUBJECT/columns/${SUBJECT}_${hemisphere}_columns.npz" \
        --output "$OUTPUT_ROOT/shared_barycentric/$SUBJECT/fsaverage/${SUBJECT}_${hemisphere}_curvature.npz"

    echo "PROGRESS subject=$SUBJECT hemisphere=$hemisphere step=complete"
done

echo "PROGRESS subject=$SUBJECT step=completed"
