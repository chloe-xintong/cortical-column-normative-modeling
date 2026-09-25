#!/bin/bash
# Map all eligible native columns to fsaverage6 without resampling metrics.
# One array task processes one subject; output is kept separate from fsaverage.
#SBATCH --job-name=col2fsavg6
#SBATCH --partition=normal
#SBATCH --array=0-298%30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail
CODE_ROOT=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
MANIFEST=${CODE_ROOT}/surface_alignment/manifests/surface_alignment_eligible_subjects.csv
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | cut -d, -f1 | tr -d '\r')
[[ -n "$SUBJECT" && "$SUBJECT" != "subject_id" ]] || { echo "No subject for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
cd "$CODE_ROOT"
python -u surface_alignment/native_columns_to_fsaverage.py \
  --subject-id "$SUBJECT" \
  --subjects-dir /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer \
  --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage6 \
  --profiles-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
  --output-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 \
  --log-file /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6/logs/${SUBJECT}.log
