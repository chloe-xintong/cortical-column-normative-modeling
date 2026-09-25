#!/bin/bash
#SBATCH --job-name=col2fsavg
#SBATCH --partition=normal
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
# csv.DictWriter emits CRLF by default. Strip CR explicitly so a subject ID
# read by bash always matches the filesystem directory name on compute nodes.
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | cut -d, -f1 | tr -d '\r')

if [[ -z "$SUBJECT" || "$SUBJECT" == "subject_id" ]]; then
    echo "No subject found for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

cd "$CODE_ROOT"
python -u surface_alignment/native_columns_to_fsaverage.py \
    --subject-id "$SUBJECT" \
    --subjects-dir /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer \
    --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage \
    --profiles-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
    --output-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage \
    --log-file /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage/logs/${SUBJECT}.log
