#!/bin/bash
#SBATCH --job-name=resampling-30
#SBATCH --partition=normal
#SBATCH --array=0-29
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/resampling_30_%A_%a.out
#SBATCH --error=logs/resampling_30_%A_%a.err

set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mkdir -p logs
export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="${TMPDIR:-/tmp}/surface_resampling_subjects"
mkdir -p "$SUBJECTS_DIR"
source_row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" manifests/template_comparison_30_subjects.csv)
subject=$(printf '%s\n' "$source_row" | cut -d, -f1)
ln -sfn "/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/$subject" "$SUBJECTS_DIR/$subject"
ln -sfn /home/apps/freesurfer/subjects/fsaverage "$SUBJECTS_DIR/fsaverage"
ln -sfn /home/apps/freesurfer/subjects/fsaverage6 "$SUBJECTS_DIR/fsaverage6"
python -u pipeline/run_resampling_pilot.py \
  --manifest manifests/template_comparison_30_subjects.csv \
  --row-index "$SLURM_ARRAY_TASK_ID" \
  --output-root /mnt/newStor/paros/paros_WORK/xintong/results/surface_resampling \
  --subjects-dir "$SUBJECTS_DIR"
