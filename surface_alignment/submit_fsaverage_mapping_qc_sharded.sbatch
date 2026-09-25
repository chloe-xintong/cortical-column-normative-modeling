#!/bin/bash
#SBATCH --job-name=fsavgmapqc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis

MANIFEST=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/surface_alignment/manifests/fsaverage_mapping_qc_shard_$SLURM_ARRAY_TASK_ID.txt
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_mapping_qc/shards/shard_$SLURM_ARRAY_TASK_ID
python -u surface_alignment/fsaverage_mapping_qc.py \
  --subjects-file "$MANIFEST" \
  --mapping-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage \
  --profiles-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
  --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage \
  --cross-subject-coverage-dir /mnt/newStor/paros/paros_WORK/xintong/results/cross_subject_fsaverage_qc \
  --output-dir "$OUT" \
  --progress-every 100
