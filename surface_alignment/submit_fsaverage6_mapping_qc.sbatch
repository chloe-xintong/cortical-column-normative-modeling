#!/bin/bash
# QC for completed fsaverage6 native-column mappings.
#SBATCH --job-name=fsavg6qc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%j.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
python -u surface_alignment/cross_subject_fsaverage_coverage.py \
  --input-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/cross_subject_fsaverage6_qc \
  --progress-every 100
python -u surface_alignment/fsaverage_mapping_qc.py \
  --mapping-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 \
  --profiles-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
  --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage6 \
  --cross-subject-coverage-dir /mnt/newStor/paros/paros_WORK/xintong/results/cross_subject_fsaverage6_qc \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/fsaverage6_mapping_qc \
  --progress-every 100
