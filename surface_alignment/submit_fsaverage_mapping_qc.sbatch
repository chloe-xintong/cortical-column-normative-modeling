#!/bin/bash
#SBATCH --job-name=fsavgmapqc
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
source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis

python -u surface_alignment/fsaverage_mapping_qc.py \
  --mapping-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage \
  --profiles-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
  --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage \
  --cross-subject-coverage-dir /mnt/newStor/paros/paros_WORK/xintong/results/cross_subject_fsaverage_qc \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_mapping_qc \
  --progress-every 100
