#!/bin/bash
# Generate coverage overlays for five representative mapped subjects.
#SBATCH --job-name=fsavgCovGreen
#SBATCH --partition=normal
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis
# Two high-coverage, one typical, and two low-coverage subjects based on the
# existing fsaverage mapping QC summary.
subjects=(D0231 D0330 D0097 D0388 D0163)
subject=${subjects[$((SLURM_ARRAY_TASK_ID/2))]}
hemis=(lh rh); hemi=${hemis[$((SLURM_ARRAY_TASK_ID%2))]}
python -u visualization/create_fsaverage_coverage_map.py \
  --subject-id "$subject" --hemisphere "$hemi" \
  --mapping-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage \
  --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_coverage_visualization_green_conflict/$subject
