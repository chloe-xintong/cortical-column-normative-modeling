#!/bin/bash
#SBATCH --job-name=fsavg6regacc
#SBATCH --partition=normal
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
subjects=(D0231 D0330 D0097 D0388 D0163)
subject=${subjects[$((SLURM_ARRAY_TASK_ID / 2))]}
hemi=$( [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ] && echo lh || echo rh )
python -u visualization/create_fsaverage_region_accuracy_map.py --subject-id "$subject" --hemisphere "$hemi" --mapping-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage6 --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/fsaverage6_region_accuracy_visualization
