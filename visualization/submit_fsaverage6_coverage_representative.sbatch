#!/bin/bash
# Representative fsaverage6 coverage and region-accuracy visualizations.
#SBATCH --job-name=fsavg6CovRep
#SBATCH --partition=normal
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
subjects=(D0231 D0330 D0097 D0388 D0163)
subject=${subjects[$((SLURM_ARRAY_TASK_ID/2))]}
hemi=$([[ $((SLURM_ARRAY_TASK_ID%2)) -eq 0 ]] && echo lh || echo rh)
maproot=/mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6
out=/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage6_coverage_visualization/$subject
python -u visualization/create_fsaverage_coverage_map.py --subject-id "$subject" --hemisphere "$hemi" --mapping-root "$maproot" --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage6 --output-dir "$out"
python -u visualization/create_fsaverage_region_accuracy_map.py --subject-id "$subject" --hemisphere "$hemi" --mapping-root "$maproot" --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage6 --output-dir "$out"
