#!/bin/bash
#SBATCH --job-name=boundaryDis
#SBATCH --partition=normal
#SBATCH --array=0-295%30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mapfile -t subjects < <(find /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 -mindepth 1 -maxdepth 1 -type d | sed 's#.*/##' | sort)
subject=${subjects[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$subject" ]]; then echo "No subject for array index $SLURM_ARRAY_TASK_ID"; exit 0; fi
echo "[$SLURM_ARRAY_TASK_ID/${#subjects[@]}] subject=$subject"
python -u -m native_surface_analysis.boundary_disagreement.analyze_boundary_disagreement --subject "$subject" --template fsaverage6
python -u -m native_surface_analysis.boundary_disagreement.analyze_boundary_disagreement --subject "$subject" --template fsaverage
