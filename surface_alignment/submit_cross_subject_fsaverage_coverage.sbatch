#!/bin/bash
#SBATCH --job-name=fsavgcover
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%j.err

set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

# This job is submitted with --dependency=afterany:<mapping-array-job>.
# The Python program skips incomplete mapping outputs and writes their reasons,
# so a few failed mapping tasks do not prevent cohort-level QC from running.
python -u surface_alignment/cross_subject_fsaverage_coverage.py \
    --input-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage \
    --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/cross_subject_fsaverage_qc \
    --progress-every 100
