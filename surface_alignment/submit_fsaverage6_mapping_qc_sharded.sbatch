#!/bin/bash
# Second-stage subject-region QC only; first-stage frequency outputs are reused.
#SBATCH --job-name=fsavg6qc2
#SBATCH --partition=normal
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage6_mapping_qc_shards/shard_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OUT"
python -u surface_alignment/fsaverage_mapping_qc.py --mapping-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 --profiles-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles --fsaverage-dir /home/apps/freesurfer/subjects/fsaverage6 --cross-subject-coverage-dir /mnt/newStor/paros/paros_WORK/xintong/results/cross_subject_fsaverage6_qc --output-dir "$OUT" --shard-index "$SLURM_ARRAY_TASK_ID" --num-shards 8 --skip-frequency --progress-every 100
