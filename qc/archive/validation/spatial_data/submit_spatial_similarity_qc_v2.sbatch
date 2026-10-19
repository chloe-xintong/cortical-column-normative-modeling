#!/bin/bash
#SBATCH --job-name=spatialqcv2
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
python -u validation/spatial_data/spatial_similarity_qc_v2.py \
  --input-root /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles \
  --dwi-root /mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/spatial_similarity_qc_v2/shards/shard_${SLURM_ARRAY_TASK_ID} \
  --k 20 --continuous-pairs 3000 --distance-bins 6 --seed 42 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --num-shards "${SPATIAL_QC_NUM_SHARDS:-4}"
