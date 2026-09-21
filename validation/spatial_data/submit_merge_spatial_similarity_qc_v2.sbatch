#!/bin/bash
#SBATCH --job-name=spatialqcmerge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%j.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
python -u validation/spatial_data/merge_spatial_similarity_qc_v2.py \
  --shards-root /mnt/newStor/paros/paros_WORK/xintong/results/spatial_similarity_qc_v2/shards \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/spatial_similarity_qc_v2 \
  --expected-shards 4
