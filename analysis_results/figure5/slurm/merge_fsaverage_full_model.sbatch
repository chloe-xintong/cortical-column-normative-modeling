#!/usr/bin/env bash
#SBATCH --job-name=F5FullMerge
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models_lsqr}"
SHARD_JOB_ID="${SHARD_JOB_ID:?Set SHARD_JOB_ID}"
SHARD_ROOT="$OUTPUT_ROOT/fsaverage_full_parallel_shards/job_$SHARD_JOB_ID"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$MODEL_ROOT" "$PROJECT_DIR/logs/figure5"
cd "$PROJECT_DIR"

"$PYTHON" analysis_results/figure5/scripts/merge_ridge_outer_fold_shards.py \
  --representation fsaverage_full_md \
  --shard-root "$SHARD_ROOT" \
  --output-root "$MODEL_ROOT" \
  --expected-outer-folds 5
