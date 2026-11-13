#!/usr/bin/env bash
#SBATCH --job-name=F5FullFS
#SBATCH --partition=normal
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
FEATURE_DIR="$OUTPUT_ROOT/features"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models_lsqr}"
SHARD_ROOT="$OUTPUT_ROOT/fsaverage_full_parallel_shards/job_$SLURM_ARRAY_JOB_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$MODEL_ROOT" "$SHARD_ROOT" "$PROJECT_DIR/logs/figure5"
cd "$PROJECT_DIR"

if [[ ! -f "$FEATURE_DIR/matrices/X_fsaverage_full_md.npy" ]]; then
  echo "Missing full-fsaverage feature matrix; build it before submitting shards" >&2
  exit 2
fi

export PROJECT_DIR PYTHON OUTPUT_ROOT FEATURE_DIR SHARD_ROOT
case "$SLURM_ARRAY_TASK_ID" in
  0) OUTER_FOLDS="0,4" ;;
  1) OUTER_FOLDS="1" ;;
  2) OUTER_FOLDS="2" ;;
  3) OUTER_FOLDS="3" ;;
  *) echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

"$PYTHON" analysis_results/figure5/scripts/run_ridge.py \
  --representation fsaverage_full_md \
  --feature-dir "$FEATURE_DIR" \
  --fold-dir "$OUTPUT_ROOT/folds" \
  --output-root "$SHARD_ROOT/shard_$SLURM_ARRAY_TASK_ID" \
  --outer-fold-indices "$OUTER_FOLDS" \
  --solver lsqr
