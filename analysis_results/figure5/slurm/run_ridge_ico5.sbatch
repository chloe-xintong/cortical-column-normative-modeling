#!/usr/bin/env bash
#SBATCH --job-name=F5Ico5
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models}"
RIDGE_SOLVER="${RIDGE_SOLVER:-auto}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR"
mkdir -p "$MODEL_ROOT"
"$PYTHON" analysis_results/figure5/scripts/run_ridge.py \
  --representation ico5_md \
  --feature-dir "$OUTPUT_ROOT/features" \
  --fold-dir "$OUTPUT_ROOT/folds" \
  --output-root "$MODEL_ROOT" \
  --solver "$RIDGE_SOLVER"
