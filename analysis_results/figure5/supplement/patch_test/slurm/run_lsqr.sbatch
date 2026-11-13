#!/usr/bin/env bash
#SBATCH --job-name=F5Ico4PatchRidge
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
SUPPLEMENT_ROOT="${SUPPLEMENT_ROOT:-$OUTPUT_ROOT/supplement/patch_test}"
MODEL_ROOT="${MODEL_ROOT:-$SUPPLEMENT_ROOT/models_lsqr}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MODEL_ROOT" "$MPLCONFIGDIR"

cd "$PROJECT_DIR"
"$PYTHON" analysis_results/figure5/scripts/run_ridge.py \
  --representation ico4_patch_mean_md \
  --feature-dir "$SUPPLEMENT_ROOT/features" \
  --fold-dir "$OUTPUT_ROOT/folds" \
  --output-root "$MODEL_ROOT" \
  --solver lsqr
