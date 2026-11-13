#!/usr/bin/env bash
#SBATCH --job-name=F5cBAGIco5
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/cbag_crossfit_lsqr"
"$PYTHON" analysis_results/figure5/scripts/compute_crossfit_cbag.py \
  --representation ico5_md \
  --feature-dir "$OUTPUT_ROOT/features" \
  --fold-dir "$OUTPUT_ROOT/folds" \
  --model-root "$OUTPUT_ROOT/models_lsqr" \
  --output-root "$OUTPUT_ROOT/cbag_crossfit_lsqr" \
  --solver lsqr
