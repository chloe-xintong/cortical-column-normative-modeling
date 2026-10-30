#!/usr/bin/env bash
#SBATCH --job-name=F5cBAG
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
REPRESENTATIONS=(regional_mean_md regional_thickness regional_profile ico3_md ico4_md ico3_md_length ico3_md_curvature ico3_md_length_curvature)
REPRESENTATION="${REPRESENTATIONS[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/cbag_crossfit_lsqr"
"$PYTHON" analysis_results/figure5/scripts/compute_crossfit_cbag.py \
  --representation "$REPRESENTATION" \
  --feature-dir "$OUTPUT_ROOT/features" \
  --fold-dir "$OUTPUT_ROOT/folds" \
  --model-root "$OUTPUT_ROOT/models_lsqr" \
  --output-root "$OUTPUT_ROOT/cbag_crossfit_lsqr" \
  --solver lsqr
