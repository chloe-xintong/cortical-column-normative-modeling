#!/usr/bin/env bash
#SBATCH --job-name=F5RegProfThick
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_ROOT/regional_profile_thickness_analysis"
cd "$PROJECT_DIR"
"$PYTHON" analysis_results/figure5/scripts/run_ridge.py \
  --representation regional_profile_thickness \
  --feature-dir "$OUTPUT_ROOT/features" \
  --fold-dir "$OUTPUT_ROOT/folds" \
  --output-root "$OUTPUT_ROOT/models_lsqr" \
  --solver lsqr
"$PYTHON" analysis_results/figure5/scripts/compute_crossfit_cbag.py \
  --representation regional_profile_thickness \
  --feature-dir "$OUTPUT_ROOT/features" \
  --fold-dir "$OUTPUT_ROOT/folds" \
  --model-root "$OUTPUT_ROOT/models_lsqr" \
  --output-root "$OUTPUT_ROOT/cbag_crossfit_lsqr" \
  --solver lsqr
"$PYTHON" analysis_results/figure5/scripts/analyze_regional_profile_thickness.py \
  --model-root "$OUTPUT_ROOT/models_lsqr" \
  --cbag-root "$OUTPUT_ROOT/cbag_crossfit_lsqr" \
  --output-dir "$OUTPUT_ROOT/regional_profile_thickness_analysis" \
  --bootstrap-replicates 10000 \
  --seed 20261028
