#!/usr/bin/env bash
#SBATCH --job-name=depth_density
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/analysis_results/depth_density_sensitivity/slurm/depth_density_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/analysis_results/depth_density_sensitivity/slurm/depth_density_%j.err

set -euo pipefail

REPO=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
FIGURE3_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/figure3_depth_thickness
FIGURE5_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs
OUTPUT_DIR="$REPO/analysis_results/depth_density_sensitivity"
export MPLCONFIGDIR="/tmp/matplotlib_depth_density_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/depth_density_cache_${SLURM_JOB_ID}"

cd "$REPO"

"$PYTHON" analysis_results/depth_density_sensitivity/scripts/prepare_depth_density_sensitivity.py \
  --figure3-root "$FIGURE3_ROOT" \
  --figure5-root "$FIGURE5_ROOT" \
  --output-dir "$OUTPUT_DIR"

for N_DEPTHS in 6 11; do
  for REPRESENTATION in regional_profile ico4_md; do
    "$PYTHON" analysis_results/figure5/scripts/run_ridge.py \
      --representation "$REPRESENTATION" \
      --feature-dir "$OUTPUT_DIR/features/d${N_DEPTHS}" \
      --fold-dir "$FIGURE5_ROOT/folds" \
      --output-root "$OUTPUT_DIR/models/d${N_DEPTHS}" \
      --solver lsqr \
      --alpha-count 21
  done
done

"$PYTHON" analysis_results/depth_density_sensitivity/scripts/summarize_depth_density_sensitivity.py \
  --figure3-root "$FIGURE3_ROOT" \
  --figure5-root "$FIGURE5_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --bootstrap-replicates 10000
