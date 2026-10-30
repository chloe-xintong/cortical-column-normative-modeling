#!/usr/bin/env bash
#SBATCH --job-name=F4PcaNull
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
python "$PROJECT_DIR/analysis_results/figure4/scripts/compute_surface_pca_null.py" \
  --manifest "${MANIFEST:?Set MANIFEST}" --index "$SLURM_ARRAY_TASK_ID" \
  --pair-dir "${PAIR_DIR:?Set PAIR_DIR}" --real-results-root "${REAL_RESULTS_ROOT:?Set REAL_RESULTS_ROOT}" \
  --subjects-dir "${SUBJECTS_DIR:?Set SUBJECTS_DIR}" --output-root "${OUTPUT_ROOT:?Set OUTPUT_ROOT}" \
  --replicates "${N_NULL_REPLICATES:-5}" --variance-explained "${PCA_VARIANCE_EXPLAINED:-0.99}" \
  --seed "${RANDOM_SEED:-20261021}" --diffusion-strength "${DIFFUSION_STRENGTH:-0}" \
  --diffusion-steps "${DIFFUSION_STEPS:-1}" \
  --rotation-mode "${ROTATION_MODE:-independent_components}"
