#!/usr/bin/env bash
#SBATCH --job-name=F4PermSummary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
python "$PROJECT_DIR/analysis_results/figure4/scripts/summarize_whole_profile_permutation_pilot.py" \
  --manifest "${MANIFEST:?Set MANIFEST}" \
  --number-of-subjects "${NUMBER_OF_SUBJECTS:-5}" \
  --permutation-root "${PERMUTATION_ROOT:?Set PERMUTATION_ROOT}" \
  --acquisition-summary "${ACQUISITION_SUMMARY:?Set ACQUISITION_SUMMARY}" \
  --output-dir "${OUTPUT_DIR:?Set OUTPUT_DIR}"
