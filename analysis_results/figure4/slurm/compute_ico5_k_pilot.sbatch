#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5K
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%5
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
MANIFEST="$PROJECT_DIR/manifests/figure4_valid_325_subjects.csv"
SUBJECT_LIST="$PROJECT_DIR/manifests/figure4_ico5_k_pilot20_subjects.txt"
PAIR_DIR="${PAIR_DIR:?Set PAIR_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
EXISTING_RESULTS_ROOT="${EXISTING_RESULTS_ROOT:-}"
NEW_K_VALUES="${NEW_K_VALUES:-}"
DIAGNOSTIC_K10="${DIAGNOSTIC_K10:-0}"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_ROOT"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
EXTRA_ARGS=()
if [[ -n "$EXISTING_RESULTS_ROOT" ]]; then
  EXTRA_ARGS+=(--existing-results-root "$EXISTING_RESULTS_ROOT")
fi
if [[ -n "$NEW_K_VALUES" ]]; then
  read -r -a K_ARGS <<< "$NEW_K_VALUES"
  EXTRA_ARGS+=(--new-k-values "${K_ARGS[@]}")
fi
if [[ "$DIAGNOSTIC_K10" == "1" ]]; then
  EXTRA_ARGS+=(--diagnostic-k10)
fi
"$PYTHON" analysis_results/figure4/scripts/compute_ico5_k_subject.py \
  --manifest "$MANIFEST" --subject-list "$SUBJECT_LIST" \
  --pair-dir "$PAIR_DIR" --output-root "$OUTPUT_ROOT" \
  --index "$SLURM_ARRAY_TASK_ID" \
  "${EXTRA_ARGS[@]}"
