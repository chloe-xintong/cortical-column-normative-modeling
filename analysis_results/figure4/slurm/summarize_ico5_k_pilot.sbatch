#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5KSum
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
RESULTS_ROOT="${RESULTS_ROOT:?Set RESULTS_ROOT}"
PAIR_OUTPUT_DIR="${PAIR_OUTPUT_DIR:?Set PAIR_OUTPUT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
EXPECTED_SUBJECTS="${EXPECTED_SUBJECTS:-20}"
SUBJECT_LIST="${SUBJECT_LIST:-}"
K_VALUES="${K_VALUES:-1 3 5 10 15 20 25 30}"
REFERENCE_K="${REFERENCE_K:-30}"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
SUBJECT_ARGS=()
if [[ -n "$SUBJECT_LIST" ]]; then
  SUBJECT_ARGS=(--subject-list "$SUBJECT_LIST")
fi
read -r -a K_ARGS <<< "$K_VALUES"
"$PYTHON" analysis_results/figure4/scripts/summarize_ico5_k_convergence.py \
  --results-root "$RESULTS_ROOT" \
  --pair-output-dir "$PAIR_OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" --expected-subjects "$EXPECTED_SUBJECTS" \
  --k-values "${K_ARGS[@]}" --reference-k "$REFERENCE_K" \
  "${SUBJECT_ARGS[@]}"
