#!/usr/bin/env bash
#SBATCH --job-name=F2RegQC
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure2_registration_qc/%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure2_registration_qc/%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
RESULT_DIR="/mnt/newStor/paros/paros_WORK/xintong/results/qc/template_registration_comparison/figure2_method_comparison_fsaverage"
QUEUE="$RESULT_DIR/qc/registration_visual_review_queue.csv"
QC_TABLE="$RESULT_DIR/qc/figure2C_registration_input_qc.csv"
OUTPUT_DIR="$RESULT_DIR/figures/registration_contact_sheets"

mapfile -t SUBJECTS < <(tail -n +2 "$QUEUE" | cut -d, -f1)
SUBJECT_ID="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs/figure2_registration_qc"

python "$PROJECT_DIR/analysis_results/figure2/figure2_registration_contact_sheets.py" \
  --review-queue "$QUEUE" \
  --registration-qc "$QC_TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --subject-id "$SUBJECT_ID"
