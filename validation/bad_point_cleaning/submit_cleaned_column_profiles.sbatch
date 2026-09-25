#!/bin/bash
# Stage each subject in a private temporary directory and publish atomically.
# Original profiles_before_clean and source regions are never modified.
#SBATCH --job-name=columnclean
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%10
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
ANALYSIS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/column_profiles"
STAGING_ROOT="$OUTPUT_ROOT/.tmp"
MANIFEST="$PROJECT_DIR/data_preparation/data_inventory/column_clean_pending_manifest.csv"
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')

[[ "$SUBJECT" =~ ^D[0-9]+$ ]] || { echo "Invalid subject: $SUBJECT"; exit 1; }
SOURCE="$ANALYSIS_ROOT/$SUBJECT"
OUTPUT="$OUTPUT_ROOT/$SUBJECT"
[[ -d "$SOURCE/profiles_before_clean" && -d "$SOURCE/regions" ]] || { echo "Missing source for $SUBJECT"; exit 1; }
mkdir -p "$STAGING_ROOT"

python -u "$PROJECT_DIR/validation/bad_point_cleaning/dry_run_cleaned_column_output.py" \
  --subject "$SUBJECT" \
  --analysis-root "$ANALYSIS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --staging-root "$STAGING_ROOT"
echo "Completed $SUBJECT on $(hostname)"
