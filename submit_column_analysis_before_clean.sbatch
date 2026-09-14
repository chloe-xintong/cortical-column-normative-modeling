#!/bin/bash
# Run existing ADRC V2 outputs through profile sampling and stop before cleaning.
# This script does not run DWI preprocessing or recon-all.
#SBATCH --job-name=ADRCprofiles
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --array=0-133%30
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
RAW_ROOT="/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE"
DWI_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi"
FREESURFER_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"
RESULTS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis"
MANIFEST="$PROJECT_DIR/data_preparation/data_inventory/column_analysis_before_clean_pending_manifest.csv"

ROW=$((SLURM_ARRAY_TASK_ID + 2))
SUBJECT=$(awk -F',' -v row="$ROW" 'NR == row {gsub(/\r/,"",$1); print $1}' "$MANIFEST")
[[ "$SUBJECT" =~ ^D[0-9]+$ ]] || { echo "Invalid subject: $SUBJECT"; exit 1; }

RAW_DIR="$RAW_ROOT/$SUBJECT"
DWI=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*.nii.gz' ! -name '*revphase*' ! -name 'T1*' | sort | head -1)
REV=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*revphase*.nii.gz' | sort | head -1)
STEM=$(basename "$DWI" .nii.gz)
JSON="$RAW_DIR/$STEM.json"
BVEC="$RAW_DIR/$STEM.bvec"
BVAL="$RAW_DIR/$STEM.bval"
READOUT=$(python - "$JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    value = json.load(f).get("TotalReadoutTime")
if not isinstance(value, (int, float)) or value <= 0:
    raise SystemExit("invalid TotalReadoutTime")
print(value)
PY
)

echo "Subject=$SUBJECT Node=$(hostname)"
echo "Using existing DWI and FreeSurfer outputs; preprocessing disabled"

python "$PROJECT_DIR/pipeline/run_subject.py" \
  --subject-id "$SUBJECT" \
  --t1 "$RAW_DIR/T1.nii.gz" \
  --dwi "$DWI" \
  --bvec "$BVEC" \
  --bval "$BVAL" \
  --reverse-phase "$REV" \
  --pe-dir PA \
  --readout-time "$READOUT" \
  --output-dir "$RESULTS_ROOT/$SUBJECT" \
  --processed-dwi-root "$DWI_ROOT" \
  --processed-freesurfer-root "$FREESURFER_ROOT" \
  --analysis-root "$RESULTS_ROOT" \
  --skip-dwi-preprocess \
  --skip-freesurfer \
  --before-clean
