#!/bin/bash
#SBATCH --job-name=ADRCretry8
#SBATCH --partition=normal
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
RAW_ROOT="/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi_retry_20260929"
RESULTS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis_retry_20260929"
FREESURFER_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"

SUBJECTS=(D0153 D0217 D0304 D0329 D0336 D0372 D0373 D0410)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
RAW_DIR="$RAW_ROOT/$SUBJECT"
OUT="$OUTPUT_ROOT/$SUBJECT"

mkdir -p "$OUT" "$RESULTS_ROOT/$SUBJECT"
echo "START subject=${SUBJECT} node=$(hostname)"

DWI=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*-2.nii.gz' | sort | head -1)
REV=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*-1.nii.gz' | sort | head -1)
STEM=$(basename "$DWI" .nii.gz)
BVEC="$RAW_DIR/${STEM}.bvec"
BVAL="$RAW_DIR/${STEM}.bval"
JSON="$RAW_DIR/${STEM}.json"

READOUT=$(python - "$JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("TotalReadoutTime")
if not isinstance(value, (int, float)) or value <= 0:
    raise SystemExit("invalid TotalReadoutTime")
print(value)
PY
)

echo "DWI=$DWI"
echo "Reverse b0=$REV"
echo "PE=AP readout=${READOUT}"

bash "$PROJECT_DIR/preprocessing/dwi/ADRC/preprocess_dwi.sh" \
    "$DWI" "$BVEC" "$BVAL" "$REV" "$OUT" "$SUBJECT" AP "$READOUT"

python "$PROJECT_DIR/pipeline/run_analysis.py" \
    --subject-id "$SUBJECT" \
    --subjects-dir "$FREESURFER_ROOT" \
    --adc-image "$OUT/${SUBJECT}_md.nii.gz" \
    --md-image "$OUT/${SUBJECT}_md.nii.gz" \
    --fa-image "$OUT/${SUBJECT}_fa.nii.gz" \
    --ad-image "$OUT/${SUBJECT}_ad.nii.gz" \
    --rd-image "$OUT/${SUBJECT}_rd.nii.gz" \
    --registration-reference "$OUT/${SUBJECT}_corrected_mean_b0_brain.nii.gz" \
    --analysis-dir "$RESULTS_ROOT/$SUBJECT" \
    --before-clean

echo "DONE subject=${SUBJECT}"
