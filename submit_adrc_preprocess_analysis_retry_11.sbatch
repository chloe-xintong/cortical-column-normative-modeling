#!/bin/bash
#SBATCH --job-name=ADRCfullRetry11
#SBATCH --partition=normal
#SBATCH --array=0-10%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
RAW_ROOT="/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi"
RESULTS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis"
FREESURFER_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"

SUBJECTS=(
    D0324 D0335 D0346 D0348 D0353 D0362
    D0368 D0383 D0400 D0409 D0418
)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
RAW_DIR="$RAW_ROOT/$SUBJECT"

mkdir -p "$OUTPUT_ROOT/$SUBJECT" "$RESULTS_ROOT/$SUBJECT"
echo "START subject=${SUBJECT} node=$(hostname)"

DWI=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*.nii.gz' \
    ! -name '*revphase*' ! -name 'T1*' | sort | head -1)
REV=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*revphase*.nii.gz' | sort | head -1)
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
    --processed-dwi-root "$OUTPUT_ROOT" \
    --processed-freesurfer-root "$FREESURFER_ROOT" \
    --analysis-root "$RESULTS_ROOT" \
    --before-clean

echo "DONE subject=${SUBJECT}"
