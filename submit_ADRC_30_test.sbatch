#!/bin/bash
#SBATCH --job-name=ADRCv2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail

PIPELINE_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
RAW_ROOT="/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi"
RESULTS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis"
SUBJECT_LIST="$PIPELINE_DIR/data_preparation/data_inventory/adrc_preprocessing_pending_manifest.csv"
PREPROCESS="$PIPELINE_DIR/preprocessing/dwi/ADRC/preprocess_dwi.sh"
RECONALL="$PIPELINE_DIR/preprocessing/t1/run_freesurfer.sh"
COLUMN_RUNNER="$PIPELINE_DIR/pipeline/run_subject.py"
FREESURFER_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"
MONITOR="$PIPELINE_DIR/monitor_job_resources.sh"
LOG_DIR="$PIPELINE_DIR/logs"

: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
mkdir -p "$LOG_DIR"
START_EPOCH=$(date +%s)
START_ISO=$(date --iso-8601=seconds)

CSV_ROW=$((SLURM_ARRAY_TASK_ID + 2))
SUBJECT=$(awk -F',' -v row="$CSV_ROW" 'NR == row {gsub(/\r/,"",$1); print $1}' "$SUBJECT_LIST")
[[ "$SUBJECT" =~ ^D[0-9]+$ ]] || { echo "Invalid subject: $SUBJECT"; exit 1; }

SUBJECT_ROOT="$RAW_ROOT/$SUBJECT"
OUT="$OUTPUT_ROOT/$SUBJECT"
mkdir -p "$OUT"
T1="$SUBJECT_ROOT/T1.nii.gz"
[[ -s "$T1" ]] || { echo "Missing T1 for $SUBJECT"; exit 1; }

# The raw folder contains one DWI, one reverse-PE b0, and matching bval/bvec.
DWI=$(find "$SUBJECT_ROOT" -maxdepth 1 -type f -name '*.nii.gz' \
    ! -name '*revphase*' ! -name 'T1*' | sort | head -1)
REV=$(find "$SUBJECT_ROOT" -maxdepth 1 -type f -name '*revphase*.nii.gz' | sort | head -1)
[[ -n "$DWI" && -n "$REV" ]] || { echo "Missing DWI or reverse b0 for $SUBJECT"; exit 1; }

STEM=$(basename "$DWI" .nii.gz)
BVEC="$SUBJECT_ROOT/$STEM.bvec"
BVAL="$SUBJECT_ROOT/$STEM.bval"
[[ -s "$BVEC" && -s "$BVAL" ]] || { echo "Missing gradients for $SUBJECT: $STEM"; exit 1; }

# JSON is the acquisition source; j is the confirmed y+ direction (PA here).
JSON="$SUBJECT_ROOT/$STEM.json"
READOUT=$(python - "$JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    value = json.load(f).get('TotalReadoutTime')
if not isinstance(value, (int, float)) or value <= 0:
    raise SystemExit('invalid TotalReadoutTime')
print(value)
PY
)

FINAL="$OUT/${SUBJECT}_rd.nii.gz"
if [[ -s "$FINAL" && -s "$OUT/${SUBJECT}_fa.nii.gz" && -s "$OUT/${SUBJECT}_md.nii.gz" \
      && -s "$OUT/${SUBJECT}_ad.nii.gz" && -s "$OUT/${SUBJECT}_04_preprocessed.mif" ]]; then
    echo "SKIP: DWI outputs already exist for $SUBJECT"
    DWI_DONE=1
else
    DWI_DONE=0
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
export MRTRIX_NTHREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Subject=$SUBJECT Node=$(hostname) CPUs=$SLURM_CPUS_PER_TASK"
echo "Start time=$START_ISO"
echo "Allocated memory=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-unknown}}"
echo "T1=$T1"
echo "DWI=$DWI"
echo "Reverse=$REV PE=PA Readout=$READOUT"

run_subject() {
    if [[ -s "$FREESURFER_ROOT/$SUBJECT/mri/wmparc.mgz" ]]; then
        echo "SKIP: FreeSurfer already complete for $SUBJECT"
    else
        bash "$RECONALL" "$SUBJECT" "$T1" "$FREESURFER_ROOT" "$SLURM_CPUS_PER_TASK"
    fi
    if (( DWI_DONE == 0 )); then
        bash "$PREPROCESS" "$DWI" "$BVEC" "$BVAL" "$REV" "$OUT" "$SUBJECT" PA "$READOUT"
    fi
}

run_subject &
PIPELINE_PID=$!
RESOURCE_LOG="$LOG_DIR/live_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SUBJECT}.log"
TIME_LOG="$LOG_DIR/final_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SUBJECT}.txt"
"$MONITOR" "$PIPELINE_PID" "$RESOURCE_LOG" 60 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
if wait "$PIPELINE_PID"; then
    STATUS=0
else
    STATUS=$?
fi

if (( STATUS == 0 )); then
    echo "Preprocessing succeeded; starting column analysis before cleaning."
    python "$COLUMN_RUNNER" \
        --subject-id "$SUBJECT" \
        --t1 "$T1" \
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
        --skip-dwi-preprocess \
        --skip-freesurfer \
        --before-clean
    STATUS=$?
else
    echo "Preprocessing failed; column analysis will not run."
fi
END_EPOCH=$(date +%s)
END_ISO=$(date --iso-8601=seconds)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
    echo "Subject: $SUBJECT"
    echo "Node: $(hostname)"
    echo "Job ID: ${SLURM_JOB_ID:-unknown}"
    echo "Array task: ${SLURM_ARRAY_TASK_ID:-unknown}"
    echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
    echo "Allocated memory: ${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-unknown}}"
    echo "Start time: $START_ISO"
    echo "End time: $END_ISO"
    echo "Elapsed seconds: $ELAPSED"
    echo "Elapsed hours: $(awk -v s="$ELAPSED" 'BEGIN {printf "%.2f", s/3600}')"
    echo "Pipeline exit status: $STATUS"
} | tee "$TIME_LOG"
exit "$STATUS"
