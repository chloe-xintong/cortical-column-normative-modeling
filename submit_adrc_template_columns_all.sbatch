#!/bin/bash
#SBATCH --job-name=ADRCcolumns
#SBATCH --partition=normal
#SBATCH --array=0-326%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PROJECT_PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
RAW_ROOT="/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE"
PROCESSED_DWI_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi"
FREESURFER_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"
RESULTS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis_template_fsaverage"
INVENTORY="$PROJECT_DIR/data_preparation/data_inventory/adrc_data_inventory.csv"
MONITOR="$PROJECT_DIR/monitor_job_resources.sh"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

mapfile -t SUBJECTS < <(
    "$PROJECT_PYTHON" - "$INVENTORY" "$PROCESSED_DWI_ROOT" "$FREESURFER_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

inventory_path = Path(sys.argv[1])
dwi_root = Path(sys.argv[2])
freesurfer_root = Path(sys.argv[3])

required_fs_suffixes = (
    "surf/lh.white",
    "surf/lh.pial",
    "surf/lh.sphere.reg",
    "surf/lh.thickness",
    "surf/rh.white",
    "surf/rh.pial",
    "surf/rh.sphere.reg",
    "surf/rh.thickness",
    "mri/orig.mgz",
)

with inventory_path.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        subject = row["subject"].strip()
        raw_paths = (
            row["t1_path"],
            row["dwi_path"],
            row["bval_path"],
            row["bvec_path"],
            row["reverse_b0_path"],
            row["dwi_json_path"],
        )
        dwi_dir = dwi_root / subject
        dwi_metric_paths = tuple(
            dwi_dir / f"{subject}_{metric}.nii.gz"
            for metric in ("fa", "ad", "rd")
        )
        md_paths = (
            dwi_dir / f"{subject}_adc.nii.gz",
            dwi_dir / f"{subject}_md.nii.gz",
        )
        dwi_paths = dwi_metric_paths + (
            dwi_dir / f"{subject}_05_biascorrected.mif",
            dwi_dir / f"{subject}_corrected_mean_b0_brain_mask.nii.gz",
        )
        fs_dir = freesurfer_root / subject
        fs_paths = tuple(fs_dir / suffix for suffix in required_fs_suffixes)
        if all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in raw_paths):
            if (
                all(path.is_file() and path.stat().st_size > 0 for path in dwi_paths + fs_paths)
                and any(path.is_file() and path.stat().st_size > 0 for path in md_paths)
            ):
                print(subject)
PY
)

if (( SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} )); then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID exceeds completed subject count ${#SUBJECTS[@]}"
    exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! "$SUBJECT" =~ ^D[0-9]+$ ]]; then
    echo "ERROR: invalid subject at manifest row $CSV_ROW: $SUBJECT"
    exit 1
fi

RAW_DIR="$RAW_ROOT/$SUBJECT"
OUTPUT_DIR="$RESULTS_ROOT/$SUBJECT"

DWI=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*.nii.gz' \
    ! -name '*revphase*' ! -name 'T1*' | sort | head -1)
REV=$(find "$RAW_DIR" -maxdepth 1 -type f -name '*revphase*.nii.gz' | sort | head -1)
T1="$RAW_DIR/T1.nii.gz"

if [[ -z "$DWI" || -z "$REV" || ! -s "$T1" ]]; then
    echo "ERROR: missing T1, DWI, or reverse-phase image for $SUBJECT"
    exit 1
fi

STEM=$(basename "$DWI" .nii.gz)
BVEC="$RAW_DIR/$STEM.bvec"
BVAL="$RAW_DIR/$STEM.bval"
JSON="$RAW_DIR/$STEM.json"

if [[ ! -s "$BVEC" || ! -s "$BVAL" || ! -s "$JSON" ]]; then
    echo "ERROR: missing gradients or JSON for $SUBJECT: $STEM"
    exit 1
fi

READOUT=$(
    "$PROJECT_PYTHON" - "$JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("TotalReadoutTime")
if not isinstance(value, (int, float)) or value <= 0:
    raise SystemExit("invalid TotalReadoutTime")
print(value)
PY
)

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
export MRTRIX_NTHREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "START subject=$SUBJECT task=$SLURM_ARRAY_TASK_ID node=$(hostname)"
echo "T1=$T1"
echo "DWI=$DWI"
echo "Reverse=$REV"
echo "Readout=$READOUT"
echo "Output=$OUTPUT_DIR"

"$PROJECT_PYTHON" "$PROJECT_DIR/pipeline/run_subject.py" \
    --subject-id "$SUBJECT" \
    --t1 "$T1" \
    --dwi "$DWI" \
    --bvec "$BVEC" \
    --bval "$BVAL" \
    --reverse-phase "$REV" \
    --pe-dir PA \
    --readout-time "$READOUT" \
    --output-dir "$OUTPUT_DIR" \
    --processed-dwi-root "$PROCESSED_DWI_ROOT" \
    --processed-freesurfer-root "$FREESURFER_ROOT" \
    --analysis-root "$RESULTS_ROOT" \
    --skip-dwi-preprocess \
    --skip-freesurfer \
    --template fsaverage

echo "DONE subject=$SUBJECT task=$SLURM_ARRAY_TASK_ID"
