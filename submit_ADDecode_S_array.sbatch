#!/bin/bash
# Preprocess the currently available ADDecode S-series DWI subjects.
# Outputs are kept separate from the existing D-series results.
#SBATCH --job-name=ADDecodeS
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-88%4
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail

PIPELINE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
RAW_ROOT="/mnt/newStor/paros/paros_WORK/ADDecode_raw_data"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi_adddecode"

source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis
cd "$PIPELINE_ROOT"
mkdir -p "$OUTPUT_ROOT"

mapfile -t SUBJECTS < <(find "$RAW_ROOT" -maxdepth 1 -type f -name 'S*_DWI_nii4D.nii.gz' -printf '%f\n' | sed 's/_DWI_nii4D\.nii\.gz$//' | sort)
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$SUBJECT" ]]; then
  echo "ERROR: no subject for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

DWI="$RAW_ROOT/${SUBJECT}_DWI_nii4D.nii.gz"
BVEC="$RAW_ROOT/${SUBJECT}_DWI_nii4D.bvecs"
BVAL="$RAW_ROOT/${SUBJECT}_DWI_nii4D.bvals"
for f in "$DWI" "$BVEC" "$BVAL"; do
  [[ -s "$f" ]] || { echo "ERROR: missing input $f" >&2; exit 1; }
done

OUT="$OUTPUT_ROOT/$SUBJECT"
echo "Subject: $SUBJECT"
echo "Node: $(hostname)"
echo "Output: $OUT"
python -u pipeline/run_preprocess.py \
  --subject-id "$SUBJECT" \
  --dwi "$DWI" \
  --bvec "$BVEC" \
  --bval "$BVAL" \
  --output-dir "$OUT"
echo "Finished: $SUBJECT $(date)"
