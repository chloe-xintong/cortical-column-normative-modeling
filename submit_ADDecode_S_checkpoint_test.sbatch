#!/bin/bash
# First checkpoint test: three S-series subjects, no topup.
#SBATCH --job-name=ADDecodeTest
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail
PIPELINE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
RAW_ROOT="/mnt/newStor/paros/paros_WORK/ADDecode_raw_data"
OUTPUT_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/addcode"
SUBJECTS=(S00775 S01412 S01470)
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis
cd "$PIPELINE_ROOT"
mkdir -p "$OUTPUT_ROOT/$SUBJECT"

python -u pipeline/run_preprocess.py \
  --subject-id "$SUBJECT" \
  --dwi "$RAW_ROOT/${SUBJECT}_DWI_nii4D.nii.gz" \
  --bvec "$RAW_ROOT/${SUBJECT}_DWI_nii4D.bvecs" \
  --bval "$RAW_ROOT/${SUBJECT}_DWI_nii4D.bvals" \
  --output-dir "$OUTPUT_ROOT/$SUBJECT"

bash preprocessing/dwi/ADDcode/check_outputs.sh "$SUBJECT" "$OUTPUT_ROOT/$SUBJECT" "$RAW_ROOT"
