#!/bin/bash
# Modern ADD preprocessing pilot for three S-series subjects.
# No reverse b0 or topup; protocol parameters are defined in ADD/preprocess_dwi.sh.
#SBATCH --job-name=ADDmodern
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

PROJECT_ROOT=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
RAW_ROOT=/mnt/newStor/paros/paros_WORK/ADDecode_raw_data
OUTPUT_ROOT=/mnt/newStor/paros/paros_WORK/xintong/data/processed/add_modern
SUBJECTS=(S00775 S01412 S01470)
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT/$SUBJECT"

echo "START subject=$SUBJECT task=${SLURM_ARRAY_TASK_ID}/2 host=$(hostname)"

python -u pipeline/run_preprocess.py \
    --subject-id "$SUBJECT" \
    --dwi "$RAW_ROOT/${SUBJECT}_DWI_nii4D.nii.gz" \
    --bvec "$RAW_ROOT/${SUBJECT}_DWI_nii4D.bvecs" \
    --bval "$RAW_ROOT/${SUBJECT}_DWI_nii4D.bvals" \
    --output-dir "$OUTPUT_ROOT/$SUBJECT"

DIFFUSION_DIR="$OUTPUT_ROOT/$SUBJECT/preprocess/diffusion"
for output in \
    "$DIFFUSION_DIR/${SUBJECT}_adc.nii.gz" \
    "$DIFFUSION_DIR/${SUBJECT}_fa.nii.gz" \
    "$DIFFUSION_DIR/${SUBJECT}_ad.nii.gz" \
    "$DIFFUSION_DIR/${SUBJECT}_rd.nii.gz"; do
    [[ -s "$output" ]] || {
        echo "FAILED subject=$SUBJECT missing=$output" >&2
        exit 1
    }
done

echo "DONE subject=$SUBJECT task=${SLURM_ARRAY_TASK_ID}/2"
