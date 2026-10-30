#!/bin/bash
#SBATCH --job-name=o2-column-20
#SBATCH --partition=normal
#SBATCH --array=0-19%4
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/o2_column_20_%A_%a.out
#SBATCH --error=logs/o2_column_20_%A_%a.err

set -euo pipefail

PROJECT_DIR=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
SUBJECT_MANIFEST="$PROJECT_DIR/manifests/o2_column_additional_20_subjects.txt"
SOURCE_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/archive/legacy_pipeline/native_column_fsaverage
COLUMN_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns
OUTPUT_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/qc/template_registration_comparison/o2_column/fsaverage

cd "$PROJECT_DIR"
mkdir -p logs

mapfile -t SUBJECTS < <(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$SUBJECT_MANIFEST")
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

echo "PROGRESS subject=$SUBJECT step=started"

for hemisphere in lh rh; do
    mapping_file="$COLUMN_ROOT/$SUBJECT/columns/${SUBJECT}_${hemisphere}_columns.npz"
    output="$OUTPUT_ROOT/$SUBJECT/${SUBJECT}_${hemisphere}_profiles.npz"
    "$PYTHON" -u analysis_results/figure2/o2_column/generate_o2_column_profiles.py \
        --subject-id "$SUBJECT" \
        --hemisphere "$hemisphere" \
        --template fsaverage \
        --mapping-root "$SOURCE_ROOT" \
        --mapping-file "$mapping_file" \
        --output "$output"
    echo "PROGRESS subject=$SUBJECT hemisphere=$hemisphere step=complete"
done

echo "PROGRESS subject=$SUBJECT step=completed"
