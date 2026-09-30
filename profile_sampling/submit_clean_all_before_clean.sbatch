#!/bin/bash
# Re-run cross-metric bad-column cleaning on existing profiles_before_clean data.
# The profile and region files are intentionally updated in place.
#SBATCH --job-name=badpoint-clean
#SBATCH --partition=normal
#SBATCH --array=0-337%8
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --output=logs/badpoint_clean_%A_%a.out
#SBATCH --error=logs/badpoint_clean_%A_%a.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mkdir -p logs

mapfile -t SUBJECTS < <(
    find /mnt/newStor/paros/paros_WORK/xintong/results/column_analysis \
        -mindepth 1 -maxdepth 1 -type d -name 'D*' -printf '%f\n' \
        | sort
)

SUBJECT_ID="${SUBJECTS[${SLURM_ARRAY_TASK_ID}]}"
ANALYSIS_ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis
PROFILE_DIR="$ANALYSIS_ROOT/$SUBJECT_ID/profiles_before_clean"
REGION_DIR="$ANALYSIS_ROOT/$SUBJECT_ID/regions"

echo "[$((SLURM_ARRAY_TASK_ID + 1))/${#SUBJECTS[@]}] START $SUBJECT_ID"

if [[ ! -d "$PROFILE_DIR" || ! -d "$REGION_DIR" ]]; then
    echo "SKIP $SUBJECT_ID: missing profile or region directory"
    exit 0
fi

python -u profile_sampling/clean_bad_trajectories.py \
    --subject-id "$SUBJECT_ID" \
    --region-dir "$REGION_DIR" \
    --profile-dir "$PROFILE_DIR" \
    --metrics md fa ad rd

echo "[$((SLURM_ARRAY_TASK_ID + 1))/${#SUBJECTS[@]}] DONE $SUBJECT_ID"
