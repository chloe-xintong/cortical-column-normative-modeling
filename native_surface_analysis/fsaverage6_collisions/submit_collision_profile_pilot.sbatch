#!/bin/bash
#SBATCH --job-name=collProf
#SBATCH --partition=normal
#SBATCH --array=0-39%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/collProf_%A_%a.out
#SBATCH --error=logs/collProf_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis
mkdir -p logs
SUBJECT=$(find /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
if [[ -z "${SUBJECT}" ]]; then echo "No subject for index ${SLURM_ARRAY_TASK_ID}"; exit 1; fi
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/collision_profile_analysis_pilot/shards
mkdir -p "$OUT"
echo "START subject=${SUBJECT} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
python -u -m native_surface_analysis.fsaverage6_collisions.analyze_collision_profiles --subjects "$SUBJECT" --output-dir "$OUT"
echo "DONE subject=${SUBJECT}"
