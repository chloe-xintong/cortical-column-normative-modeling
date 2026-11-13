#!/usr/bin/env bash
#SBATCH --job-name=F5O2FS
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%A_%a.err

set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-o2fs-${SLURM_JOB_ID}"
STAGE="${1:?stage}"
OUTPUT="${2:?output directory}"
EXTRA=()
if [[ "$STAGE" == prepare ]]; then
    EXTRA=(--subject-index "${SLURM_ARRAY_TASK_ID}")
elif [[ "$STAGE" == train ]]; then
    EXTRA=(--model-index "${SLURM_ARRAY_TASK_ID}")
fi
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python -u \
    analysis_results/figure5/scripts/option2_fs_baselines.py \
    "$STAGE" --output "$OUTPUT" "${EXTRA[@]}"
