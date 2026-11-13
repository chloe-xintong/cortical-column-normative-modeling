#!/usr/bin/env bash
#SBATCH --job-name=Figure2Full
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/figure2-full-${SLURM_JOB_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${1:?Path to full_cohort.py}"
STAGE="${2:?stage}"
shift 2
EXTRA=()
if [[ "$STAGE" == subject ]]; then EXTRA=(--index "${SLURM_ARRAY_TASK_ID}"); fi
"$PYTHON_BIN" -u "$RUNNER" "$STAGE" "$@" "${EXTRA[@]}"
