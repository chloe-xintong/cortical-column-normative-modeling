#!/usr/bin/env bash
#SBATCH --job-name=Figure3
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/figure3-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
# sbatch copies this script into its spool; pass the source runner explicitly.
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${1:?Pass the path to run_all_figure3.py}"
shift
"$PYTHON_BIN" -u "$RUNNER" "$@"
