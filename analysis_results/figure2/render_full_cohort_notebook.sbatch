#!/usr/bin/env bash
#SBATCH --job-name=F2FullNotebook
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/f2nb-mpl-${SLURM_JOB_ID}"
export IPYTHONDIR="${SLURM_TMPDIR:-/tmp}/f2nb-ipython-${SLURM_JOB_ID}"
export JUPYTER_CONFIG_DIR="${SLURM_TMPDIR:-/tmp}/f2nb-config-${SLURM_JOB_ID}"
export JUPYTER_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/f2nb-runtime-${SLURM_JOB_ID}"
"${NOTEBOOK_PYTHON:?Set NOTEBOOK_PYTHON to a Jupyter-enabled Python}" -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 "${1:?Notebook path}"
