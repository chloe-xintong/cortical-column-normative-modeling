#!/usr/bin/env bash
#SBATCH --job-name=PaletteNotebook
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/palette-mpl-${SLURM_JOB_ID}"
export IPYTHONDIR="${SLURM_TMPDIR:-/tmp}/palette-ipython-${SLURM_JOB_ID}"
export JUPYTER_CONFIG_DIR="${SLURM_TMPDIR:-/tmp}/palette-config-${SLURM_JOB_ID}"
export JUPYTER_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/palette-runtime-${SLURM_JOB_ID}"
palette_kernel_dir="${SLURM_TMPDIR:-/tmp}/palette-kernel-${SLURM_JOB_ID}"
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python -m ipykernel install \
    --prefix "$palette_kernel_dir" --name figure3-cortical
export JUPYTER_PATH="$palette_kernel_dir/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
for notebook in "$@"; do
    /home/apps/ubuntu-22.04/anaconda3/bin/python -m jupyter nbconvert \
        --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=figure3-cortical \
        --ExecutePreprocessor.timeout=1800 "$notebook"
done
