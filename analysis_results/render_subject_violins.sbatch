#!/usr/bin/env bash
#SBATCH --job-name=SubjectViolins
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/subject-violins-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python \
  analysis_results/render_subject_violins.py
