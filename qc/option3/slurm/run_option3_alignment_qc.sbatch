#!/usr/bin/env bash
#SBATCH --job-name=O3QC_v2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/option3_qc/alignment_v2_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/option3_qc/alignment_v2_%j.err
set -euo pipefail
PROJECT_DIR=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
cd "$PROJECT_DIR"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/o3_qc_mpl_${SLURM_JOB_ID}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PYTHON="${PYTHON:-/home/apps/ubuntu-22.04/anaconda3/bin/python}"
"$PYTHON" -u qc/option3/scripts/run_option3_alignment_qc.py "$@"
