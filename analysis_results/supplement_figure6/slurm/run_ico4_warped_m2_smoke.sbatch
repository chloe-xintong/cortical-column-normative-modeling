#!/bin/bash
#SBATCH --job-name=fig6_i4_smoke
#SBATCH --array=0-199%50
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=analysis_results/figure6/ico4_warped_m2_186/logs/smoke_%A_%a.out
#SBATCH --error=analysis_results/figure6/ico4_warped_m2_186/logs/smoke_%A_%a.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
RESULT_ROOT="${REPO_ROOT}/analysis_results/figure6/ico4_warped_m2_186"
FEATURE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/features"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fig6_i4_smoke_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fig6_i4_smoke_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/run_ico4_warped_m2_task.py" \
  --result-root "${RESULT_ROOT}" \
  --subject-order "${FEATURE_ROOT}/subject_order.csv" \
  --ico4-md "${FEATURE_ROOT}/matrices/X_ico4_md.npy" \
  --stage smoke \
  --node-list "${RESULT_ROOT}/smoke_nodes.csv" \
  --depth-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
