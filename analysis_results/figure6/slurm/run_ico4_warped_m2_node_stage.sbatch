#!/bin/bash
#SBATCH --job-name=fig6_i4_node
#SBATCH --array=0-999%20
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=analysis_results/figure6/ico4_warped_m2_186/logs/node_%A_%a.out
#SBATCH --error=analysis_results/figure6/ico4_warped_m2_186/logs/node_%A_%a.err

set -euo pipefail
if [[ "$#" -ne 3 ]]; then
  echo "usage: sbatch ... $0 STAGE DEPTH_INDICES NODE_OFFSET" >&2
  exit 2
fi
STAGE="$1"
DEPTH_INDICES="$2"
NODE_OFFSET="$3"
GLOBAL_NODE=$((NODE_OFFSET + SLURM_ARRAY_TASK_ID))
REPO_ROOT="${SLURM_SUBMIT_DIR}"
RESULT_ROOT="${REPO_ROOT}/analysis_results/figure6/ico4_warped_m2_186"
FEATURE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/features"
PYTHON="${REPO_ROOT}/.venv/figure6-pcntoolkit-1.3.0/bin/python"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fig6_i4_${STAGE}_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fig6_i4_${STAGE}_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

for FOLD in 0 1 2 3 4; do
  GLOBAL_TASK=$((GLOBAL_NODE * 5 + FOLD))
  "${PYTHON}" "${REPO_ROOT}/analysis_results/figure6/scripts/run_ico4_warped_m2_task.py" \
    --result-root "${RESULT_ROOT}" \
    --subject-order "${FEATURE_ROOT}/subject_order.csv" \
    --ico4-md "${FEATURE_ROOT}/matrices/X_ico4_md.npy" \
    --stage "${STAGE}" \
    --depth-indices "${DEPTH_INDICES}" \
    --task-id "${GLOBAL_TASK}"
done
