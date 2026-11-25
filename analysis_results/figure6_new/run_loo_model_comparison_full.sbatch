#!/usr/bin/env bash
#SBATCH --job-name=F6YoungLOO
# Submit five arrays with NODE_OFFSET=0,1000,2000,3000,4000. Tasks whose
# offset node is >=4687 exit successfully without writing a shard.
#SBATCH --array=0-999%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=analysis_results/figure6_new/outputs/loo_model_comparison_full/logs/node_%A_%a.out
#SBATCH --error=analysis_results/figure6_new/outputs/loo_model_comparison_full/logs/node_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/figure6-new-cache-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/figure6-new-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
  analysis_results/figure6_new/outputs/loo_model_comparison_full/{logs,shards,status}

.venv/figure6-pcntoolkit-1.3.0/bin/python \
  analysis_results/figure6_new/compare_loo_models.py \
  --action task \
  --node-index "$(( ${NODE_OFFSET:?Set NODE_OFFSET when submitting} + SLURM_ARRAY_TASK_ID ))" \
  --output-dir analysis_results/figure6_new/outputs/loo_model_comparison_full
