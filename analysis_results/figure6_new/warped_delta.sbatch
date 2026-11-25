#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/delta-cache-$SLURM_JOB_ID"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
mkdir -p "$MPLCONFIGDIR"
python_bin=$1 script=$2 action=$3 root=$4
if [[ "$action" == shard ]]; then
    start=$((SLURM_ARRAY_TASK_ID * 5))
    exec "$python_bin" -u "$script" "$action" "$root" "$start" "$((start + 5))"
fi
exec "$python_bin" -u "$script" "$action" "$root"
