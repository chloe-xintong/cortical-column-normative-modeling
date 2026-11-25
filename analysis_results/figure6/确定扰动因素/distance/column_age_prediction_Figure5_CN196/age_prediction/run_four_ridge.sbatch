#!/bin/bash
#SBATCH --job-name=F5_column_Ridge
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/logs/%A_%a.out
#SBATCH --error=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/logs/%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
root=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction
models=(column_E_local column_plateau column_E_local_plateau ico4_md)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$root/logs" "$root/models_lsqr" "$MPLCONFIGDIR"
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python analysis_results/figure5/scripts/run_ridge.py \
 --representation "${models[$SLURM_ARRAY_TASK_ID]}" --feature-dir "$root/features" \
 --fold-dir "$root/folds" --output-root "$root/models_lsqr" --solver lsqr --alpha-count 21
