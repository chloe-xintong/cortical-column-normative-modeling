#!/bin/bash
#SBATCH --job-name=F5_graph_Ridge
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/graph_ridge/logs/%A_%a.out
#SBATCH --error=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/graph_ridge/logs/%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/graph_ridge/logs
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python analysis_results/figure6/确定扰动因素/distance/scripts/run_graph_ridge_outer_fold.py --outer-fold "$SLURM_ARRAY_TASK_ID"
