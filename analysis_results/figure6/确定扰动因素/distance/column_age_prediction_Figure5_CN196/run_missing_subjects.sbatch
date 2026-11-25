#!/bin/bash
#SBATCH --job-name=ico4_F5CN196_E100
#SBATCH --array=0-195%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/logs/%A_%a.out
#SBATCH --error=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/logs/%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
root=analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196
sid=$(/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python -c "import pandas as pd; print(pd.read_csv('$root/index/subject_order.csv',dtype={'subject_id':str}).iloc[$SLURM_ARRAY_TASK_ID].subject_id)")
if test -f "$root/subjects/$sid/completion.json"; then
  echo "already_complete index=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python \
 analysis_results/figure6/确定扰动因素/distance/scripts/compute_ico4_CN37_frozen_subject.py \
 --feature-root "$root" --index "$SLURM_ARRAY_TASK_ID"
