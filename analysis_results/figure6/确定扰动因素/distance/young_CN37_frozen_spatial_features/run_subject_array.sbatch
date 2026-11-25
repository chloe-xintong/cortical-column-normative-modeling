#!/bin/bash
#SBATCH --job-name=ico4_CN37_E100
#SBATCH --array=0-36%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=analysis_results/figure6/确定扰动因素/distance/young_CN37_frozen_spatial_features/logs/%A_%a.out
#SBATCH --error=analysis_results/figure6/确定扰动因素/distance/young_CN37_frozen_spatial_features/logs/%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
sid=$(/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python -c "import pandas as pd; print(pd.read_csv('analysis_results/figure6/确定扰动因素/distance/young_CN37_frozen_spatial_features/index/young_cn37_roster.csv',dtype={'subject_id':str}).iloc[$SLURM_ARRAY_TASK_ID].subject_id)")
done_file="analysis_results/figure6/确定扰动因素/distance/young_CN37_frozen_spatial_features/subjects/${sid}/completion.json"
if test -f "$done_file"; then
  echo "already_complete index=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python \
  analysis_results/figure6/确定扰动因素/distance/scripts/compute_ico4_CN37_frozen_subject.py \
  --index "$SLURM_ARRAY_TASK_ID"
