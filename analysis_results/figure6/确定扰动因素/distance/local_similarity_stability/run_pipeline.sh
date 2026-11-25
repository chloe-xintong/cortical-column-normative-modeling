#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
REPO=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
LOG_DIR=/tmp/ico4_local_similarity_subject_logs
mkdir -p "$LOG_DIR"
cd "$REPO"

"$PYTHON" 'analysis_results/figure6/确定扰动因素/distance/scripts/build_ico4_local_similarity_pairs.py'
for index in $(seq 0 36); do
  "$PYTHON" 'analysis_results/figure6/确定扰动因素/distance/scripts/compute_ico4_local_similarity_subject.py' \
    --index "$index" > "$LOG_DIR/$index.log" 2>&1
done
"$PYTHON" 'analysis_results/figure6/确定扰动因素/distance/scripts/summarize_ico4_local_similarity_stability.py'
"$PYTHON" 'analysis_results/figure6/确定扰动因素/distance/scripts/summarize_ico4_platform_k_difference.py'
