#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
REPO=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
SCRIPT_ROOT='analysis_results/figure6/确定扰动因素/distance/scripts'
LOG_DIR=/tmp/ico4_platform_sensitivity_logs
mkdir -p "$LOG_DIR"
cd "$REPO"

"$PYTHON" "$SCRIPT_ROOT/build_ico4_platform_k300_pairs.py"
for index in $(seq 0 36); do
  "$PYTHON" "$SCRIPT_ROOT/compute_ico4_platform_sensitivity_subject.py" \
    --index "$index" > "$LOG_DIR/$index.log" 2>&1
done
MPLCONFIGDIR=/tmp/matplotlib \
  "$PYTHON" "$SCRIPT_ROOT/summarize_ico4_platform_sampling_sensitivity.py"
