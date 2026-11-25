#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
REPO=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
SCRIPT_ROOT='analysis_results/figure6/确定扰动因素/distance/scripts'
LOG_DIR=/tmp/ico4_null_E_logs
mkdir -p "$LOG_DIR"
cd "$REPO"

seq 0 9 | xargs -n 1 -P 3 -I IDX bash -c \
  '"'$PYTHON'" "'$SCRIPT_ROOT'/compute_ico4_null_adjusted_E_subject.py" --index IDX > "'$LOG_DIR'/IDX.log" 2>&1'
MPLCONFIGDIR=/tmp/matplotlib \
  "$PYTHON" "$SCRIPT_ROOT/summarize_ico4_null_adjusted_E_stability.py"
