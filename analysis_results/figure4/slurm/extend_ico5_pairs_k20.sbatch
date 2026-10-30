#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5Ext20
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
SUBJECTS_DIR="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"
FSAVERAGE5_DIR="/home/xintong/.conda/envs/cortical_trajectory_analysis/lib/python3.11/site-packages/nilearn/datasets/data/fsaverage5"
OLD_PAIR_DIR="${OLD_PAIR_DIR:?Set OLD_PAIR_DIR}"
OUTPUT_PAIR_DIR="${OUTPUT_PAIR_DIR:?Set OUTPUT_PAIR_DIR}"
REPORT="${REPORT:?Set REPORT}"
OLD_MAX_RANK="${OLD_MAX_RANK:-10}"
NEW_MAX_RANK="${NEW_MAX_RANK:-20}"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_PAIR_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" analysis_results/figure4/scripts/extend_ico5_pairs_to_k20.py \
  --subjects-dir "$SUBJECTS_DIR" --fsaverage5-dir "$FSAVERAGE5_DIR" \
  --old-pair-dir "$OLD_PAIR_DIR" --output-pair-dir "$OUTPUT_PAIR_DIR" \
  --report "$REPORT" --n-seeds 10 --batch-size 24 \
  --old-max-rank "$OLD_MAX_RANK" --new-max-rank "$NEW_MAX_RANK"
