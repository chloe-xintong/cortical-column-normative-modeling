#!/usr/bin/env bash
#SBATCH --job-name=F4Ico5Pairs
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
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs/figure4" "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON" analysis_results/figure4/scripts/build_ico5_within_pairs.py \
  --subjects-dir "$SUBJECTS_DIR" \
  --fsaverage5-dir "$FSAVERAGE5_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --n-seeds 10 --batch-size 24
