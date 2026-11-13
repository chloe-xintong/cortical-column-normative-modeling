#!/usr/bin/env bash
#SBATCH --job-name=F4Pairs
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure4/%x_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
PAIR_DIR="${PAIR_DIR:?Set PAIR_DIR}"
SUBJECTS_DIR="${SUBJECTS_DIR:?Set SUBJECTS_DIR}"

mkdir -p "$PROJECT_DIR/logs/figure4" "$PAIR_DIR"

"$PYTHON" "$PROJECT_DIR/analysis_results/figure4/scripts/build_geodesic_pair_tables.py" \
    --subjects-dir "$SUBJECTS_DIR" \
    --template fsaverage \
    --output-dir "$PAIR_DIR" \
    --anchors-per-hemisphere "${ANCHORS_PER_HEMISPHERE:-2048}" \
    --max-neighbors-per-anchor-bin "${MAX_NEIGHBORS_PER_ANCHOR_BIN:-20}" \
    --batch-size "${PAIR_BATCH_SIZE:-32}" \
    --seed "${RANDOM_SEED:-20261020}" \
    --resume
