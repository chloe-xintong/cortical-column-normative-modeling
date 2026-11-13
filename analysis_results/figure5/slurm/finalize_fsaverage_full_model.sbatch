#!/usr/bin/env bash
#SBATCH --job-name=F5FullFinalize
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models_lsqr}"
NOTEBOOK="$PROJECT_DIR/analysis_results/figure5/notebooks/plot_figure5.ipynb"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_ROOT/summary_lsqr" "$PROJECT_DIR/logs/figure5"
cd "$PROJECT_DIR"

"$PYTHON" analysis_results/figure5/scripts/summarize_models.py \
  --model-root "$MODEL_ROOT" \
  --output-dir "$OUTPUT_ROOT/summary_lsqr"
"$PYTHON" analysis_results/figure5/scripts/compute_figure5_paired_mae.py \
  --model-root "$MODEL_ROOT" \
  --output-dir analysis_results/figure5/outputs/statistics \
  --bootstrap-replicates 10000 \
  --seed 20261028
"$PYTHON" analysis_results/figure5/scripts/add_fsaverage_full_to_notebook.py \
  --notebook "$NOTEBOOK"
"$PYTHON" -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  "$NOTEBOOK"
