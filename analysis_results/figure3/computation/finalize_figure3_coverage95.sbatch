#!/usr/bin/env bash
#SBATCH --job-name=Fig3Finalize
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail

REPO="${1:?Pass repository root}"
OUTPUT="${2:?Pass completed Figure 3 output directory}"
NOTEBOOK="${3:?Pass Figure 3 notebook path}"
SUBMISSION="${4:?Pass active submission JSON}"
CONDA_PY=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
SYSTEM_PY=/home/apps/ubuntu-22.04/anaconda3/bin/python
JUPYTER=/home/apps/ubuntu-22.04/anaconda3/bin/jupyter

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/figure3-final-mpl-${SLURM_JOB_ID}"
export JUPYTER_CONFIG_DIR="${SLURM_TMPDIR:-/tmp}/figure3-final-jupyter-config-${SLURM_JOB_ID}"
export JUPYTER_DATA_DIR="${SLURM_TMPDIR:-/tmp}/figure3-final-jupyter-data-${SLURM_JOB_ID}"
export JUPYTER_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/figure3-final-jupyter-runtime-${SLURM_JOB_ID}"
export IPYTHONDIR="${SLURM_TMPDIR:-/tmp}/figure3-final-ipython-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"

"$CONDA_PY" -u "$REPO/analysis_results/figure3/computation/compare_pc1_pc2_thickness.py" \
  --output-dir "$OUTPUT" --seed 20261103 --bootstrap-samples 10000
"$CONDA_PY" -u "$REPO/analysis_results/figure3/computation/plot_pc2_vs_thickness.py" \
  --output-dir "$OUTPUT"
"$SYSTEM_PY" "$REPO/analysis_results/figure3/computation/update_figure3_notebook_panel_a.py" "$NOTEBOOK"
cd "$REPO"
"$JUPYTER" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cortical_trajectory_analysis \
  --ExecutePreprocessor.timeout=1800 "$NOTEBOOK"
"$CONDA_PY" "$REPO/analysis_results/figure3/computation/finalize_coverage95_run.py" \
  --output-dir "$OUTPUT" --submission-json "$SUBMISSION" \
  --expected-subjects 313 --minimum-coverage 0.95
