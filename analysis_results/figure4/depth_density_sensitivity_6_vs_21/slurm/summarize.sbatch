#!/usr/bin/env bash
#SBATCH --job-name=F4Depth6Sum
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/analysis_results/figure4/depth_density_sensitivity_6_vs_21/slurm/summary_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/analysis_results/figure4/depth_density_sensitivity_6_vs_21/slurm/summary_%j.err

set -euo pipefail

REPO=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
FORMAL=/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/figure4_ico5_formal_k15
ACTIVE="$REPO/analysis_results/figure4/outputs/registration_qc_pass_320"
OUTPUT="$REPO/analysis_results/figure4/depth_density_sensitivity_6_vs_21"
export PYTHONPATH="$REPO/analysis_results/figure4/scripts:$REPO:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/matplotlib_f4_depth6_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/f4_depth6_cache_${SLURM_JOB_ID}"

"$PYTHON" "$OUTPUT/scripts/summarize.py" \
  --manifest "$REPO/manifests/figure4_registration_qc_pass_320_subjects.csv" \
  --six-root "$OUTPUT" \
  --twentyone-subject-bins "$ACTIVE/real/cohort/figure4A_ico5_k15_real_subject_bins.csv" \
  --twentyone-fits "$ACTIVE/correlation_length/cohort/figure4C_correlation_length_fits.csv" \
  --pair-file "$FORMAL/pairs/figure4_geodesic_pairs_all.parquet" \
  --split-assignment "$FORMAL/pair_qc/formal_fixed_stratified_split_assignment.csv" \
  --output-dir "$OUTPUT" \
  --bootstrap-replicates 10000
