#!/usr/bin/env bash
#SBATCH --job-name=F4Depth6
#SBATCH --partition=normal
#SBATCH --array=0-319%32
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/analysis_results/figure4/depth_density_sensitivity_6_vs_21/slurm/subject_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/analysis_results/figure4/depth_density_sensitivity_6_vs_21/slurm/subject_%A_%a.err

set -euo pipefail

REPO=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
PYTHON=/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python
FORMAL=/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/figure4_ico5_formal_k15
OUTPUT="$REPO/analysis_results/figure4/depth_density_sensitivity_6_vs_21"
export PYTHONPATH="$REPO/analysis_results/figure4/scripts:$REPO:${PYTHONPATH:-}"

"$PYTHON" "$OUTPUT/scripts/compute_subject.py" \
  --manifest "$REPO/manifests/figure4_registration_qc_pass_320_subjects.csv" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --pair-file "$FORMAL/pairs/figure4_geodesic_pairs_all.parquet" \
  --split-assignment "$FORMAL/pair_qc/formal_fixed_stratified_split_assignment.csv" \
  --output-root "$OUTPUT"
