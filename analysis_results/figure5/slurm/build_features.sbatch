#!/usr/bin/env bash
#SBATCH --job-name=F5Features
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/figure5/%x_%j.err

set -euo pipefail
PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
COHORT="${COHORT:?Set COHORT}"
PROFILE_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns"
NATIVE_FS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"
TEMPLATE_FS_ROOT="/home/apps/freesurfer/subjects"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/template" "$OUTPUT_ROOT/features"
REFERENCE_SUBJECT=$("$PYTHON" -c 'import pandas as pd,sys; print(pd.read_csv(sys.argv[1], usecols=["subject_id"]).iloc[0,0])' "$COHORT")
"$PYTHON" analysis_results/figure5/scripts/audit_template_nesting.py \
  --freesurfer-subjects-dir "$TEMPLATE_FS_ROOT" \
  --profile-root "$PROFILE_ROOT" \
  --reference-subject "$REFERENCE_SUBJECT" \
  --output-dir "$OUTPUT_ROOT/template"
"$PYTHON" analysis_results/figure5/scripts/build_features.py \
  --cohort "$COHORT" \
  --profile-root "$PROFILE_ROOT" \
  --native-freesurfer-root "$NATIVE_FS_ROOT" \
  --vertex-order-dir "$OUTPUT_ROOT/template" \
  --output-dir "$OUTPUT_ROOT/features"
