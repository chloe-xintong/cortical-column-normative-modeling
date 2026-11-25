#!/bin/bash
#SBATCH --job-name=fig6_regional
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=analysis_results/figure6/logs/regional_pilot_%j.out
#SBATCH --error=analysis_results/figure6/logs/regional_pilot_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PILOT_ROOT="${REPO_ROOT}/analysis_results/figure6"
FIGURE5_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs"
FIGURE3_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/figure3_depth_thickness"
: "${FIG6_PYTHON:?Set FIG6_PYTHON to the Python executable from environment.yml}"

export XDG_CACHE_HOME="${TMPDIR:-/tmp}/figure6_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/figure6_mpl_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/prepare_pilot.py" \
  --subject-order "${FIGURE5_ROOT}/features/subject_order.csv" \
  --cohort-manifest "${REPO_ROOT}/analysis_results/figure5/cohort_selection/outputs/cohort_manifest.csv" \
  --regional-profile "${FIGURE5_ROOT}/features/matrices/X_regional_profile.npy" \
  --regional-thickness "${FIGURE5_ROOT}/features/matrices/X_regional_thickness.npy" \
  --profile-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_profile.csv" \
  --thickness-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_thickness.csv" \
  --figure5-folds "${FIGURE5_ROOT}/folds/outer_folds.json" \
  --figure3-ranking "${FIGURE3_ROOT}/tables/figure3A_region_thickness_ranking.csv" \
  --output-dir "${PILOT_ROOT}"

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/run_models.py" \
  --subject-order "${FIGURE5_ROOT}/features/subject_order.csv" \
  --cohort-audit "${PILOT_ROOT}/cohort_audit.csv" \
  --cohort-manifest "${REPO_ROOT}/analysis_results/figure5/cohort_selection/outputs/cohort_manifest.csv" \
  --regional-profile "${FIGURE5_ROOT}/features/matrices/X_regional_profile.npy" \
  --regional-thickness "${FIGURE5_ROOT}/features/matrices/X_regional_thickness.npy" \
  --profile-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_profile.csv" \
  --thickness-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_thickness.csv" \
  --folds "${PILOT_ROOT}/fold_assignments.json" \
  --output-dir "${PILOT_ROOT}" --regions all --n-jobs "${SLURM_CPUS_PER_TASK}"

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/summarize_pilot.py" \
  --output-dir "${PILOT_ROOT}" --n-bootstrap 10000 --bootstrap-seed 20261107

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/diagnose_selection.py" --output-dir "${PILOT_ROOT}"

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/fit_selected_centiles.py" \
  --subject-order "${FIGURE5_ROOT}/features/subject_order.csv" \
  --cohort-audit "${PILOT_ROOT}/cohort_audit.csv" \
  --cohort-manifest "${REPO_ROOT}/analysis_results/figure5/cohort_selection/outputs/cohort_manifest.csv" \
  --regional-profile "${FIGURE5_ROOT}/features/matrices/X_regional_profile.npy" \
  --regional-thickness "${FIGURE5_ROOT}/features/matrices/X_regional_thickness.npy" \
  --profile-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_profile.csv" \
  --thickness-metadata "${FIGURE5_ROOT}/features/feature_metadata/features_regional_thickness.csv" \
  --figure3-ranking "${FIGURE3_ROOT}/tables/figure3A_region_thickness_ranking.csv" \
  --output-dir "${PILOT_ROOT}" --selected-model M2

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/plot_pilot.py" --output-dir "${PILOT_ROOT}"

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/build_figure6_notebooks.py" --figure6-dir "${PILOT_ROOT}"

"${FIG6_PYTHON}" "${PILOT_ROOT}/scripts/build_artifact_manifest.py" --output-dir "${PILOT_ROOT}"
