#!/usr/bin/env bash
# Submit the two-subject gate and full-cohort dependency graph.
#
# This script never executes Jupyter notebooks. It prepares their cached tables;
# the investigator runs the notebooks manually.

set -euo pipefail

PROJECT_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
PYTHON="${PYTHON:-python}"
RESULTS_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/qc"
O3_ROOT="/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns"
SUBJECTS_DIR="/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"
DWI_ROOT="/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi"
RUN_ROOT="$RESULTS_ROOT/full_cohort_spatial_scale"
MANIFEST="$PROJECT_DIR/manifests/full_300_subject_manifest.csv"
INVENTORY="$RUN_ROOT/full_cohort_input_inventory.csv"
MISSING="$RUN_ROOT/full_cohort_missing_inputs.csv"
METHOD_COMPARISON_MANIFEST="$PROJECT_DIR/manifests/method_comparison_available_subjects.csv"
PREFLIGHT_ROOT="$RUN_ROOT/preflight"
OPTION3_ROOT="$RUN_ROOT/option3_qc"
PAIR_DIR="$RUN_ROOT/figure4_pairs"
FIGURE4_REAL_ROOT="$RUN_ROOT/figure4_real"
FIGURE4_NULL_ROOT="$RUN_ROOT/figure4_null"
STATUS_ROOT="$RUN_ROOT/status"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
N_NULL_REPLICATES="${N_NULL_REPLICATES:-5}"

mkdir -p \
    "$RUN_ROOT" \
    "$PROJECT_DIR/logs/full_cohort" \
    "$PROJECT_DIR/logs/option3_qc" \
    "$PROJECT_DIR/logs/figure4"

"$PYTHON" "$PROJECT_DIR/qc/option3/scripts/build_full_cohort_manifest.py" \
    --o3-root "$O3_ROOT" \
    --subjects-dir "$SUBJECTS_DIR" \
    --dwi-root "$DWI_ROOT" \
    --output-manifest "$MANIFEST" \
    --inventory-output "$INVENTORY" \
    --missing-output "$MISSING"

"$PYTHON" "$PROJECT_DIR/qc/build_method_comparison_manifest.py" \
    --subject-list "$PROJECT_DIR/manifests/o2_fs_50_fsaverage_subjects.txt" \
    --o3-root "$O3_ROOT" \
    --o2-column-root "$RESULTS_ROOT/template_registration_comparison/o2_column/fsaverage" \
    --o2-fs-root "$RESULTS_ROOT/o2_fs" \
    --output "$METHOD_COMPARISON_MANIFEST"

N_SUBJECTS=$(($(wc -l < "$MANIFEST") - 1))
if ((N_SUBJECTS < 2)); then
    echo "ERROR: fewer than two input-complete subjects in $MANIFEST" >&2
    exit 1
fi
ARRAY_END=$((N_SUBJECTS - 1))

COMMON_EXPORT="ALL,PYTHON=$PYTHON,MANIFEST=$MANIFEST,SUBJECTS_DIR=$SUBJECTS_DIR"
NULL_EXPORT="$COMMON_EXPORT,PAIR_DIR=$PAIR_DIR,OUTPUT_ROOT=$FIGURE4_NULL_ROOT"
NULL_EXPORT="$NULL_EXPORT,N_NULL_REPLICATES=$N_NULL_REPLICATES"
NULL_EXPORT="$NULL_EXPORT,NULL_METHOD=interpolation_only,PSF_FWHM_MM=0"
REPORT_EXPORT="$COMMON_EXPORT,OPTION3_ROOT=$OPTION3_ROOT"
REPORT_EXPORT="$REPORT_EXPORT,FIGURE4_REAL_ROOT=$FIGURE4_REAL_ROOT"
REPORT_EXPORT="$REPORT_EXPORT,FIGURE4_NULL_ROOT=$FIGURE4_NULL_ROOT"
REPORT_EXPORT="$REPORT_EXPORT,STATUS_ROOT=$STATUS_ROOT"
REPORT_EXPORT="$REPORT_EXPORT,METHOD_COMPARISON_MANIFEST=$METHOD_COMPARISON_MANIFEST"

J0=$(sbatch --parsable \
    --export="$COMMON_EXPORT,PREFLIGHT_ROOT=$PREFLIGHT_ROOT" \
    "$PROJECT_DIR/qc/full_cohort/slurm/submit_full_cohort_preflight.sbatch")

J1=$(sbatch --parsable \
    --dependency="afterok:$J0" \
    --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
    --export="$COMMON_EXPORT,OUTPUT_ROOT=$OPTION3_ROOT" \
    "$PROJECT_DIR/qc/option3/slurm/submit_option3_qc.sbatch")

J2=$(sbatch --parsable \
    --dependency="afterok:$J0" \
    --export="$COMMON_EXPORT,PAIR_DIR=$PAIR_DIR" \
    "$PROJECT_DIR/analysis_results/figure4/slurm/submit_figure4_pairs.sbatch")

J3=$(sbatch --parsable \
    --dependency="afterok:$J2" \
    --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
    --export="$COMMON_EXPORT,PAIR_DIR=$PAIR_DIR,OUTPUT_ROOT=$FIGURE4_REAL_ROOT" \
    "$PROJECT_DIR/analysis_results/figure4/slurm/submit_figure4_real.sbatch")

J4=$(sbatch --parsable \
    --dependency="afterok:$J0:$J2" \
    --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
    --export="$NULL_EXPORT" \
    "$PROJECT_DIR/analysis_results/figure4/slurm/submit_figure4_null.sbatch")

J5=$(sbatch --parsable \
    --dependency="afterany:$J1" \
    --export="$COMMON_EXPORT,OUTPUT_ROOT=$OPTION3_ROOT" \
    "$PROJECT_DIR/qc/option3/slurm/summarize_option3_qc.sbatch")

J6=$(sbatch --parsable \
    --dependency="afterany:$J3" \
    --export="$COMMON_EXPORT,OUTPUT_ROOT=$FIGURE4_REAL_ROOT" \
    "$PROJECT_DIR/analysis_results/figure4/slurm/summarize_figure4_real.sbatch")

J7=$(sbatch --parsable \
    --dependency="afterany:$J4" \
    --export="$COMMON_EXPORT,OUTPUT_ROOT=$FIGURE4_NULL_ROOT" \
    "$PROJECT_DIR/analysis_results/figure4/slurm/summarize_figure4_null.sbatch")

J8=$(sbatch --parsable \
    --dependency="afterany:$J5:$J6:$J7" \
    --export="$REPORT_EXPORT" \
    "$PROJECT_DIR/qc/full_cohort/slurm/write_full_cohort_report.sbatch")

cat <<EOF
Submitted full cohort with $N_SUBJECTS requested subjects.
J0 preflight:          $J0
J1 O3 QC array:       $J1
J2 pair table:        $J2
J3 Figure 4 real:     $J3
J4 Figure 4 null:     $J4
J5 O3 summary:        $J5
J6 real summary:      $J6
J7 null summary:      $J7
J8 final status only: $J8

No notebook is executed by this dependency graph.
EOF
