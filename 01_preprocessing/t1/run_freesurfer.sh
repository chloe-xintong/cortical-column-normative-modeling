#!/usr/bin/env bash

# Reconstruct Native Cortical Surfaces with FreeSurfer
#
# ## Description
#
# Run `recon-all` to generate the native cortical surfaces required for column
# construction.
#
# ## Methods
#
# 1. Validate the FreeSurfer environment and T1 input.
# 2. Run the complete FreeSurfer reconstruction for one subject.
# 3. Verify the subject output directory.
#
# ## Inputs
#
# - Subject ID, T1-weighted image, and FreeSurfer SUBJECTS_DIR
# - Optional OpenMP thread count
#
# ## Outputs
#
# - FreeSurfer subject directory containing white, pial, thickness, sphere.reg,
#   annotation, and orig.mgz files
#
# ## Usage
#
# bash 01_preprocessing/t1/run_freesurfer.sh \
#     SUBJECT_ID T1_IMAGE SUBJECTS_DIR [THREADS]
#
# ## Notes
#
# - FreeSurfer must be installed and sourced before execution.
# - Existing subject data in the target SUBJECTS_DIR are not removed.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage:"
    echo "  $0 <subject_id> <t1_image> <subjects_dir> [threads]"
    exit 1
fi

SUBJECT_ID="$1"
T1_IMAGE="$2"
SUBJECTS_DIR_INPUT="$3"
THREADS="${4:-8}"

# Check FreeSurfer environment
if [ -z "${FREESURFER_HOME:-}" ]; then
    echo "Error: FreeSurfer environment is not configured."
    echo "Run this script on a system with FreeSurfer installed and sourced."
    exit 1
fi

if ! command -v recon-all >/dev/null 2>&1; then
    echo "Error: recon-all was not found."
    exit 1
fi

# Check input T1
if [ ! -f "$T1_IMAGE" ]; then
    echo "Error: T1 image does not exist:"
    echo "$T1_IMAGE"
    exit 1
fi

mkdir -p "$SUBJECTS_DIR_INPUT"

export SUBJECTS_DIR="$SUBJECTS_DIR_INPUT"

SUBJECT_DIR="${SUBJECTS_DIR}/${SUBJECT_ID}"
WMPARC="${SUBJECT_DIR}/mri/wmparc.mgz"
ORIG="${SUBJECT_DIR}/mri/orig/001.mgz"

echo "Subject ID:   $SUBJECT_ID"
echo "T1 image:     $T1_IMAGE"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "Threads:      $THREADS"

# Skip completed reconstruction
if [ -f "$WMPARC" ]; then
    echo "FreeSurfer reconstruction already appears complete."
    echo "Found:"
    echo "$WMPARC"
    exit 0
fi

# Resume an existing reconstruction
if [ -f "$ORIG" ]; then
    echo "Existing FreeSurfer reconstruction detected."
    echo "Resuming recon-all."

    recon-all \
        -s "$SUBJECT_ID" \
        -sd "$SUBJECTS_DIR" \
        -all \
        -parallel \
        -openmp "$THREADS"

# Start a new reconstruction
else
    echo "Starting new FreeSurfer reconstruction."

    recon-all \
        -s "$SUBJECT_ID" \
        -i "$T1_IMAGE" \
        -sd "$SUBJECTS_DIR" \
        -all \
        -parallel \
        -openmp "$THREADS"
fi

echo "FreeSurfer reconstruction finished."
echo "Output:"
echo "$SUBJECT_DIR"
