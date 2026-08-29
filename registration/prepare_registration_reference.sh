#!/usr/bin/env bash

# Prepare the 3D diffusion reference image used for DWI-to-T1 registration.
#
# This reproduces the registration-reference construction used in the
# historical pipeline:
#
#   4D preprocessed DWI
#       -> remove volume 0
#       -> sum remaining diffusion volumes
#       -> apply the supplied registration mask
#       -> 3D masked DWI
#
# The registration mask should already be generated during DWI preprocessing.
#
# Usage:
#
# bash registration/prepare_registration_reference.sh \
#     PREPROCESSED_DWI \
#     OUTPUT_DIR \
#     SUBJECT_ID \
#     REGISTRATION_MASK

set -euo pipefail


# -------------------------------------------------------------------------
# Check command-line arguments
# -------------------------------------------------------------------------

if [[ "$#" -ne 4 ]]; then
    echo "Usage:"
    echo "bash registration/prepare_registration_reference.sh \\"
    echo "    PREPROCESSED_DWI \\"
    echo "    OUTPUT_DIR \\"
    echo "    SUBJECT_ID \\"
    echo "    REGISTRATION_MASK"
    exit 1
fi


# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------

PREPROCESSED_DWI="$1"
OUTPUT_DIR="$2"
SUBJECT_ID="$3"
REGISTRATION_MASK="$4"


# -------------------------------------------------------------------------
# Check required input files
# -------------------------------------------------------------------------

if [[ ! -f "$PREPROCESSED_DWI" ]]; then
    echo "ERROR: Preprocessed DWI not found:"
    echo "$PREPROCESSED_DWI"
    exit 1
fi

if [[ ! -f "$REGISTRATION_MASK" ]]; then
    echo "ERROR: Registration mask not found:"
    echo "$REGISTRATION_MASK"
    exit 1
fi


# -------------------------------------------------------------------------
# Create output directory
# -------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"


# -------------------------------------------------------------------------
# Output files
# -------------------------------------------------------------------------

DWI_NO_B0="${OUTPUT_DIR}/${SUBJECT_ID}_dwi4D_no_b0.mif"
DWI_SUM="${OUTPUT_DIR}/${SUBJECT_ID}_dwi_sum.nii.gz"
DWI_MASKED="${OUTPUT_DIR}/${SUBJECT_ID}_dwi_masked.nii.gz"


# -------------------------------------------------------------------------
# Remove volume 0
#
# This intentionally follows the historical pipeline rather than removing
# volumes according to their b-values.
# -------------------------------------------------------------------------

mrconvert \
    "$PREPROCESSED_DWI" \
    -coord 3 1:end \
    "$DWI_NO_B0" \
    -force


# -------------------------------------------------------------------------
# Sum the remaining diffusion-weighted volumes into one 3D image
# -------------------------------------------------------------------------

mrmath \
    "$DWI_NO_B0" \
    sum \
    "$DWI_SUM" \
    -axis 3 \
    -force


# -------------------------------------------------------------------------
# Confirm that the summed DWI and registration mask have matching dimensions
# -------------------------------------------------------------------------

DWI_SIZE=$(mrinfo "$DWI_SUM" -size)
MASK_SIZE=$(mrinfo "$REGISTRATION_MASK" -size)

echo
echo "Summed DWI size:"
echo "$DWI_SIZE"

echo
echo "Registration mask size:"
echo "$MASK_SIZE"

if [[ "$DWI_SIZE" != "$MASK_SIZE" ]]; then
    echo
    echo "ERROR: DWI sum and registration mask dimensions do not match."
    exit 1
fi


# -------------------------------------------------------------------------
# Apply the registration mask
# -------------------------------------------------------------------------

mrcalc \
    "$DWI_SUM" \
    "$REGISTRATION_MASK" \
    -mult \
    "$DWI_MASKED" \
    -force


# -------------------------------------------------------------------------
# Verify output
# -------------------------------------------------------------------------

if [[ ! -f "$DWI_MASKED" ]]; then
    echo "ERROR: Registration DWI was not generated:"
    echo "$DWI_MASKED"
    exit 1
fi


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

echo
echo "Registration reference preparation completed."

echo
echo "Registration mask:"
echo "$REGISTRATION_MASK"

echo
echo "Summed DWI:"
echo "$DWI_SUM"

echo
echo "3D masked DWI for registration:"
echo "$DWI_MASKED"

echo
echo "Final registration DWI size:"
mrinfo "$DWI_MASKED" -size
