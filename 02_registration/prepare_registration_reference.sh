#!/usr/bin/env bash

# Prepare the DWI Registration Reference
#
# ## Description
#
# Create the masked 3D diffusion image used as the bbregister moving image.
#
# ## Methods
#
# 1. Remove volume index 0 from the preprocessed 4D DWI.
# 2. Sum all remaining volumes into one 3D image.
# 3. Confirm that the image and registration mask dimensions match.
# 4. Apply the mask to produce the final registration reference.
#
# ## Inputs
#
# - Preprocessed 4D DWI and matching registration mask
# - Output directory and subject ID
#
# ## Outputs
#
# - DWI without volume 0, summed DWI, and `<SUBJECT_ID>_dwi_masked.nii.gz`
#
# ## Usage
#
# bash 02_registration/prepare_registration_reference.sh \
#     PREPROCESSED_DWI OUTPUT_DIR SUBJECT_ID REGISTRATION_MASK
#
# ## Notes
#
# - Volume removal follows the historical index-based procedure, not b-values.
# - The supplied mask must already be in the DWI grid.

set -euo pipefail


# -------------------------------------------------------------------------
# Check command-line arguments
# -------------------------------------------------------------------------

if [[ "$#" -ne 4 ]]; then
    echo "Usage:"
    echo "bash 02_registration/prepare_registration_reference.sh \\"
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
