#!/usr/bin/env bash

# Create diffusion-space masks and a 3D DWI reference for registration.
#
# The mean-b0 BET mask is generated for QC.
# The registration reference follows the old pipeline:
#
#   4D preprocessed DWI
#       -> remove volume 0
#       -> sum remaining diffusion volumes
#       -> apply registration mask
#       -> 3D masked DWI
#
# Usage:
#
#   bash create_brain_mask.sh \
#       PREPROCESSED_DWI \
#       OUTPUT_DIR \
#       SUBJECT_ID \
#       REGISTRATION_MASK

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage:"
    echo "bash create_brain_mask.sh \\"
    echo "    PREPROCESSED_DWI \\"
    echo "    OUTPUT_DIR \\"
    echo "    SUBJECT_ID \\"
    echo "    REGISTRATION_MASK"
    exit 1
fi

PREPROCESSED_DWI="$1"
OUTPUT_DIR="$2"
SUBJECT_ID="$3"
REGISTRATION_MASK="$4"

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

mkdir -p "$OUTPUT_DIR"

B0_MIF="${OUTPUT_DIR}/${SUBJECT_ID}_b0.mif"
MEAN_B0_MIF="${OUTPUT_DIR}/${SUBJECT_ID}_mean_b0.mif"
MEAN_B0_NII="${OUTPUT_DIR}/${SUBJECT_ID}_mean_b0.nii.gz"
MEAN_B0_BRAIN="${OUTPUT_DIR}/${SUBJECT_ID}_mean_b0_brain.nii.gz"
B0_MASK="${OUTPUT_DIR}/${SUBJECT_ID}_mean_b0_brain_mask.nii.gz"

DWI_NO_B0="${OUTPUT_DIR}/${SUBJECT_ID}_dwi4D_no_b0.mif"
DWI_SUM="${OUTPUT_DIR}/${SUBJECT_ID}_dwi_sum.nii.gz"
DWI_MASKED="${OUTPUT_DIR}/${SUBJECT_ID}_dwi_masked.nii.gz"

# Extract all b0 volumes.
dwiextract \
    "$PREPROCESSED_DWI" \
    "$B0_MIF" \
    -bzero \
    -force

# Average all b0 volumes into one 3D image.
mrmath \
    "$B0_MIF" \
    mean \
    "$MEAN_B0_MIF" \
    -axis 3 \
    -force

# Convert mean b0 to NIfTI for BET.
mrconvert \
    "$MEAN_B0_MIF" \
    "$MEAN_B0_NII" \
    -force

# Generate a BET mask for QC.
PATH="$FSLDIR/bin:$PATH" bet \
    "$MEAN_B0_NII" \
    "$MEAN_B0_BRAIN" \
    -f 0.20 \
    -R \
    -m

if [[ ! -f "$B0_MASK" ]]; then
    echo "ERROR: BET mask was not generated:"
    echo "$B0_MASK"
    exit 1
fi

# Remove volume 0, matching the old pipeline.
mrconvert \
    "$PREPROCESSED_DWI" \
    -coord 3 1:end \
    "$DWI_NO_B0" \
    -force

# Sum the remaining diffusion volumes into one 3D image.
mrmath \
    "$DWI_NO_B0" \
    sum \
    "$DWI_SUM" \
    -axis 3 \
    -force

# Check that the summed DWI and registration mask have matching dimensions.
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

# Apply the registration mask to the summed DWI.
mrcalc \
    "$DWI_SUM" \
    "$REGISTRATION_MASK" \
    -mult \
    "$DWI_MASKED" \
    -force

if [[ ! -f "$DWI_MASKED" ]]; then
    echo "ERROR: Registration DWI was not generated:"
    echo "$DWI_MASKED"
    exit 1
fi

echo
echo "Brain mask and registration reference creation completed."

echo
echo "Mean b0:"
echo "$MEAN_B0_NII"

echo
echo "BET mask for QC:"
echo "$B0_MASK"

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