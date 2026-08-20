#!/usr/bin/env bash

# Create a brain mask from the preprocessed diffusion image.
#
# Steps:
#   1. Extract the b0 image
#   2. Convert b0 to NIfTI format
#   3. Run FSL BET
#   4. Generate a binary brain mask
#
# Usage:
#   bash create_brain_mask.sh \
#       PREPROCESSED_DWI \
#       OUTPUT_DIR \
#       SUBJECT_ID

set -euo pipefail

PREPROCESSED_DWI=$1
OUTPUT_DIR=$2
SUBJECT_ID=$3

mkdir -p "$OUTPUT_DIR"

B0_MIF="${OUTPUT_DIR}/${SUBJECT_ID}_b0.mif"
B0_NII="${OUTPUT_DIR}/${SUBJECT_ID}_b0.nii.gz"
B0_BRAIN="${OUTPUT_DIR}/${SUBJECT_ID}_b0_brain.nii.gz"
B0_MASK="${OUTPUT_DIR}/${SUBJECT_ID}_mask.nii.gz"

# Extract the non-diffusion-weighted image.
dwiextract \
    "$PREPROCESSED_DWI" \
    "$B0_MIF" \
    -bzero \
    -force

# Convert the b0 image to NIfTI format for FSL BET.
mrconvert \
    "$B0_MIF" \
    "$B0_NII" \
    -force

# Run brain extraction on the b0 image.
# The fractional intensity threshold can be adjusted after visual QC.
PATH="$FSLDIR/bin:$PATH" bet \
    "$B0_NII" \
    "$B0_BRAIN" \
    -f 0.20 \
    -R \
    -m

# Rename the BET-generated mask to the pipeline naming convention.
BET_MASK="${OUTPUT_DIR}/${SUBJECT_ID}_b0_brain_mask.nii.gz"

if [[ ! -f "$BET_MASK" ]]; then
    echo "ERROR: BET mask was not generated: $BET_MASK" >&2
    exit 1
fi

cp "$BET_MASK" "$B0_MASK"

echo "Brain mask creation completed."
echo "B0 image:"
echo "$B0_NII"
echo "Brain mask:"
echo "$B0_MASK"