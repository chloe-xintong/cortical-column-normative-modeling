#!/bin/bash

# Preprocess diffusion MRI data.
#
# Input:
#   Raw diffusion image in MRtrix .mif format
#
# Output:
#   Denoised diffusion image
#   Estimated noise image
#   b0 image
#   Brain-extracted b0 image
#   Brain mask

set -e

INPUT_DWI=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$OUTPUT_DIR/dwi_denoised.mif" ] || [ ! -f "$OUTPUT_DIR/noise.mif" ]; then
    dwidenoise \
        "$INPUT_DWI" \
        "$OUTPUT_DIR/dwi_denoised.mif" \
        -noise "$OUTPUT_DIR/noise.mif"
else
    echo "Denoising outputs already exist. Skipping dwidenoise."
fi

if [ ! -f "$OUTPUT_DIR/b0.mif" ]; then
    dwiextract \
        "$OUTPUT_DIR/dwi_denoised.mif" \
        "$OUTPUT_DIR/b0.mif" \
        -bzero
else
    echo "b0.mif already exists. Skipping b0 extraction."
fi

if [ ! -f "$OUTPUT_DIR/b0.nii.gz" ]; then
    mrconvert \
        "$OUTPUT_DIR/b0.mif" \
        "$OUTPUT_DIR/b0.nii.gz"
else
    echo "b0.nii.gz already exists. Skipping conversion."
fi

if [ ! -f "$OUTPUT_DIR/b0_brain_mask.nii.gz" ]; then
    PATH="$FSLDIR/bin:$PATH" bet \
        "$OUTPUT_DIR/b0.nii.gz" \
        "$OUTPUT_DIR/b0_brain" \
        -f 0.20 \
        -m
else
    echo "Brain mask already exists. Skipping BET."
fi