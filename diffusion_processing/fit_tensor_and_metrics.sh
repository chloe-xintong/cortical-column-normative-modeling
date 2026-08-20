#!/usr/bin/env bash

# Fit the diffusion tensor and generate diffusion metric maps.
#
# Inputs:
#   Bias-corrected DWI in MRtrix format
#   Brain mask
#
# Outputs:
#   Diffusion tensor
#   ADC / MD
#   FA
#   AD
#   RD
#
# Usage:
#   bash fit_tensor_and_metrics.sh \
#       PREPROCESSED_DWI \
#       BRAIN_MASK \
#       OUTPUT_DIR \
#       SUBJECT_ID

set -euo pipefail

PREPROCESSED_DWI=$1
BRAIN_MASK=$2
OUTPUT_DIR=$3
SUBJECT_ID=$4

mkdir -p "$OUTPUT_DIR"

TENSOR="${OUTPUT_DIR}/${SUBJECT_ID}_07_dt.mif"

ADC="${OUTPUT_DIR}/${SUBJECT_ID}_adc.nii.gz"
FA="${OUTPUT_DIR}/${SUBJECT_ID}_fa.nii.gz"
AD="${OUTPUT_DIR}/${SUBJECT_ID}_ad.nii.gz"
RD="${OUTPUT_DIR}/${SUBJECT_ID}_rd.nii.gz"

# Fit the diffusion tensor using the gradient information
# stored in the preprocessed MRtrix image.
dwi2tensor \
    "$PREPROCESSED_DWI" \
    "$TENSOR" \
    -mask "$BRAIN_MASK" \
    -force

# Generate mean diffusivity / apparent diffusion coefficient.
tensor2metric \
    "$TENSOR" \
    -adc "$ADC" \
    -mask "$BRAIN_MASK" \
    -force

# Generate fractional anisotropy.
tensor2metric \
    "$TENSOR" \
    -fa "$FA" \
    -mask "$BRAIN_MASK" \
    -force

# Generate axial diffusivity.
tensor2metric \
    "$TENSOR" \
    -ad "$AD" \
    -mask "$BRAIN_MASK" \
    -force

# Generate radial diffusivity.
tensor2metric \
    "$TENSOR" \
    -rd "$RD" \
    -mask "$BRAIN_MASK" \
    -force

echo "Tensor fitting completed."
echo "Tensor:"
echo "$TENSOR"
echo "Diffusion metrics:"
echo "$ADC"
echo "$FA"
echo "$AD"
echo "$RD"