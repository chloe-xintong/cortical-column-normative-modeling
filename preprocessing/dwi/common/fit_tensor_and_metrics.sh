#!/usr/bin/env bash

# Optional helper to fit the diffusion tensor and generate scalar diffusion
# metrics from an already preprocessed DWI image and brain mask.
#
# The ADRC and ADD protocol-specific scripts already perform this step when
# run directly. Do not run this helper after either complete script unless a
# separate workflow intentionally needs to refit the tensor.
#
# Usage with embedded MRtrix gradients (ADRC / D-series):
#   bash fit_tensor_and_metrics.sh \
#       PREPROCESSED_DWI BRAIN_MASK OUTPUT_DIR SUBJECT_ID
#
# Usage with external FSL gradients (ADD / S-series):
#   bash fit_tensor_and_metrics.sh \
#       PREPROCESSED_DWI BRAIN_MASK OUTPUT_DIR SUBJECT_ID BVEC BVAL
#
# Processing steps:
#   1. Fit the voxel-wise diffusion tensor with dwi2tensor.
#   2. Derive FA, ADC, AD, and RD with tensor2metric.
#
# The input DWI must already have completed the cohort-specific
# preprocessing, including denoising, distortion/motion correction,
# bias-field correction, and brain-mask generation.
#
# Outputs:
#   <SUBJECT_ID>_tensor.mif
#   <SUBJECT_ID>_fa.nii.gz
#   <SUBJECT_ID>_adc.nii.gz
#   <SUBJECT_ID>_ad.nii.gz
#   <SUBJECT_ID>_rd.nii.gz

# Fit the diffusion tensor and generate diffusion metric maps.
#
# Two gradient modes are supported:
#
#   Embedded gradients:
#       Used by ADRC / D-series subjects.
#
#   External FSL bvec / bval:
#       Used by ADD / S-series subjects.
#
# Usage:
#
# Embedded gradients:
#
#   bash fit_tensor_and_metrics.sh \
#       DWI \
#       BRAIN_MASK \
#       OUTPUT_DIR \
#       SUBJECT_ID
#
# External gradients:
#
#   bash fit_tensor_and_metrics.sh \
#       DWI \
#       BRAIN_MASK \
#       OUTPUT_DIR \
#       SUBJECT_ID \
#       BVEC \
#       BVAL

set -euo pipefail


if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo "ERROR: Expected 4 or 6 arguments."
    exit 1
fi


DWI="$1"
BRAIN_MASK="$2"
OUTPUT_DIR="$3"
SUBJECT_ID="$4"

BVEC=""
BVAL=""


if [[ $# -eq 6 ]]; then
    BVEC="$5"
    BVAL="$6"
fi


for file in "$DWI" "$BRAIN_MASK"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Input file not found: $file"
        exit 1
    fi
done


if [[ $# -eq 6 ]]; then

    for file in "$BVEC" "$BVAL"; do
        if [[ ! -f "$file" ]]; then
            echo "ERROR: Gradient file not found: $file"
            exit 1
        fi
    done

fi


mkdir -p "$OUTPUT_DIR"


TENSOR="${OUTPUT_DIR}/${SUBJECT_ID}_07_dt.mif"

ADC="${OUTPUT_DIR}/${SUBJECT_ID}_adc.nii.gz"
FA="${OUTPUT_DIR}/${SUBJECT_ID}_fa.nii.gz"
AD="${OUTPUT_DIR}/${SUBJECT_ID}_ad.nii.gz"
RD="${OUTPUT_DIR}/${SUBJECT_ID}_rd.nii.gz"


echo
echo "============================================================"
echo "Diffusion tensor fitting"
echo "============================================================"
echo
echo "Subject: $SUBJECT_ID"
echo "DWI:     $DWI"
echo "Mask:    $BRAIN_MASK"


# -------------------------------------------------------------------------
# Tensor fitting
# -------------------------------------------------------------------------

if [[ $# -eq 4 ]]; then

    echo
    echo "Gradient mode: embedded"

    dwi2tensor \
        "$DWI" \
        "$TENSOR" \
        -mask "$BRAIN_MASK" \
        -force

else

    echo
    echo "Gradient mode: external FSL bvec/bval"

    dwi2tensor \
        "$DWI" \
        "$TENSOR" \
        -fslgrad "$BVEC" "$BVAL" \
        -mask "$BRAIN_MASK" \
        -force

fi


# -------------------------------------------------------------------------
# Diffusion metrics
# -------------------------------------------------------------------------

tensor2metric \
    "$TENSOR" \
    -adc "$ADC" \
    -mask "$BRAIN_MASK" \
    -force


tensor2metric \
    "$TENSOR" \
    -fa "$FA" \
    -mask "$BRAIN_MASK" \
    -force


tensor2metric \
    "$TENSOR" \
    -ad "$AD" \
    -mask "$BRAIN_MASK" \
    -force


tensor2metric \
    "$TENSOR" \
    -rd "$RD" \
    -mask "$BRAIN_MASK" \
    -force


echo
echo "============================================================"
echo "Tensor fitting completed."
echo "============================================================"
echo
echo "Tensor:"
echo "$TENSOR"
echo
echo "ADC / MD:"
echo "$ADC"
echo
echo "FA:"
echo "$FA"
echo
echo "AD:"
echo "$AD"
echo
echo "RD:"
echo "$RD"
