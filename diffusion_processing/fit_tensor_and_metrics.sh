#!/usr/bin/env bash

# Fit a diffusion tensor model and generate diffusion metric maps.
#
# This script expects a preprocessed 4D diffusion image together with
# FSL-format b-vectors, b-values, and a brain mask.
#
# Processing:
#   preprocessed DWI
#       -> dwi2tensor
#       -> diffusion tensor
#       -> tensor2metric
#       -> ADC/MD, FA, AD, and RD maps
#
# Inputs:
#   --dwi        Preprocessed 4D diffusion image
#   --bvec       FSL-format b-vector file
#   --bval       FSL-format b-value file
#   --mask       Diffusion-space brain mask
#   --output-dir Output directory
#
# Outputs:
#   tensor.mif
#   adc.nii.gz
#   fa.nii.gz
#   ad.nii.gz
#   rd.nii.gz
#
# Usage:
#   bash diffusion_processing/fit_tensor_and_metrics.sh \
#       --dwi <preprocessed_dwi> \
#       --bvec <bvec_file> \
#       --bval <bval_file> \
#       --mask <brain_mask> \
#       --output-dir <output_directory>

set -euo pipefail

usage() {
    echo "Usage:"
    echo "  bash diffusion_processing/fit_tensor_and_metrics.sh \\"
    echo "    --dwi <preprocessed_dwi.nii.gz|mif> \\"
    echo "    --bvec <bvec_file> \\"
    echo "    --bval <bval_file> \\"
    echo "    --mask <brain_mask.nii.gz|mif> \\"
    echo "    --output-dir <output_directory>"
}

DWI=""
BVEC=""
BVAL=""
MASK=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dwi)
            DWI="$2"
            shift 2
            ;;
        --bvec)
            BVEC="$2"
            shift 2
            ;;
        --bval)
            BVAL="$2"
            shift 2
            ;;
        --mask)
            MASK="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$DWI" || -z "$BVEC" || -z "$BVAL" || -z "$MASK" || -z "$OUTPUT_DIR" ]]; then
    usage
    exit 1
fi

for file in "$DWI" "$BVEC" "$BVAL" "$MASK"; do
    if [[ ! -f "$file" ]]; then
        echo "Input file not found: $file"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

TENSOR="$OUTPUT_DIR/tensor.mif"
ADC="$OUTPUT_DIR/adc.nii.gz"
FA="$OUTPUT_DIR/fa.nii.gz"
AD="$OUTPUT_DIR/ad.nii.gz"
RD="$OUTPUT_DIR/rd.nii.gz"

echo "Diffusion tensor fitting"
echo "DWI:        $DWI"
echo "bvec:       $BVEC"
echo "bval:       $BVAL"
echo "mask:       $MASK"
echo "output dir: $OUTPUT_DIR"

if [[ ! -f "$TENSOR" ]]; then
    echo
    echo "Fitting diffusion tensor..."

    dwi2tensor \
        "$DWI" \
        "$TENSOR" \
        -fslgrad "$BVEC" "$BVAL" \
        -mask "$MASK" \
        -force
else
    echo
    echo "Tensor already exists, skipping:"
    echo "$TENSOR"
fi

if [[ ! -f "$ADC" || ! -f "$FA" || ! -f "$AD" || ! -f "$RD" ]]; then
    echo
    echo "Generating diffusion metrics..."

    tensor2metric \
        "$TENSOR" \
        -adc "$ADC" \
        -fa "$FA" \
        -ad "$AD" \
        -rd "$RD" \
        -mask "$MASK" \
        -force
else
    echo
    echo "Diffusion metric files already exist, skipping."
fi

echo
echo "Diffusion tensor fitting complete."
echo
echo "Outputs:"
echo "  $TENSOR"
echo "  $ADC"
echo "  $FA"
echo "  $AD"
echo "  $RD"