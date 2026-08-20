#!/usr/bin/env bash

# Preprocess diffusion MRI using the MRtrix workflow recovered
# from the original diffusion preprocessing pipeline.
#
# Steps:
#   1. Import raw DWI with bvec/bval
#   2. MP-PCA denoising
#   3. Gibbs ringing correction
#   4. Motion / eddy / susceptibility correction
#   5. ANTs-based bias field correction
#
# Usage:
#   bash preprocess_dwi.sh \
#       INPUT_DWI \
#       BVEC \
#       BVAL \
#       OUTPUT_DIR \
#       SUBJECT_ID \
#       PE_DIR \
#       READOUT_TIME \
#       SE_EPI \
#       EDDY_MASK

set -euo pipefail

if [[ $# -ne 9 ]]; then
    echo "Usage:"
    echo "  bash preprocess_dwi.sh \\"
    echo "    INPUT_DWI \\"
    echo "    BVEC \\"
    echo "    BVAL \\"
    echo "    OUTPUT_DIR \\"
    echo "    SUBJECT_ID \\"
    echo "    PE_DIR \\"
    echo "    READOUT_TIME \\"
    echo "    SE_EPI \\"
    echo "    EDDY_MASK"
    exit 1
fi

INPUT_DWI=$1
BVEC=$2
BVAL=$3
OUTPUT_DIR=$4
SUBJECT_ID=$5
PE_DIR=$6
READOUT_TIME=$7
SE_EPI=$8
EDDY_MASK=$9

# Check required input files.
for file in "$INPUT_DWI" "$BVEC" "$BVAL" "$SE_EPI" "$EDDY_MASK"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Input file not found: $file" >&2
        exit 1
    fi
done

# Check phase-encoding direction.
case "$PE_DIR" in
    AP|PA|LR|RL|IS|SI)
        ;;
    *)
        echo "ERROR: Invalid phase-encoding direction: $PE_DIR" >&2
        echo "Expected one of: AP PA LR RL IS SI" >&2
        exit 1
        ;;
esac

# Check that readout time is numeric and positive.
if ! awk -v value="$READOUT_TIME" \
    'BEGIN { exit !(value ~ /^[0-9]*\.?[0-9]+$/ && value > 0) }'; then
    echo "ERROR: Invalid readout time: $READOUT_TIME" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

DWI_01="${OUTPUT_DIR}/${SUBJECT_ID}_01_dwi_nii4D.mif"
DWI_02="${OUTPUT_DIR}/${SUBJECT_ID}_02_dwi_nii4D_denoised.mif"
DWI_03="${OUTPUT_DIR}/${SUBJECT_ID}_03_dwi_nii4D_degibbs.mif"
DWI_04="${OUTPUT_DIR}/${SUBJECT_ID}_04_dwi_nii4D_preprocessed.mif"
DWI_05="${OUTPUT_DIR}/${SUBJECT_ID}_05_dwi_nii4D_biascorrected.mif"

echo "Subject: $SUBJECT_ID"
echo "Input DWI: $INPUT_DWI"
echo "Output directory: $OUTPUT_DIR"
echo "Phase-encoding direction: $PE_DIR"
echo "Readout time: $READOUT_TIME"

# Import raw DWI and attach FSL-format gradient information.
mrconvert \
    "$INPUT_DWI" \
    "$DWI_01" \
    -fslgrad "$BVEC" "$BVAL" \
    -force

if [[ ! -s "$DWI_01" ]]; then
    echo "ERROR: MRtrix import failed." >&2
    exit 1
fi

# Check that diffusion gradients were imported.
echo "Checking imported diffusion gradients..."
mrinfo "$DWI_01" -dwgrad

# Remove random noise using MP-PCA denoising.
dwidenoise \
    "$DWI_01" \
    "$DWI_02" \
    -force

if [[ ! -s "$DWI_02" ]]; then
    echo "ERROR: Denoising output was not generated." >&2
    exit 1
fi

# Remove Gibbs ringing artifacts.
mrdegibbs \
    "$DWI_02" \
    "$DWI_03" \
    -force

if [[ ! -s "$DWI_03" ]]; then
    echo "ERROR: Gibbs correction output was not generated." >&2
    exit 1
fi

# Correct motion, eddy-current distortion, and susceptibility distortion.
# These parameters must match the acquisition protocol of the dataset.
dwifslpreproc \
    "$DWI_03" \
    "$DWI_04" \
    -pe_dir "$PE_DIR" \
    -rpe_pair \
    -se_epi "$SE_EPI" \
    -readout_time "$READOUT_TIME" \
    -eddy_options " --repol --slm=linear " \
    -eddy_mask "$EDDY_MASK" \
    -force

if [[ ! -s "$DWI_04" ]]; then
    echo "ERROR: DWI preprocessing output was not generated." >&2
    exit 1
fi

# Correct low-frequency intensity bias using ANTs through MRtrix.
dwibiascorrect ants \
    "$DWI_04" \
    "$DWI_05" \
    -force

if [[ ! -s "$DWI_05" ]]; then
    echo "ERROR: Bias-corrected DWI was not generated." >&2
    exit 1
fi

echo "DWI preprocessing completed successfully."
echo "Final preprocessed DWI:"
echo "$DWI_05"