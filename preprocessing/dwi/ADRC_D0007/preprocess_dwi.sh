#!/usr/bin/env bash

# Preprocess diffusion MRI using MRtrix/FSL/ANTs.
#
# Steps:
#   1. Import raw DWI with bvec/bval
#   2. MP-PCA denoising
#   3. Gibbs ringing correction
#   4. Extract forward b0
#   5. Build forward/reverse SE-EPI pair
#   6. Create eddy brain mask
#   7. Motion / eddy / susceptibility correction
#   8. ANTs-based bias field correction
#
# Usage:
#   bash preprocess_dwi.sh \
#       INPUT_DWI \
#       BVEC \
#       BVAL \
#       REVERSE_PHASE \
#       OUTPUT_DIR \
#       SUBJECT_ID \
#       PE_DIR \
#       READOUT_TIME

set -o pipefail

if [[ $# -ne 8 ]]; then
    echo "Usage:"
    echo "  bash preprocess_dwi.sh \\"
    echo "    INPUT_DWI \\"
    echo "    BVEC \\"
    echo "    BVAL \\"
    echo "    REVERSE_PHASE \\"
    echo "    OUTPUT_DIR \\"
    echo "    SUBJECT_ID \\"
    echo "    PE_DIR \\"
    echo "    READOUT_TIME"
    exit 1
fi

INPUT_DWI=$1
BVEC=$2
BVAL=$3
REVERSE_PHASE=$4
OUTPUT_DIR=$5
SUBJECT_ID=$6
PE_DIR=$7
READOUT_TIME=$8

for file in "$INPUT_DWI" "$BVEC" "$BVAL" "$REVERSE_PHASE"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Input file not found: $file" >&2
        exit 1
    fi
done

case "$PE_DIR" in
    AP|PA|LR|RL|IS|SI)
        ;;
    *)
        echo "ERROR: Invalid phase-encoding direction: $PE_DIR" >&2
        echo "Expected one of: AP PA LR RL IS SI" >&2
        exit 1
        ;;
esac

if ! awk -v value="$READOUT_TIME" \
    'BEGIN { exit !(value ~ /^[0-9]*\.?[0-9]+$/ && value > 0) }'; then
    echo "ERROR: Invalid readout time: $READOUT_TIME" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

DWI_01="${OUTPUT_DIR}/${SUBJECT_ID}_01_dwi_raw.mif"
DWI_02="${OUTPUT_DIR}/${SUBJECT_ID}_02_dwi_denoised.mif"
DWI_03="${OUTPUT_DIR}/${SUBJECT_ID}_03_dwi_degibbs.mif"
DWI_04="${OUTPUT_DIR}/${SUBJECT_ID}_04_dwi_preprocessed.mif"
DWI_05="${OUTPUT_DIR}/${SUBJECT_ID}_05_dwi_biascorrected.mif"

FORWARD_B0="${OUTPUT_DIR}/${SUBJECT_ID}_forward_b0.nii.gz"
SE_EPI="${OUTPUT_DIR}/${SUBJECT_ID}_rev_seepi_pair.nii.gz"

MEAN_B0="${OUTPUT_DIR}/${SUBJECT_ID}_mean_b0.nii.gz"
EDDY_MASK="${OUTPUT_DIR}/${SUBJECT_ID}_eddy_mask.nii.gz"

echo "Subject: $SUBJECT_ID"
echo "Input DWI: $INPUT_DWI"
echo "Reverse phase: $REVERSE_PHASE"
echo "Output directory: $OUTPUT_DIR"
echo "Phase-encoding direction: $PE_DIR"
echo "Readout time: $READOUT_TIME"

# Import raw DWI and attach gradients.
mrconvert \
    "$INPUT_DWI" \
    "$DWI_01" \
    -fslgrad "$BVEC" "$BVAL" \
    -force

echo "Checking imported diffusion gradients..."
mrinfo "$DWI_01" -dwgrad

# MP-PCA denoising.
dwidenoise \
    "$DWI_01" \
    "$DWI_02" \
    -force

# Gibbs ringing correction.
mrdegibbs \
    "$DWI_02" \
    "$DWI_03" \
    -force

# Extract all b0 volumes and compute their mean.
dwiextract \
    "$DWI_03" \
    -bzero \
    - \
    | mrmath \
        - \
        mean \
        "$FORWARD_B0" \
        -axis 3 \
        -force

# Combine forward and reverse phase-encoded b0 images.
mrcat \
    "$FORWARD_B0" \
    "$REVERSE_PHASE" \
    "$SE_EPI" \
    -axis 3 \
    -force

# Create a brain mask for eddy.
dwiextract \
    "$DWI_03" \
    -bzero \
    - \
    | mrmath \
        - \
        mean \
        "$MEAN_B0" \
        -axis 3 \
        -force

bet \
    "$MEAN_B0" \
    "${OUTPUT_DIR}/${SUBJECT_ID}_mean_b0_brain" \
    -m \
    -f 0.3

BET_MASK="${OUTPUT_DIR}/${SUBJECT_ID}_mean_b0_brain_mask.nii.gz"

if [[ ! -f "$BET_MASK" ]]; then
    echo "ERROR: BET mask was not generated." >&2
    exit 1
fi

cp "$BET_MASK" "$EDDY_MASK"

# Motion, eddy-current, and susceptibility correction.
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

# ANTs-based bias-field correction.
dwibiascorrect ants \
    "$DWI_04" \
    "$DWI_05" \
    -force

echo
echo "DWI preprocessing completed successfully."
echo "Final preprocessed DWI:"
echo "$DWI_05"
echo
echo "SE-EPI pair:"
echo "$SE_EPI"
echo
echo "Eddy mask:"
echo "$EDDY_MASK"