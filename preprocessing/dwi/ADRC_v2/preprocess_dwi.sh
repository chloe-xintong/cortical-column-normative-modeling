#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
    echo "Usage:"
    echo "bash preprocess_dwi.sh DWI BVEC BVAL REVERSE_B0 OUTPUT_DIR SUBJECT_ID PE_DIR READOUT_TIME"
    exit 1
fi

DWI="$1"
BVEC="$2"
BVAL="$3"
REVERSE_B0="$4"
OUTPUT_DIR="$5"
SUBJECT_ID="$6"
PE_DIR="$7"
READOUT_TIME="$8"

for file in "$DWI" "$BVEC" "$BVAL" "$REVERSE_B0"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing input: $file" >&2
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

mkdir -p "$OUTPUT_DIR/qc"
mkdir -p "$OUTPUT_DIR/eddy_scratch"

RAW_MIF="$OUTPUT_DIR/${SUBJECT_ID}_01_raw.mif"
DENOISED_MIF="$OUTPUT_DIR/${SUBJECT_ID}_02_denoised.mif"
DEGIBBS_MIF="$OUTPUT_DIR/${SUBJECT_ID}_03_degibbs.mif"
PREPROCESSED_MIF="$OUTPUT_DIR/${SUBJECT_ID}_04_preprocessed.mif"
BIASCORRECTED_MIF="$OUTPUT_DIR/${SUBJECT_ID}_05_biascorrected.mif"

NOISE_MIF="$OUTPUT_DIR/qc/${SUBJECT_ID}_noise.mif"
RESIDUAL_MIF="$OUTPUT_DIR/qc/${SUBJECT_ID}_denoising_residual.mif"

FORWARD_B0="$OUTPUT_DIR/${SUBJECT_ID}_forward_b0.nii.gz"
SE_EPI="$OUTPUT_DIR/${SUBJECT_ID}_se_epi_pair.nii.gz"

CORRECTED_MEAN_B0="$OUTPUT_DIR/${SUBJECT_ID}_corrected_mean_b0.nii.gz"
BRAIN_BASE="$OUTPUT_DIR/${SUBJECT_ID}_corrected_mean_b0_brain"
BRAIN_MASK="$OUTPUT_DIR/${SUBJECT_ID}_corrected_mean_b0_brain_mask.nii.gz"

TENSOR_MIF="$OUTPUT_DIR/${SUBJECT_ID}_tensor.mif"
FA="$OUTPUT_DIR/${SUBJECT_ID}_fa.nii.gz"
MD="$OUTPUT_DIR/${SUBJECT_ID}_md.nii.gz"
AD="$OUTPUT_DIR/${SUBJECT_ID}_ad.nii.gz"
RD="$OUTPUT_DIR/${SUBJECT_ID}_rd.nii.gz"

echo "Subject: $SUBJECT_ID"
echo "Input DWI: $DWI"
echo "Reverse b0: $REVERSE_B0"
echo "Output: $OUTPUT_DIR"
echo "Phase-encoding direction: $PE_DIR"
echo "Readout time: $READOUT_TIME"

mrconvert \
    "$DWI" \
    "$RAW_MIF" \
    -fslgrad "$BVEC" "$BVAL" \
    -force

echo "Checking imported diffusion gradients..."

mrinfo \
    "$RAW_MIF" \
    -dwgrad

dwidenoise \
    "$RAW_MIF" \
    "$DENOISED_MIF" \
    -noise "$NOISE_MIF" \
    -force

mrcalc \
    "$RAW_MIF" \
    "$DENOISED_MIF" \
    -subtract \
    "$RESIDUAL_MIF" \
    -force

mrdegibbs \
    "$DENOISED_MIF" \
    "$DEGIBBS_MIF" \
    -force

dwiextract \
    "$DEGIBBS_MIF" \
    -bzero \
    - \
    | mrmath \
        - \
        mean \
        "$FORWARD_B0" \
        -axis 3 \
        -force

mrcat \
    "$FORWARD_B0" \
    "$REVERSE_B0" \
    "$SE_EPI" \
    -axis 3 \
    -force

dwifslpreproc \
    "$DEGIBBS_MIF" \
    "$PREPROCESSED_MIF" \
    -pe_dir "$PE_DIR" \
    -rpe_pair \
    -se_epi "$SE_EPI" \
    -readout_time "$READOUT_TIME" \
    -eddy_options " --repol --slm=linear " \
    -scratch "$OUTPUT_DIR/eddy_scratch" \
    -nocleanup \
    -force

dwiextract \
    "$PREPROCESSED_MIF" \
    -bzero \
    - \
    | mrmath \
        - \
        mean \
        "$CORRECTED_MEAN_B0" \
        -axis 3 \
        -force

bet \
    "$CORRECTED_MEAN_B0" \
    "$BRAIN_BASE" \
    -m \
    -f 0.30

if [[ ! -s "$BRAIN_MASK" ]]; then
    echo "ERROR: Corrected mean-b0 mask was not generated: $BRAIN_MASK" >&2
    exit 1
fi

dwibiascorrect ants \
    "$PREPROCESSED_MIF" \
    "$BIASCORRECTED_MIF" \
    -mask "$BRAIN_MASK" \
    -force

dwi2tensor \
    "$BIASCORRECTED_MIF" \
    "$TENSOR_MIF" \
    -mask "$BRAIN_MASK" \
    -force

tensor2metric \
    "$TENSOR_MIF" \
    -fa "$FA" \
    -adc "$MD" \
    -ad "$AD" \
    -rd "$RD" \
    -mask "$BRAIN_MASK" \
    -force

for output in \
    "$NOISE_MIF" \
    "$RESIDUAL_MIF" \
    "$PREPROCESSED_MIF" \
    "$CORRECTED_MEAN_B0" \
    "$BRAIN_MASK" \
    "$BIASCORRECTED_MIF" \
    "$TENSOR_MIF" \
    "$FA" \
    "$MD" \
    "$AD" \
    "$RD"
do
    if [[ ! -s "$output" ]]; then
        echo "ERROR: Missing or empty output: $output" >&2
        exit 1
    fi
done

# Keep only the production V2 outputs after validation.  The raw, denoised,
# degibbs, forward-b0, reverse-PE construction, BET brain image, Eddy scratch,
# and denoising-QC files are intermediate/QC products and are intentionally
# removed from the production output directory.
rm -f \
    "$RAW_MIF" \
    "$DENOISED_MIF" \
    "$DEGIBBS_MIF" \
    "$FORWARD_B0" \
    "$SE_EPI" \
    "${BRAIN_BASE}.nii.gz" \
    "$NOISE_MIF" \
    "$RESIDUAL_MIF"
rm -rf \
    "$OUTPUT_DIR/eddy_scratch" \
    "$OUTPUT_DIR/qc"

echo
echo "V2 preprocessing completed successfully."
echo "Noise map: $NOISE_MIF"
echo "Denoising residual: $RESIDUAL_MIF"
echo "Corrected mean b0: $CORRECTED_MEAN_B0"
echo "Brain mask: $BRAIN_MASK"
echo "Eddy scratch: $OUTPUT_DIR/eddy_scratch"
echo "Bias-corrected DWI: $BIASCORRECTED_MIF"
echo "FA: $FA"
echo "MD: $MD"
echo "AD: $AD"
echo "RD: $RD"
