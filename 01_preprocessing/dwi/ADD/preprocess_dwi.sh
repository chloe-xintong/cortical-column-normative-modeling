#!/usr/bin/env bash

# Preprocess ADD Diffusion MRI
#
# ## Description
#
# Preprocess one ADD/S-series DWI scan without a reverse phase-encoded b0.
#
# ## Methods
#
# 1. Import FSL gradients and perform MP-PCA denoising and degibbs correction.
# 2. Run eddy correction with `dwifslpreproc -rpe_none`.
# 3. Create a corrected mean-b0 BET mask and apply ANTs bias correction.
# 4. Fit the tensor and derive ADC/MD, FA, AD, and RD.
#
# ## Inputs
#
# - Raw 4D DWI and FSL bvec/bval files
# - Output directory and ADD subject ID
#
# ## Outputs
#
# - Bias-corrected DWI and corrected mean-b0 brain mask
# - Tensor image and `<SUBJECT_ID>_{adc,fa,ad,rd}.nii.gz`
#
# ## Usage
#
# bash 01_preprocessing/dwi/ADD/preprocess_dwi.sh \
#     DWI BVEC BVAL OUTPUT_DIR SUBJECT_ID
#
# ## Notes
#
# - The current ADD protocol uses PA encoding and readout time 0.119063 s.
# - No topup susceptibility correction is performed because reverse b0 is absent.

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage: preprocess_dwi.sh DWI BVEC BVAL OUTPUT_DIR SUBJECT_ID" >&2
    exit 1
fi

DWI="$1"
BVEC="$2"
BVAL="$3"
OUTPUT_DIR="$4"
SUBJECT_ID="$5"
PE_DIR="PA"
READOUT_TIME="0.119063"

for file in "$DWI" "$BVEC" "$BVAL"; do
    [[ -f "$file" ]] || { echo "ERROR: Missing input: $file" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR/qc" "$OUTPUT_DIR/eddy_scratch"
RAW_MIF="$OUTPUT_DIR/${SUBJECT_ID}_01_raw.mif"
DENOISED_MIF="$OUTPUT_DIR/${SUBJECT_ID}_02_denoised.mif"
DEGIBBS_MIF="$OUTPUT_DIR/${SUBJECT_ID}_03_degibbs.mif"
PREPROCESSED_MIF="$OUTPUT_DIR/${SUBJECT_ID}_04_preprocessed.mif"
BIASCORRECTED_MIF="$OUTPUT_DIR/${SUBJECT_ID}_05_biascorrected.mif"
NOISE_MIF="$OUTPUT_DIR/qc/${SUBJECT_ID}_noise.mif"
RESIDUAL_MIF="$OUTPUT_DIR/qc/${SUBJECT_ID}_denoising_residual.mif"
CORRECTED_MEAN_B0="$OUTPUT_DIR/${SUBJECT_ID}_corrected_mean_b0.nii.gz"
BRAIN_BASE="$OUTPUT_DIR/${SUBJECT_ID}_corrected_mean_b0_brain"
BRAIN_MASK="$OUTPUT_DIR/${SUBJECT_ID}_corrected_mean_b0_brain_mask.nii.gz"
TENSOR_MIF="$OUTPUT_DIR/${SUBJECT_ID}_tensor.mif"
FA="$OUTPUT_DIR/${SUBJECT_ID}_fa.nii.gz"
MD="$OUTPUT_DIR/${SUBJECT_ID}_adc.nii.gz"
AD="$OUTPUT_DIR/${SUBJECT_ID}_ad.nii.gz"
RD="$OUTPUT_DIR/${SUBJECT_ID}_rd.nii.gz"

echo "Subject: $SUBJECT_ID"
echo "Phase encoding: $PE_DIR"
echo "Total readout time: $READOUT_TIME seconds"

mrconvert "$DWI" "$RAW_MIF" -fslgrad "$BVEC" "$BVAL" -force
mrinfo "$RAW_MIF" -dwgrad
dwidenoise "$RAW_MIF" "$DENOISED_MIF" -noise "$NOISE_MIF" -force
mrcalc "$RAW_MIF" "$DENOISED_MIF" -subtract "$RESIDUAL_MIF" -force
mrdegibbs "$DENOISED_MIF" "$DEGIBBS_MIF" -force

# No reverse b0 and no topup; eddy uses the single-acquisition rpe_none mode.
dwifslpreproc "$DEGIBBS_MIF" "$PREPROCESSED_MIF" \
    -pe_dir "$PE_DIR" -rpe_none -readout_time "$READOUT_TIME" \
    -eddy_options " --repol --slm=linear " \
    -scratch "$OUTPUT_DIR/eddy_scratch" -nocleanup -force

dwiextract "$PREPROCESSED_MIF" -bzero - | mrmath - mean "$CORRECTED_MEAN_B0" -axis 3 -force
bet "$CORRECTED_MEAN_B0" "$BRAIN_BASE" -m -f 0.30
[[ -s "$BRAIN_MASK" ]] || { echo "ERROR: Missing brain mask" >&2; exit 1; }

dwibiascorrect ants "$PREPROCESSED_MIF" "$BIASCORRECTED_MIF" -mask "$BRAIN_MASK" -force
dwi2tensor "$BIASCORRECTED_MIF" "$TENSOR_MIF" -mask "$BRAIN_MASK" -force
tensor2metric "$TENSOR_MIF" -fa "$FA" -adc "$MD" -ad "$AD" -rd "$RD" -mask "$BRAIN_MASK" -force

for output in "$PREPROCESSED_MIF" "$CORRECTED_MEAN_B0" "$BRAIN_MASK" "$BIASCORRECTED_MIF" "$TENSOR_MIF" "$FA" "$MD" "$AD" "$RD"; do
    [[ -s "$output" ]] || { echo "ERROR: Missing or empty output: $output" >&2; exit 1; }
done

rm -f "$RAW_MIF" "$DENOISED_MIF" "$DEGIBBS_MIF" "${BRAIN_BASE}.nii.gz" "$NOISE_MIF" "$RESIDUAL_MIF"
rm -rf "$OUTPUT_DIR/eddy_scratch" "$OUTPUT_DIR/qc"
echo "ADD preprocessing completed successfully."
