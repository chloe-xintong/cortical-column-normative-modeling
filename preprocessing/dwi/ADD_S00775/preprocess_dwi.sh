#!/usr/bin/env bash

# S00775 / MUSE cohort diffusion preprocessing.
#
# Reconstructed from the original AD Decode / DTC workflow.
#
# Steps:
#   1. Extract mean b=1000 DWI for BET mask generation
#   2. Generate diffusion-space brain mask with BET
#   3. Mask the raw 4D DWI
#   4. Historical DTC MP-PCA denoising
#   5. Volume-wise ANTs affine registration to volume 0
#
# Usage:
#
# bash preprocess_dwi.sh \
#     INPUT_DWI \
#     BVEC \
#     BVAL \
#     OUTPUT_DIR \
#     SUBJECT_ID
#
# Optional:
#
# BET_F=0.40 bash preprocess_dwi.sh ...
#
# Default BET threshold is the historical value:
#   BET_F=0.50

set -o pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage:"
    echo "  bash preprocess_dwi.sh INPUT_DWI BVEC BVAL OUTPUT_DIR SUBJECT_ID"
    exit 1
fi


INPUT_DWI="$1"
BVEC="$2"
BVAL="$3"
OUTPUT_DIR="$4"
SUBJECT_ID="$5"

BET_F="${BET_F:-0.50}"

DTC_ROOT="${DTC_ROOT:-/mnt/newStor/paros/paros_WORK/jacques/gitfolder/DTC}"


# -------------------------------------------------------------------------
# Check inputs
# -------------------------------------------------------------------------

for file in "$INPUT_DWI" "$BVEC" "$BVAL"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Input file not found:"
        echo "$file"
        exit 1
    fi
done


for command in \
    select_dwi_vols \
    bet \
    fslmaths \
    python
do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $command"
        exit 1
    fi
done


mkdir -p "$OUTPUT_DIR"


# -------------------------------------------------------------------------
# Output names
# -------------------------------------------------------------------------

DWI_B1000="${OUTPUT_DIR}/${SUBJECT_ID}_01_b1000_mean.nii.gz"

BET_BASE="${OUTPUT_DIR}/${SUBJECT_ID}_02_dwi_brain"
BRAIN_MASK="${OUTPUT_DIR}/${SUBJECT_ID}_02_dwi_brain_mask.nii.gz"

MASKED_DWI="${OUTPUT_DIR}/${SUBJECT_ID}_03_dwi_masked_int16.nii.gz"

DENOISED_DWI="${OUTPUT_DIR}/${SUBJECT_ID}_04_dwi_mpca.nii.gz"

COREG_DIR="${OUTPUT_DIR}/coregistration"

FINAL_DWI="${OUTPUT_DIR}/${SUBJECT_ID}_05_subjspace_coreg.nii.gz"


echo
echo "============================================================"
echo "S00775 / MUSE diffusion preprocessing"
echo "============================================================"
echo
echo "Subject:       $SUBJECT_ID"
echo "Input DWI:     $INPUT_DWI"
echo "bvec:          $BVEC"
echo "bval:          $BVAL"
echo "BET -f:        $BET_F"
echo "Output dir:    $OUTPUT_DIR"


# -------------------------------------------------------------------------
# Step 1
# Extract b=1000 volumes and generate mean image for BET.
#
# Historical command:
#
# select_dwi_vols raw_nii bvals raw_dwi 1000 -m
# -------------------------------------------------------------------------

echo
echo "[1/4] Creating mean b=1000 DWI..."

select_dwi_vols \
    "$INPUT_DWI" \
    "$BVAL" \
    "$DWI_B1000" \
    1000 \
    -m

if [[ $? -ne 0 ]]; then
    echo "ERROR: select_dwi_vols failed."
    exit 1
fi


# -------------------------------------------------------------------------
# Step 2
# BET brain mask.
#
# Historical command:
#
# bet raw_dwi tmp -m -n -R
#
# Historical default:
#   -f 0.50
# -------------------------------------------------------------------------

echo
echo "[2/4] Generating diffusion-space BET mask..."

bet \
    "$DWI_B1000" \
    "$BET_BASE" \
    -m \
    -n \
    -R \
    -f "$BET_F"

if [[ $? -ne 0 || ! -f "$BRAIN_MASK" ]]; then
    echo "ERROR: BET brain mask generation failed."
    exit 1
fi


# -------------------------------------------------------------------------
# Step 3
# Apply brain mask to raw 4D DWI.
#
# Historical command:
#
# fslmaths raw_nii -mas tmp_mask masked_nii -odt input
#
# "-odt input" preserves the original int16 datatype.
# -------------------------------------------------------------------------

echo
echo "[3/4] Applying diffusion brain mask..."

fslmaths \
    "$INPUT_DWI" \
    -mas "$BRAIN_MASK" \
    "$MASKED_DWI" \
    -odt input

if [[ $? -ne 0 ]]; then
    echo "ERROR: DWI masking failed."
    exit 1
fi


# -------------------------------------------------------------------------
# Step 4
# Historical DTC MP-PCA denoising.
#
# Parameters:
#
#   patch_radius = 2
#   pca_method   = eig
#   processes    = 1
# -------------------------------------------------------------------------

echo
echo "[4/4] Running historical DTC MP-PCA denoising..."

DTC_ROOT="$DTC_ROOT" \
MASKED_DWI="$MASKED_DWI" \
DENOISED_DWI="$DENOISED_DWI" \
python - <<'PY'

import os
import sys

from dipy.io.image import load_nifti, save_nifti

sys.path.insert(0, os.environ["DTC_ROOT"])

from diff_handlers.denoise_processes import mppca

input_file = os.environ["MASKED_DWI"]
output_file = os.environ["DENOISED_DWI"]

data, affine, _ = load_nifti(
    input_file,
    return_voxsize=True
)

print("Input shape:", data.shape)
print("Input dtype:", data.dtype)

denoised = mppca(
    data,
    mask=None,
    patch_radius=2,
    pca_method="eig",
    return_sigma=False,
    out_dtype=None,
    processes=1,
    verbose=True
)

save_nifti(
    output_file,
    denoised,
    affine
)

print("Saved:", output_file)

PY

if [[ $? -ne 0 || ! -f "$DENOISED_DWI" ]]; then
    echo "ERROR: MP-PCA denoising failed."
    exit 1
fi


# -------------------------------------------------------------------------
# Step 5
# Volume-wise affine registration.
#
# Historical workflow:
#
#   volume 0 = registration target
#   volumes 1...N -> volume 0
#   antsRegistration Affine
#   antsApplyTransforms -n Linear
#
# Implemented in a separate script.
# -------------------------------------------------------------------------

mkdir -p "$COREG_DIR"

echo
echo "============================================================"
echo "S00775 preprocessing stage 1 completed."
echo "============================================================"
echo
echo "Brain mask:"
echo "$BRAIN_MASK"
echo
echo "MP-PCA DWI:"
echo "$DENOISED_DWI"
echo
echo "Next step:"
echo "  preprocessing/dwi/S00775/coregister_4d_affine.sh"
echo
echo "Expected final output:"
echo "$FINAL_DWI"