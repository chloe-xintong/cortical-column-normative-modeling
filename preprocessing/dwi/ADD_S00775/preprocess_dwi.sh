#!/usr/bin/env bash

# Preprocess S-series / ADD diffusion MRI using the reconstructed
# historical AD Decode / DTC workflow.
#
# Steps:
#   1. Extract mean b=1000 DWI for brain-mask generation
#   2. Generate diffusion-space brain mask with BET
#   3. Apply brain mask to the raw 4D DWI
#   4. Historical DTC MP-PCA denoising
#   5. Regrid raw 4D DWI to 0.5-mm resolution
#   6. Regrid diffusion brain mask to 0.5-mm resolution
#
# The 0.5-mm DWI and mask are prepared for the historical
# cortical-column tensor-fitting workflow.
#
# Usage:
#   bash preprocess_dwi.sh \
#       INPUT_DWI \
#       BVEC \
#       BVAL \
#       OUTPUT_DIR \
#       SUBJECT_ID

set -o pipefail

# BET's imtest helper is a Python wrapper.  On the cluster, the regular
# analysis environment does not expose FSL's Python packages, so expose the
# version-matched FSL site-packages explicitly.
export PYTHONPATH="/home/apps/fsl/6.0.7.6/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"


if [[ $# -ne 5 ]]; then
    echo "Usage:"
    echo "  bash preprocess_dwi.sh \\"
    echo "    INPUT_DWI \\"
    echo "    BVEC \\"
    echo "    BVAL \\"
    echo "    OUTPUT_DIR \\"
    echo "    SUBJECT_ID"
    exit 1
fi


INPUT_DWI=$1
BVEC=$2
BVAL=$3
OUTPUT_DIR=$4
SUBJECT_ID=$5


DTC_ROOT="${DTC_ROOT:-/mnt/newStor/paros/paros_WORK/jacques/gitfolder/DTC}"


# -------------------------------------------------------------------------
# Check subject ID.
#
# S-series subjects use the ADD preprocessing workflow.
# -------------------------------------------------------------------------

if [[ "${SUBJECT_ID:0:1}" != "S" ]]; then
    echo "ERROR: ADD preprocessing expects an S-series subject." >&2
    echo "Received: $SUBJECT_ID" >&2
    exit 1
fi


# -------------------------------------------------------------------------
# Check input files.
# -------------------------------------------------------------------------

for file in "$INPUT_DWI" "$BVEC" "$BVAL"; do

    if [[ ! -f "$file" ]]; then
        echo "ERROR: Input file not found: $file" >&2
        exit 1
    fi

done


# -------------------------------------------------------------------------
# Check required software.
# -------------------------------------------------------------------------

for command in \
    select_dwi_vols \
    bet \
    fslmaths \
    fslstats \
    mrgrid \
    python
do

    if ! command -v "$command" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $command" >&2
        exit 1
    fi

done


mkdir -p "$OUTPUT_DIR"


# -------------------------------------------------------------------------
# Output files.
# -------------------------------------------------------------------------

DWI_B1000="${OUTPUT_DIR}/${SUBJECT_ID}_01_b1000_mean.nii.gz"

BET_BASE="${OUTPUT_DIR}/${SUBJECT_ID}_02_dwi_brain"
BRAIN_MASK="${OUTPUT_DIR}/${SUBJECT_ID}_02_dwi_brain_mask.nii.gz"

MASKED_DWI="${OUTPUT_DIR}/${SUBJECT_ID}_03_dwi_masked_int16.nii.gz"

DENOISED_DWI="${OUTPUT_DIR}/${SUBJECT_ID}_04_dwi_mpca.nii.gz"

DWI_UPSAMPLED="${OUTPUT_DIR}/${SUBJECT_ID}_05_4D_upsampled.nii.gz"

MASK_UPSAMPLED="${OUTPUT_DIR}/${SUBJECT_ID}_05_subjspace_mask_upsampled.nii.gz"


echo "Subject: $SUBJECT_ID"
echo "Input DWI: $INPUT_DWI"
echo "Output directory: $OUTPUT_DIR"


# -------------------------------------------------------------------------
# Step 1
# Extract all b=1000 volumes and compute their mean.
#
# Historical command:
#
#   select_dwi_vols raw_nii bvals raw_dwi 1000 -m
# -------------------------------------------------------------------------

echo
echo "[1/6] Creating mean b=1000 DWI..."

select_dwi_vols \
    "$INPUT_DWI" \
    "$BVAL" \
    "$DWI_B1000" \
    1000 \
    -m

if [[ $? -ne 0 || ! -f "$DWI_B1000" ]]; then
    echo "ERROR: Mean b=1000 generation failed." >&2
    exit 1
fi


# -------------------------------------------------------------------------
# Step 2
# Create diffusion-space brain mask with BET.
#
# Historical command:
#
#   bet raw_dwi tmp -m -n -R
#
# FSL BET uses f=0.50 by default. The reconstructed S00775 mask
# generated with -R -m -n -f 0.50 matches the historical pipeline
# mask voxel-by-voxel.
# -------------------------------------------------------------------------

echo
echo "[2/6] Generating diffusion-space brain mask..."

bet \
    "$DWI_B1000" \
    "$BET_BASE" \
    -m \
    -n \
    -R \
    -f 0.50

if [[ $? -ne 0 || ! -f "$BRAIN_MASK" ]]; then
    echo "ERROR: BET brain mask generation failed." >&2
    exit 1
fi


echo
echo "Brain-mask voxel count:"

fslstats \
    "$BRAIN_MASK" \
    -V


# -------------------------------------------------------------------------
# Step 3
# Apply the diffusion brain mask to the raw 4D DWI.
#
# Historical command:
#
#   fslmaths raw_nii -mas tmp_mask masked_nii -odt input
#
# "-odt input" preserves the input datatype.
# -------------------------------------------------------------------------

echo
echo "[3/6] Applying diffusion brain mask..."

fslmaths \
    "$INPUT_DWI" \
    -mas "$BRAIN_MASK" \
    "$MASKED_DWI" \
    -odt input

if [[ $? -ne 0 || ! -f "$MASKED_DWI" ]]; then
    echo "ERROR: DWI masking failed." >&2
    exit 1
fi


# -------------------------------------------------------------------------
# Step 4
# Run the historical DTC MP-PCA denoising implementation.
#
# Historical parameters:
#
#   patch_radius = 2
#   pca_method   = eig
#   return_sigma = False
#   out_dtype    = None
#   processes    = 1
#
# This is the DTC implementation and is intentionally not replaced by
# MRtrix dwidenoise.
# -------------------------------------------------------------------------

echo
echo "[4/6] Running historical DTC MP-PCA denoising..."

DTC_ROOT="$DTC_ROOT" \
MASKED_DWI="$MASKED_DWI" \
DENOISED_DWI="$DENOISED_DWI" \
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python - <<'PY'

import os
import sys

from dipy.io.image import load_nifti, save_nifti


sys.path.insert(
    0,
    os.environ["DTC_ROOT"],
)

from diff_handlers.denoise_processes import mppca


input_file = os.environ["MASKED_DWI"]
output_file = os.environ["DENOISED_DWI"]


data, affine, _ = load_nifti(
    input_file,
    return_voxsize=True,
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
    verbose=True,
)


save_nifti(
    output_file,
    denoised,
    affine,
)


print("Saved:", output_file)

PY


if [[ $? -ne 0 || ! -f "$DENOISED_DWI" ]]; then
    echo "ERROR: MP-PCA denoising failed." >&2
    exit 1
fi


# -------------------------------------------------------------------------
# Step 5
# Regrid the raw 4D DWI to 0.5-mm isotropic resolution.
#
# Historical Hanwen cortical-column workflow:
#
#   mrgrid raw_DWI regrid \
#       -voxel 0.5,0.5,0.5 \
#       4D_upsampled.nii.gz
#
# Important:
# The historical tensor branch uses the upsampled raw DWI rather than
# the MP-PCA / volume-wise coregistered DWI.
# -------------------------------------------------------------------------

echo
echo "[5/6] Regridding raw DWI to 0.5 mm..."

mrgrid \
    "$INPUT_DWI" \
    regrid \
    -voxel 0.5,0.5,0.5 \
    "$DWI_UPSAMPLED" \
    -force

if [[ $? -ne 0 || ! -f "$DWI_UPSAMPLED" ]]; then
    echo "ERROR: DWI regridding failed." >&2
    exit 1
fi


# -------------------------------------------------------------------------
# Step 6
# Regrid the diffusion-space brain mask to the same 0.5-mm grid.
#
# Historical Hanwen cortical-column workflow:
#
#   mrgrid subjspace_mask.nii.gz regrid \
#       -voxel 0.5,0.5,0.5 \
#       subjspace_mask_upsampled.nii.gz
#
# This mask is passed to the common tensor-fitting script together with
# the 0.5-mm DWI.
# -------------------------------------------------------------------------

echo
echo "[6/6] Regridding brain mask to 0.5 mm..."

mrgrid \
    "$BRAIN_MASK" \
    regrid \
    -voxel 0.5,0.5,0.5 \
    "$MASK_UPSAMPLED" \
    -force

if [[ $? -ne 0 || ! -f "$MASK_UPSAMPLED" ]]; then
    echo "ERROR: Brain-mask regridding failed." >&2
    exit 1
fi


# -------------------------------------------------------------------------
# Final checks.
# -------------------------------------------------------------------------

for file in \
    "$DWI_B1000" \
    "$BRAIN_MASK" \
    "$MASKED_DWI" \
    "$DENOISED_DWI" \
    "$DWI_UPSAMPLED" \
    "$MASK_UPSAMPLED"
do

    if [[ ! -f "$file" ]]; then
        echo "ERROR: Expected output not found: $file" >&2
        exit 1
    fi

done


echo
echo "DWI preprocessing completed successfully."
echo
echo "Mean b=1000:"
echo "$DWI_B1000"
echo
echo "Brain mask:"
echo "$BRAIN_MASK"
echo
echo "Masked DWI:"
echo "$MASKED_DWI"
echo
echo "MP-PCA DWI:"
echo "$DENOISED_DWI"
echo
echo "0.5-mm DWI for tensor fitting:"
echo "$DWI_UPSAMPLED"
echo
echo "0.5-mm mask for tensor fitting:"
echo "$MASK_UPSAMPLED"
