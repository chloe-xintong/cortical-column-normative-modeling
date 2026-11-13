#!/usr/bin/env bash

# Register DWI to Native T1 Space
#
# ## Description
#
# Estimate a DWI-to-T1 affine registration with FreeSurfer bbregister.
#
# ## Methods
#
# 1. Validate the moving DWI reference and FreeSurfer subject directory.
# 2. Initialize registration with FSL.
# 3. Run 12-parameter DTI-mode bbregister.
# 4. Save the FreeSurfer register.dat transform.
#
# ## Inputs
#
# - Subject ID, masked 3D DWI moving image, and FreeSurfer SUBJECTS_DIR
# - Output path for the registration transform
#
# ## Outputs
#
# - DWI-to-T1 `.dat` transform and bbregister companion QC files
#
# ## Usage
#
# bash 02_registration/register_dwi_to_t1.sh \
#     SUBJECT_ID MOVING_IMAGE SUBJECTS_DIR OUTPUT_DAT
#
# ## Notes
#
# - FreeSurfer and bbregister must be available in the active environment.
# - Registration acceptance still requires the project registration QC.

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage:"
    echo "bash 02_registration/register_dwi_to_t1.sh \\"
    echo "  <subject_id> \\"
    echo "  <moving_image> \\"
    echo "  <subjects_dir> \\"
    echo "  <output_dat>"
    exit 1
fi

SUBJECT_ID="$1"
MOVING_IMAGE="$2"
SUBJECTS_DIR="$3"
OUTPUT_DAT="$4"

if [ ! -f "$MOVING_IMAGE" ]; then
    echo "Moving image not found: $MOVING_IMAGE"
    exit 1
fi

if [ ! -d "$SUBJECTS_DIR/$SUBJECT_ID" ]; then
    echo "FreeSurfer subject not found:"
    echo "$SUBJECTS_DIR/$SUBJECT_ID"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DAT")"

export SUBJECTS_DIR="$SUBJECTS_DIR"

echo "Subject:      $SUBJECT_ID"
echo "Moving image: $MOVING_IMAGE"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "Output:       $OUTPUT_DAT"

bbregister \
    --s "$SUBJECT_ID" \
    --mov "$MOVING_IMAGE" \
    --reg "$OUTPUT_DAT" \
    --sd "$SUBJECTS_DIR" \
    --dti \
    --init-fsl \
    --12

echo "Registration finished."
echo "Transform saved to:"
echo "$OUTPUT_DAT"
