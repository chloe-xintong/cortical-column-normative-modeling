#!/usr/bin/env bash
set -euo pipefail

# Purpose:
# Register the DWI image to FreeSurfer anatomical space using bbregister.
# Usage:
# bash register_dwi_to_t1.sh <subject_id> <dwi_image> <subjects_dir> <output_dat>
# Inputs:
# subject_id, DWI image, FreeSurfer subjects directory
# Output:
# DWI2T1_dti.dat registration transform


SUBJECT_ID="$1"
DWI_IMAGE="$2"
SUBJECTS_DIR="$3"
OUTPUT_DAT="$4"

if [ "$#" -ne 4 ]; then
    echo "Usage:"
    echo "bash registration/register_dwi_to_t1.sh \\"
    echo "  <subject_id> \\"
    echo "  <dwi_image> \\"
    echo "  <subjects_dir> \\"
    echo "  <output_dat>"
    exit 1
fi

if [ ! -f "$DWI_IMAGE" ]; then
    echo "DWI image not found: $DWI_IMAGE"
    exit 1
fi

if [ ! -d "$SUBJECTS_DIR/$SUBJECT_ID" ]; then
    echo "FreeSurfer subject not found:"
    echo "$SUBJECTS_DIR/$SUBJECT_ID"
    exit 1
fi

export SUBJECTS_DIR="$SUBJECTS_DIR"

echo "Subject:      $SUBJECT_ID"
echo "Moving image: $DWI_IMAGE"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "Output:       $OUTPUT_DAT"

bbregister \
    --s "$SUBJECT_ID" \
    --mov "$DWI_IMAGE" \
    --reg "$OUTPUT_DAT" \
    --sd "$SUBJECTS_DIR" \
    --dti \
    --init-fsl \
    --12

echo "Registration finished."
echo "Transform saved to:"
echo "$OUTPUT_DAT"