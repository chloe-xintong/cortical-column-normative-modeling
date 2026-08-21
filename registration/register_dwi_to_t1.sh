#!/usr/bin/env bash
set -euo pipefail

# Register a masked diffusion image to FreeSurfer anatomical space
# using bbregister.
#
# Usage:
# bash registration/register_dwi_to_t1.sh \
#   SUBJECT_ID \
#   MOVING_IMAGE \
#   SUBJECTS_DIR \
#   OUTPUT_DAT

if [ "$#" -ne 4 ]; then
    echo "Usage:"
    echo "bash registration/register_dwi_to_t1.sh \\"
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