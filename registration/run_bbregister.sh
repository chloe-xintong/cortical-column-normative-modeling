#!/usr/bin/env bash
set -euo pipefail

# Register DWI to the FreeSurfer subject using bbregister.
# This script follows the registration step used in the
# reference ADRC_MUSE_master_script.sh workflow.
#
# Usage:
#   bash registration/run_bbregister.sh \
#       <subject_id> \
#       <freesurfer_subjects_dir> \
#       <masked_dwi.nii.gz> \
#       <output_transform.dat>

SUBJECT_ID="$1"
SUBJECTS_DIR="$2"
DWI="$3"
OUTPUT="$4"

# FreeSurfer uses SUBJECTS_DIR to locate the reconstructed subject.
export SUBJECTS_DIR

# Run DTI-oriented boundary-based registration.
# --s: FreeSurfer subject ID
# --mov: moving DWI image
# --reg: output registration file
# --sd: FreeSurfer subjects directory
# --dti: use DTI registration settings
# --init-fsl: initialize registration with FSL
# --12: allow 12-degree-of-freedom affine registration
bbregister \
    --s "$SUBJECT_ID" \
    --mov "$DWI" \
    --reg "$OUTPUT" \
    --sd "$SUBJECTS_DIR" \
    --dti \
    --init-fsl \
    --12

echo "Registration complete."
echo "Saved transform: $OUTPUT"