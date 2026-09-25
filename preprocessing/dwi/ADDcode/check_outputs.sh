#!/usr/bin/env bash
set -euo pipefail

subject="$1"
out="$2"
raw="$3"

check_file() {
  local f="$1"
  [[ -s "$f" ]] || { echo "CHECK_FAILED: missing or empty $f" >&2; exit 1; }
}

check_file "$out/${subject}_01_b1000_mean.nii.gz"
check_file "$out/${subject}_02_dwi_brain_mask.nii.gz"
check_file "$out/${subject}_03_dwi_masked_int16.nii.gz"
check_file "$out/${subject}_04_dwi_mpca.nii.gz"
check_file "$out/${subject}_05_4D_upsampled.nii.gz"
check_file "$out/${subject}_05_subjspace_mask_upsampled.nii.gz"
check_file "$out/${subject}_07_dt.mif"
check_file "$out/${subject}_adc.nii.gz"
check_file "$out/${subject}_fa.nii.gz"
check_file "$out/${subject}_ad.nii.gz"
check_file "$out/${subject}_rd.nii.gz"

python - "$subject" "$out" "$raw" <<'PY'
import subprocess, sys
from pathlib import Path
subject, out, raw = sys.argv[1:]
mask = Path(out) / f"{subject}_05_subjspace_mask_upsampled.nii.gz"
dwi = Path(out) / f"{subject}_05_4D_upsampled.nii.gz"
for image in (dwi, mask):
    result = subprocess.run(["mrinfo", str(image), "-size"], text=True, capture_output=True, check=True)
    if not result.stdout.strip():
        raise SystemExit(f"CHECK_FAILED: no image dimensions for {image}")
print(f"CHECKPOINTS_OK: {subject}; DWI and tensor metric outputs are non-empty and readable")
PY
