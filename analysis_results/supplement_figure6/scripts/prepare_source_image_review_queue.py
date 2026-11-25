#!/usr/bin/env python3
"""Copy a compact, reproducible source-image QC packet into the workspace.

Description
-----------
Copy a compact, reproducible source-image QC packet into the workspace.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python analysis_results/figure6/scripts/prepare_source_image_review_queue.py --help
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


SUBJECTS = {
    "reference_confirmed_fail": {
        "D0271": "confirmed source-image structural abnormality",
        "D0285": "confirmed structural missing tissue",
    },
    "primary_review": {
        "D0203": "lh parahippocampal, depth 0.40",
        "D0136": "rh superiorfrontal, depth 0.55",
        "D0057": "rh middletemporal, depth 0.75",
        "D0127": "lh parsorbitalis, depth 0.65",
        "D0381": "lh parsopercularis, depth 0.50",
        "D0140": "rh paracentral, depth 0.80",
        "D0344": "lh caudalmiddlefrontal, depth 0.50",
        "D0275": "lh parahippocampal, depth 0.40",
    },
    "secondary_review": {
        "D0390": "lh posteriorcingulate, depth 0.90",
        "D0206": "lh supramarginal, depth 0.60",
        "D0073": "rh caudalanteriorcingulate, depth 0.50",
        "D0123": "rh transversetemporal, depth 0.25",
        "D0074": "rh supramarginal, depth 0.65",
        "D0402": "rh parahippocampal, depth 0.60",
        "D0237": "lh rostralmiddlefrontal, depth 0.75",
    },
}

RAW_ROOT = Path("/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE")
DWI_ROOT = Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi")
FS_ROOT = Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer")
COLUMN_ROOT = Path("/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256(source)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256(destination) != source_hash:
            raise RuntimeError(f"Refusing to overwrite a non-identical review file: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        "source_path": str(source), "destination_path": str(destination),
        "bytes": source.stat().st_size, "sha256": source_hash,
    }


def subject_files(subject: str) -> dict[str, Path]:
    fs = FS_ROOT / subject
    return {
        "raw_T1.nii.gz": RAW_ROOT / subject / "T1.nii.gz",
        "fs_T1.mgz": fs / "mri" / "T1.mgz",
        "fs_brainmask.mgz": fs / "mri" / "brainmask.mgz",
        "fs_aparc+aseg.mgz": fs / "mri" / "aparc+aseg.mgz",
        "lh.white": fs / "surf" / "lh.white",
        "lh.pial": fs / "surf" / "lh.pial",
        "rh.white": fs / "surf" / "rh.white",
        "rh.pial": fs / "surf" / "rh.pial",
        "corrected_mean_b0.nii.gz": DWI_ROOT / subject / f"{subject}_corrected_mean_b0.nii.gz",
        "corrected_mean_b0_brain_mask.nii.gz": DWI_ROOT / subject / f"{subject}_corrected_mean_b0_brain_mask.nii.gz",
        "DWI2T1.dat": COLUMN_ROOT / subject / "registration" / f"{subject}_DWI2T1.dat",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    checklist: list[dict[str, object]] = []
    for tier, subjects in SUBJECTS.items():
        for subject, focus in subjects.items():
            destination_dir = output / tier / subject
            for name, source in subject_files(subject).items():
                record = copy_verified(source, destination_dir / name)
                manifest.append({"tier": tier, "subject_id": subject, "focus": focus,
                                 "file_role": name, **record})
            checklist.append({
                "tier": tier, "subject_id": subject, "focus": focus,
                "raw_T1_complete": "", "structural_abnormality": "",
                "freesurfer_white_surface": "", "freesurfer_pial_surface": "",
                "corrected_b0_signal": "", "b0_brain_mask": "",
                "local_b0_T1_alignment": "", "decision": "", "exclusion_reason": "", "notes": "",
            })

    with (output / "copy_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    with (output / "review_checklist.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checklist[0]))
        writer.writeheader(); writer.writerows(checklist)
    print({"output_dir": str(output), "subjects": len(checklist), "files": len(manifest),
           "bytes": sum(int(x["bytes"]) for x in manifest)})


if __name__ == "__main__":
    main()
