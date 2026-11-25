#!/usr/bin/env python3
"""Build the compact local-FreeSurfer review package used for prior bbregister QC.

Description
-----------
Build the compact local-FreeSurfer review package used for prior bbregister QC.

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
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/prepare_source_image_freeview_package.py --help
"""

from __future__ import annotations

import argparse
import csv
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

FS_ROOT = Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer")
COLUMN_ROOT = Path("/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns")


def source_files(subject: str) -> dict[str, Path]:
    fs = FS_ROOT / subject
    registration = COLUMN_ROOT / subject / "registration"
    return {
        "orig.mgz": fs / "mri" / "orig.mgz",
        "dwi_masked.nii.gz": registration / f"{subject}_dwi_masked.nii.gz",
        "DWI2T1.dat": registration / f"{subject}_DWI2T1.dat",
        "lh.white": fs / "surf" / "lh.white",
        "lh.pial": fs / "surf" / "lh.pial",
        "rh.white": fs / "surf" / "rh.white",
        "rh.pial": fs / "surf" / "rh.pial",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for tier, subjects in SUBJECTS.items():
        tier_dir = output / tier
        tier_dir.mkdir(exist_ok=True)
        for subject, focus in subjects.items():
            for suffix, source in source_files(subject).items():
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination = tier_dir / f"{subject}_{suffix}"
                if destination.exists() and destination.stat().st_size != source.stat().st_size:
                    raise RuntimeError(f"Refusing to overwrite a different-size file: {destination}")
                if not destination.exists():
                    shutil.copy2(source, destination)
                manifest.append({"subject_id": subject, "tier": tier, "focus": focus,
                                 "file_role": suffix, "source_path": str(source),
                                 "package_path": str(destination.relative_to(output)),
                                 "bytes": source.stat().st_size})

    with (output / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["subject_id", "tier", "focus"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for tier, subjects in SUBJECTS.items():
            for subject, focus in subjects.items():
                writer.writerow({"subject_id": subject, "tier": tier, "focus": focus})
    with (output / "file_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)

    commands = """#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v freeview >/dev/null 2>&1; then
    echo "freeview was not found. Source the local FreeSurfer environment first." >&2
    exit 1
fi
while IFS=, read -r subject_id tier focus; do
    [[ "$subject_id" == "subject_id" ]] && continue
    data_dir="$package_dir/$tier"
    echo "Opening $subject_id | $tier | focus: $focus"
    freeview \\
      -v "$data_dir/${subject_id}_orig.mgz:colormap=grayscale" \\
         "$data_dir/${subject_id}_dwi_masked.nii.gz:reg=$data_dir/${subject_id}_DWI2T1.dat:colormap=grayscale:opacity=0.5" \\
      -f "$data_dir/${subject_id}_lh.white:edgecolor=yellow" \\
         "$data_dir/${subject_id}_rh.white:edgecolor=yellow" \\
         "$data_dir/${subject_id}_lh.pial:edgecolor=red" \\
         "$data_dir/${subject_id}_rh.pial:edgecolor=red"
done < "$package_dir/selection_manifest.csv"
"""
    command_path = output / "freeview_commands.sh"
    command_path.write_text(commands, encoding="utf-8")
    command_path.chmod(0o755)
    (output / "README.md").write_text(
        "# Source-image QC Freeview package\n\n"
        "This package follows the prior bbregister QC transfer format. Extract it on the local machine, "
        "source FreeSurfer, and run:\n\n```bash\nbash freeview_commands.sh\n```\n\n"
        "Close each Freeview window to advance to the next subject. Yellow contours are white surfaces; "
        "red contours are pial surfaces. The T1 `orig.mgz` and registered masked DWI are included per subject.\n",
        encoding="utf-8",
    )
    print({"output_dir": str(output), "subjects": len({x["subject_id"] for x in manifest}),
           "files": len(manifest), "bytes": sum(int(x["bytes"]) for x in manifest)})


if __name__ == "__main__":
    main()
