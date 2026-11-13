#!/usr/bin/env python3
"""Inventory the complete fsaverage O3 cohort without changing its outputs.

Description
-----------
Inventory the complete fsaverage O3 cohort without changing its outputs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--o3-root`` command-line input
- ``--subjects-dir`` command-line input
- ``--dwi-root`` command-line input
- ``--output-manifest`` command-line input
- ``--inventory-output`` command-line input
- ``--missing-output`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/option3/scripts/build_full_cohort_manifest.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HEMISPHERES = ("lh", "rh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o3-root", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--dwi-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument("--missing-output", type=Path, required=True)
    return parser.parse_args()


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return paths[0]


def inspect_subject(
    subject_id: str,
    o3_root: Path,
    subjects_dir: Path,
    dwi_root: Path,
) -> dict[str, object]:
    subject_root = o3_root / subject_id
    fs_root = subjects_dir / subject_id
    dwi_subject = dwi_root / subject_id
    transform = subject_root / "registration" / f"{subject_id}_DWI2T1.dat"
    md_image = first_existing(
        [
            dwi_subject / f"{subject_id}_md.nii.gz",
            dwi_subject / f"{subject_id}_adc.nii.gz",
        ]
    )

    row: dict[str, object] = {
        "subject_id": subject_id,
        "o3_subject_dir": str(subject_root),
        "freesurfer_subject_dir": str(fs_root),
        "transform_path": str(transform),
        "mincost_path": str(Path(f"{transform}.mincost")),
        "md_image": str(md_image),
    }
    required_paths = {
        "freesurfer_subject": fs_root,
        "transform": transform,
        "mincost": Path(f"{transform}.mincost"),
        "md_image": md_image,
    }
    for hemisphere in HEMISPHERES:
        column_path = (
            subject_root / "columns" / f"{subject_id}_{hemisphere}_columns.npz"
        )
        profile_path = (
            subject_root
            / "profiles_before_clean"
            / f"{subject_id}_{hemisphere}_md_profiles.npz"
        )
        clean_profile_path = (
            subject_root / "profiles" / f"{subject_id}_{hemisphere}_md_profiles.npz"
        )
        row[f"column_{hemisphere}"] = str(column_path)
        row[f"profile_{hemisphere}"] = str(profile_path)
        row[f"clean_profile_{hemisphere}"] = str(clean_profile_path)
        required_paths[f"column_{hemisphere}"] = column_path
        required_paths[f"profile_{hemisphere}"] = profile_path
        required_paths[f"clean_profile_{hemisphere}"] = clean_profile_path
        for surface_name in ("white", "pial", "sphere.reg", "thickness"):
            key = f"{hemisphere}_{surface_name.replace('.', '_')}"
            required_paths[key] = fs_root / "surf" / f"{hemisphere}.{surface_name}"
        required_paths[f"{hemisphere}_aparc"] = (
            fs_root / "label" / f"{hemisphere}.aparc.annot"
        )

    missing = [
        name
        for name, path in required_paths.items()
        if not path.exists() or (path.is_file() and path.stat().st_size == 0)
    ]
    row["input_complete"] = not missing
    row["missing_inputs"] = ";".join(missing)
    return row


def main() -> None:
    args = parse_args()
    subject_ids = sorted(
        path.name
        for path in args.o3_root.iterdir()
        if path.is_dir() and path.name.startswith("D")
    )
    rows = [
        inspect_subject(
            subject_id,
            args.o3_root,
            args.subjects_dir,
            args.dwi_root,
        )
        for subject_id in subject_ids
    ]
    inventory = pd.DataFrame(rows)
    complete = inventory.loc[inventory["input_complete"]].reset_index(drop=True)
    missing = inventory.loc[~inventory["input_complete"]].reset_index(drop=True)

    for path in (
        args.output_manifest,
        args.inventory_output,
        args.missing_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.inventory_output, index=False)
    missing.to_csv(args.missing_output, index=False)
    inventory.to_csv(args.output_manifest, index=False)

    print(f"Requested O3 directories: {len(inventory)}")
    print(f"Input-complete subjects: {len(complete)}")
    print(f"Subjects with missing inputs: {len(missing)}")
    print(f"Manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
