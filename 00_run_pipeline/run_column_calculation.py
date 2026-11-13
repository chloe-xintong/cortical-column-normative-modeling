#!/usr/bin/env python3
"""Calculate Fsaverage-Indexed Cortical Columns

Description
-----------
Build subject-specific cortical columns indexed by fsaverage vertices. The
script can stop after column construction or optionally sample one arbitrary
3D scalar metric; MD is not required.

Methods
-------
1. Validate the subject and template FreeSurfer reconstructions.
2. Optionally resolve or calculate a DWI-to-T1 registration transform.
3. Map fsaverage vertices to the subject's registered spherical surface.
4. Construct equally spaced pial-to-white points for every mapped vertex.
5. Optionally sample a user-selected scalar image along the columns.

Inputs
------
- Subject ID and FreeSurfer SUBJECTS_DIR
- FreeSurfer template name and desired number of depth points
- Optional scalar metric image, metric name, registration, and DWI reference

Outputs
-------
- Left/right ``*_columns.npz`` files and mapping-QC JSON files
- Optional DWI-to-T1 registration transform
- Optional left/right vertex-by-depth metric profile NPZ files

Usage
-----
python 00_run_pipeline/run_column_calculation.py     --subject-id D0007     --subjects-dir /path/to/freesurfer     --output-dir /path/to/D0007_column_output

Notes
-----
- Column-only mode requires no DWI or MD image.
- Any sampled metric must share the DWI-reference voxel grid and affine.
- Columns are stored in native FreeSurfer surface RAS and indexed by template
  vertex ID.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_COLUMNS = PROJECT_ROOT / "03_column" / "build_template_columns.py"
SAMPLE_COLUMNS = PROJECT_ROOT / "04_profile_sampling" / "sample_template_columns.py"
REGISTER_DWI = PROJECT_ROOT / "02_registration" / "register_dwi_to_t1.sh"


def require_file(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found:\n{resolved}")
    return resolved


def require_dir(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} not found:\n{resolved}")
    return resolved


def run(command: list[Path | str], label: str) -> None:
    command_text = [str(value) for value in command]
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(" ".join(command_text))
    subprocess.run(command_text, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fsaverage-indexed native cortical columns and optionally "
            "sample one user-selected scalar metric."
        )
    )
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--template",
        choices=("fsaverage", "fsaverage6"),
        default="fsaverage",
    )
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument(
        "--metric-image",
        help="Optional 3D scalar image. Omit it to stop after column creation.",
    )
    parser.add_argument(
        "--metric-name",
        help="Short output label for --metric-image, for example md or myelin.",
    )
    parser.add_argument(
        "--registration",
        help="Existing bbregister DWI-to-T1 .dat transform.",
    )
    parser.add_argument(
        "--registration-reference",
        help=(
            "DWI-space image used to create a new bbregister transform when "
            "--registration is omitted. It can also define sampling geometry."
        ),
    )
    parser.add_argument(
        "--dwi-reference",
        help=(
            "Image defining the voxel grid used by the transform and metric. "
            "Defaults to --registration-reference."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_id = args.subject_id.strip()
    subjects_dir = require_dir(args.subjects_dir, "FreeSurfer SUBJECTS_DIR")
    require_dir(subjects_dir / subject_id, "FreeSurfer subject")
    require_dir(subjects_dir / args.template, "FreeSurfer template")
    require_file(BUILD_COLUMNS, "Column builder")

    if args.points < 2:
        raise ValueError("--points must be at least 2")
    if args.metric_image is None and any(
        value is not None
        for value in (args.metric_name, args.registration, args.dwi_reference)
    ):
        raise ValueError(
            "Metric-specific options were provided without --metric-image."
        )
    if args.metric_image is not None and not args.metric_name:
        raise ValueError("--metric-name is required with --metric-image")
    if args.metric_name and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.metric_name):
        raise ValueError("--metric-name may contain only letters, numbers, . _ and -")

    output_dir = Path(args.output_dir).expanduser().resolve()
    columns_dir = output_dir / "columns"
    profiles_dir = output_dir / "profiles_before_clean"
    registration_dir = output_dir / "registration"

    # Registration is needed only when a metric will be sampled.  Resolve or
    # create it before Step 03 so the numbered stages run in their documented
    # order.  Column-only mode skips Step 02 entirely.
    metric_image = None
    dwi_reference = None
    registration = None
    if args.metric_image is not None:
        metric_image = require_file(args.metric_image, "Metric image")
        dwi_reference_value = args.dwi_reference or args.registration_reference
        if not dwi_reference_value:
            raise ValueError(
                "Sampling requires --dwi-reference or --registration-reference."
            )
        dwi_reference = require_file(dwi_reference_value, "DWI reference")
        if args.registration:
            registration = require_file(args.registration, "Registration transform")
        else:
            registration_reference = require_file(
                args.registration_reference,
                "Registration reference",
            )
            if shutil.which("bbregister") is None:
                raise RuntimeError(
                    "bbregister is not available in the active environment"
                )
            registration_dir.mkdir(parents=True, exist_ok=True)
            registration = registration_dir / f"{subject_id}_DWI2T1.dat"
            run(
                [
                    "bash",
                    REGISTER_DWI,
                    subject_id,
                    registration_reference,
                    subjects_dir,
                    registration,
                ],
                "STEP 02: REGISTER DWI TO T1",
            )
            require_file(registration, "Registration transform")

    run(
        [
            sys.executable,
            BUILD_COLUMNS,
            "--subject-id",
            subject_id,
            "--subjects-dir",
            subjects_dir,
            "--template",
            args.template,
            "--output-dir",
            columns_dir,
            "--points",
            args.points,
        ],
        "STEP 03: BUILD FSAVERAGE-INDEXED COLUMNS",
    )
    for hemisphere in ("lh", "rh"):
        require_file(
            columns_dir / f"{subject_id}_{hemisphere}_columns.npz",
            "Column output",
        )

    if args.metric_image is None:
        print(f"\nColumn calculation complete:\n{columns_dir}")
        return

    require_file(SAMPLE_COLUMNS, "Column sampler")
    run(
        [
            sys.executable,
            SAMPLE_COLUMNS,
            "--subject-id",
            subject_id,
            "--columns-dir",
            columns_dir,
            "--metric-image",
            metric_image,
            "--metric-name",
            args.metric_name,
            "--registration",
            registration,
            "--dwi",
            dwi_reference,
            "--output-dir",
            profiles_dir,
        ],
        f"STEP 04: SAMPLE {args.metric_name} ALONG COLUMNS",
    )
    for hemisphere in ("lh", "rh"):
        require_file(
            profiles_dir
            / f"{subject_id}_{hemisphere}_{args.metric_name}_profiles.npz",
            "Sampled fsaverage vertex profile",
        )
    print(f"\nColumn sampling complete:\n{profiles_dir}")


if __name__ == "__main__":
    main()
