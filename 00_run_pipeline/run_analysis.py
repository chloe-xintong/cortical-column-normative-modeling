#!/usr/bin/env python3
"""Run Column Construction and Metric-Profile Analysis

Description
-----------
Run registration, fsaverage-indexed cortical-column construction, optional
multi-metric sampling, shared profile cleaning, and regional aggregation for
one subject with prepared FreeSurfer and DWI inputs.

Methods
-------
1. Register the DWI reference to the subject's native T1 space.
2. Map every fsaverage vertex to the subject's registered spherical surface.
3. Construct pial-to-white columns in native T1 surface RAS.
4. Sample supplied scalar metric images at every column depth.
5. Optionally apply a shared validity mask and aggregate regional profiles.

Inputs
------
- FreeSurfer SUBJECTS_DIR containing the subject and template
- Masked DWI registration reference
- MD/ADC image and optional FA, AD, and RD images
- Subject ID, template name, and analysis output directory

Outputs
-------
- DWI-to-T1 registration transform
- Left/right fsaverage-indexed column NPZ files
- Raw metric profiles for every fsaverage vertex
- Optional cleaned profiles and regional summaries

Usage
-----
python 00_run_pipeline/run_analysis.py     --subject-id D0007     --subjects-dir /path/to/freesurfer     --registration-reference /path/to/D0007_dwi_masked.nii.gz     --md-image /path/to/D0007_adc.nii.gz     --analysis-dir /path/to/output/D0007

Notes
-----
- This entry point does not perform T1 or DWI preprocessing.
- Use ``--before-clean`` to stop after vertex-level raw profile output.
- FreeSurfer, registration tools, and Python dependencies must be available.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEPTH_POINTS = 21
EXPECTED_REGIONS = 68


def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


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


def require_command(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(
            f"Required command not available: {command}. "
            "Load the required software environment first."
        )
    return executable


def run_command(command: list[Path | str], step_name: str) -> None:
    print_section(step_name)
    command_text = [str(item) for item in command]
    print("Command:")
    print(" ".join(command_text))
    print()
    result = subprocess.run(command_text, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def count_files(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run target-driven cortical column analysis for one subject."
    )
    parser.add_argument("--subject-id", required=True)
    parser.add_argument(
        "--subjects-dir",
        required=True,
        help="FreeSurfer SUBJECTS_DIR containing the subject and template.",
    )
    parser.add_argument(
        "--md-image",
        help="Mean diffusivity image used for cortical profile sampling.",
    )
    parser.add_argument(
        "--adc-image",
        help=(
            "Deprecated alias for --md-image. Kept temporarily for older job "
            "scripts."
        ),
    )
    for metric in ("fa", "ad", "rd"):
        parser.add_argument(
            f"--{metric}-image",
            help=f"{metric.upper()} image for cortical profile sampling.",
        )
    parser.add_argument(
        "--template",
        default="fsaverage",
        choices=("fsaverage", "fsaverage6"),
        help="FreeSurfer template whose vertices define the columns.",
    )
    parser.add_argument(
        "--before-clean",
        action="store_true",
        help="Stop after raw template-column profiles are generated.",
    )
    parser.add_argument(
        "--registration-reference",
        required=True,
        help="Masked diffusion-space image used by bbregister.",
    )
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="Output directory for this subject's analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_id = args.subject_id
    repo_root = Path(__file__).resolve().parents[1]

    subjects_dir = require_dir(args.subjects_dir, "FreeSurfer SUBJECTS_DIR")
    subject_dir = require_dir(
        subjects_dir / subject_id,
        "FreeSurfer subject directory",
    )
    template_dir = require_dir(
        subjects_dir / args.template,
        "FreeSurfer template directory",
    )

    md_image = args.md_image or args.adc_image
    if not md_image:
        raise ValueError("One of --md-image or --adc-image must be provided")

    metric_images = {"md": require_file(md_image, "MD image")}
    for metric in ("fa", "ad", "rd"):
        image = getattr(args, f"{metric}_image")
        if image:
            metric_images[metric] = require_file(image, f"{metric.upper()} image")

    required_metrics = {"md", "fa", "ad", "rd"}
    if args.before_clean and not required_metrics.issubset(metric_images):
        missing = ", ".join(sorted(required_metrics - set(metric_images)))
        raise ValueError(
            "--before-clean requires all four metric images; " f"missing: {missing}"
        )

    registration_reference = require_file(
        args.registration_reference,
        "Registration reference",
    )

    freesurfer_required = []
    for hemisphere in ("lh", "rh"):
        freesurfer_required.extend(
            [
                subject_dir / "surf" / f"{hemisphere}.white",
                subject_dir / "surf" / f"{hemisphere}.pial",
                subject_dir / "surf" / f"{hemisphere}.thickness",
                subject_dir / "surf" / f"{hemisphere}.sphere.reg",
                template_dir / "surf" / f"{hemisphere}.sphere.reg",
                template_dir / "label" / f"{hemisphere}.aparc.annot",
            ]
        )
    freesurfer_required.append(subject_dir / "mri" / "orig.mgz")
    for path in freesurfer_required:
        require_file(path, "Required FreeSurfer output")

    require_command("bbregister")

    register_script = require_file(
        repo_root / "02_registration" / "register_dwi_to_t1.sh",
        "Registration script",
    )
    build_columns_script = require_file(
        repo_root / "03_column" / "build_template_columns.py",
        "Template-column builder",
    )
    sample_columns_script = require_file(
        repo_root / "04_profile_sampling" / "sample_template_columns.py",
        "Template-column sampling script",
    )
    clean_profiles_script = require_file(
        repo_root / "04_profile_sampling" / "clean_template_profiles.py",
        "Template-profile cleaning script",
    )
    regional_profiles_script = require_file(
        repo_root / "04_profile_sampling" / "regional_template_profiles.py",
        "Template regional-profile script",
    )

    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    registration_dir = analysis_dir / "registration"
    columns_dir = analysis_dir / "columns"
    raw_profile_dir = analysis_dir / "profiles_before_clean"
    profile_dir = analysis_dir / "profiles"
    regional_profile_dir = analysis_dir / "regional_profiles"

    for directory in (registration_dir, columns_dir, raw_profile_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.before_clean:
        profile_dir.mkdir(parents=True, exist_ok=True)
        regional_profile_dir.mkdir(parents=True, exist_ok=True)

    print_section("TEMPLATE-DRIVEN CORTICAL COLUMN ANALYSIS")
    print(f"Subject:                 {subject_id}")
    print(f"Template:                {args.template}")
    print(f"FreeSurfer SUBJECTS_DIR: {subjects_dir}")
    print(f"MD image:                {metric_images['md']}")
    print(f"Registration reference:  {registration_reference}")
    print(f"Analysis directory:      {analysis_dir}")

    transform = registration_dir / f"{subject_id}_DWI2T1.dat"
    run_command(
        [
            "bash",
            register_script,
            subject_id,
            registration_reference,
            subjects_dir,
            transform,
        ],
        "STEP 1: DWI -> T1 REGISTRATION",
    )
    require_file(transform, "bbregister transform")

    run_command(
        [
            sys.executable,
            build_columns_script,
            "--subject-id",
            subject_id,
            "--subjects-dir",
            subjects_dir,
            "--template",
            args.template,
            "--output-dir",
            columns_dir,
            "--points",
            str(DEPTH_POINTS),
        ],
        "STEP 2: BUILD TEMPLATE-DRIVEN COLUMNS",
    )
    column_count = count_files(columns_dir, f"{subject_id}_*_columns.npz")
    if column_count != 2:
        raise RuntimeError(f"Expected 2 hemisphere column files, found {column_count}")

    for metric_name, metric_image in metric_images.items():
        run_command(
            [
                sys.executable,
                sample_columns_script,
                "--subject-id",
                subject_id,
                "--columns-dir",
                columns_dir,
                "--metric-image",
                metric_image,
                "--metric-name",
                metric_name,
                "--registration",
                transform,
                "--dwi",
                registration_reference,
                "--output-dir",
                raw_profile_dir,
            ],
            f"STEP 3: SAMPLE {metric_name.upper()} ALONG TEMPLATE COLUMNS",
        )
        profile_count = count_files(
            raw_profile_dir,
            f"{subject_id}_*_{metric_name}_profiles.npz",
        )
        if profile_count != 2:
            raise RuntimeError(
                f"Expected 2 raw {metric_name.upper()} profile files, "
                f"found {profile_count}"
            )

    if args.before_clean:
        print_section("STOP BEFORE CLEANING")
        print(f"Template columns saved in: {columns_dir}")
        print(f"Raw profiles saved in:     {raw_profile_dir}")
        return

    run_command(
        [
            sys.executable,
            clean_profiles_script,
            "--subject-id",
            subject_id,
            "--input-dir",
            raw_profile_dir,
            "--output-dir",
            profile_dir,
            "--metrics",
            *metric_images.keys(),
        ],
        "STEP 4: APPLY SHARED PROFILE VALIDITY MASK",
    )
    cleaning_count = count_files(
        profile_dir,
        f"{subject_id}_*_cleaning.json",
    )
    if cleaning_count != 2:
        raise RuntimeError(f"Expected 2 cleaning summaries, found {cleaning_count}")

    run_command(
        [
            sys.executable,
            regional_profiles_script,
            "--subject-id",
            subject_id,
            "--profile-dir",
            profile_dir,
            "--subjects-dir",
            subjects_dir,
            "--template",
            args.template,
            "--metric-name",
            "md",
            "--output-dir",
            regional_profile_dir,
        ],
        "STEP 5: AGGREGATE CLEANED MD BY TEMPLATE DK REGION",
    )
    region_mean_dir = require_dir(
        regional_profile_dir / "region_mean",
        "Regional mean profile directory",
    )
    regional_mean_count = count_files(
        region_mean_dir,
        f"{subject_id}_*_md_mean.csv",
    )
    if regional_mean_count != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} regional MD mean files, "
            f"found {regional_mean_count}"
        )
    qa_file = require_file(
        regional_profile_dir / f"{subject_id}_profiles_QA_md.csv",
        "Regional MD profile QA file",
    )

    print_section(f"{subject_id} DOWNSTREAM COMPLETE")
    print(f"Template column files:       {column_count}")
    print(f"Raw profile files:           {count_files(raw_profile_dir, '*.npz')}")
    print(f"Cleaned profile files:       {count_files(profile_dir, '*.npz')}")
    print(f"Regional MD mean files:      {regional_mean_count}")
    print(f"Regional QA summary:         {qa_file}")
    print(f"Analysis output:             {analysis_dir}")


if __name__ == "__main__":
    main()
