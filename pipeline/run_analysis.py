#!/usr/bin/env python3

"""
Run cortical-column downstream analysis after preprocessing.

This runner does NOT perform T1 or DWI preprocessing.

It uses the completed preprocessing outputs and calls the existing
downstream modules in this repository:

    registration/register_dwi_to_t1.sh
    trajectory_processing/build_trajectories.py
    trajectory_processing/transform_trajectories.py
    trajectory_processing/assign_regions.py
    profile_sampling/sample_profiles.py
    profile_sampling/clean_bad_trajectories.py
    profile_sampling/regional_profiles.py


======================================================================
EXPECTED OUTPUT STRUCTURE
======================================================================

<analysis_dir>/
├── registration/
├── trajectories/
├── regions/
├── profiles/
└── regional_profiles/


======================================================================
REQUIRED INPUTS
======================================================================

1. FreeSurfer SUBJECTS_DIR

   The directory must contain:

       <subjects_dir>/<subject_id>/

   with:

       surf/lh.white
       surf/lh.pial
       surf/rh.white
       surf/rh.pial
       label/lh.aparc.annot
       label/rh.aparc.annot
       mri/orig.mgz


2. ADC image

   This is the diffusivity map used for cortical sampling.

   Example:

       D0007_adc.nii.gz

   or, for the currently reconstructed S00775 workflow, the physical
   input file may still be named:

       S00775_md_new.nii.gz

   but downstream analysis is labeled consistently as "adc".


3. Registration reference

   3D masked diffusion-space image used as the moving image for
   bbregister.

   Example:

       S00775_dwi_sum_masked_upsampled.nii.gz


======================================================================
USAGE
======================================================================

python pipeline/run_analysis.py \
    --subject-id S00775 \
    --subjects-dir /path/to/freesurfer \
    --adc-image /path/to/S00775_adc.nii.gz \
    --registration-reference /path/to/S00775_dwi_sum_masked.nii.gz \
    --analysis-dir /path/to/S00775/analysis


D0007 example:

python pipeline/run_analysis.py \
    --subject-id D0007 \
    --subjects-dir /path/to/D0007/freesurfer \
    --adc-image /path/to/D0007_adc.nii.gz \
    --registration-reference /path/to/D0007_registration_reference.nii.gz \
    --analysis-dir /path/to/D0007/analysis


======================================================================
IMPORTANT
======================================================================

FreeSurfer must already be loaded before running this script.

Check:

    which bbregister

The script does not source FreeSurfer automatically because the
FreeSurfer installation differs across machines.

"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ======================================================================
# CONSTANTS
# ======================================================================

DEPTH_POINTS = 21
EXPECTED_REGIONS = 68


# ======================================================================
# UTILITIES
# ======================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require_file(path, label):
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    return path


def require_dir(path, label):
    path = Path(path).expanduser().resolve()

    if not path.is_dir():
        raise NotADirectoryError(
            f"{label} not found:\n{path}"
        )

    return path


def require_command(command):
    executable = shutil.which(command)

    if executable is None:
        raise RuntimeError(
            f"Required command not available: {command}\n"
            f"Load the required software environment first."
        )

    return executable


def run_command(command, step_name):
    print_section(step_name)

    command = [
        str(item)
        for item in command
    ]

    print("Command:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with "
            f"exit code {result.returncode}"
        )


def count_files(directory, pattern):
    return len(
        list(
            Path(directory).glob(pattern)
        )
    )


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run cortical-column downstream analysis "
            "from completed preprocessing outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject identifier, e.g. D0007 or S00775.",
    )

    parser.add_argument(
        "--subjects-dir",
        required=True,
        help=(
            "FreeSurfer SUBJECTS_DIR containing "
            "<subject-id>/."
        ),
    )

    parser.add_argument(
        "--adc-image",
        required=True,
        help=(
            "ADC / mean diffusivity image used for "
            "cortical profile sampling."
        ),
    )

    for metric in ("md", "fa", "ad", "rd"):
        parser.add_argument(
            f"--{metric}-image",
            help=f"{metric.upper()} image for cortical profile sampling.",
        )

    parser.add_argument(
        "--before-clean",
        action="store_true",
        help=(
            "Sample all supplied metrics into profiles_before_clean and "
            "stop before bad-column cleaning."
        ),
    )

    parser.add_argument(
        "--registration-reference",
        required=True,
        help=(
            "3D masked diffusion-space image used "
            "as the bbregister moving image."
        ),
    )

    parser.add_argument(
        "--analysis-dir",
        required=True,
        help=(
            "Output directory for downstream analysis."
        ),
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    subject_id = args.subject_id


    # ------------------------------------------------------------------
    # Repository root
    # ------------------------------------------------------------------

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )


    # ------------------------------------------------------------------
    # Input paths
    # ------------------------------------------------------------------

    subjects_dir = require_dir(
        args.subjects_dir,
        "FreeSurfer SUBJECTS_DIR",
    )

    subject_dir = require_dir(
        subjects_dir / subject_id,
        "FreeSurfer subject directory",
    )

    adc_image = require_file(
        args.adc_image,
        "ADC image",
    )

    metric_images = {"adc": adc_image}
    for metric in ("md", "fa", "ad", "rd"):
        image = getattr(args, f"{metric}_image")
        if image:
            metric_images[metric] = require_file(image, f"{metric.upper()} image")
    if args.before_clean and not {"md", "fa", "ad", "rd"}.issubset(metric_images):
        missing = ", ".join(sorted({"md", "fa", "ad", "rd"} - set(metric_images)))
        raise ValueError(f"--before-clean requires all four metric images; missing: {missing}")

    registration_reference = require_file(
        args.registration_reference,
        "Registration reference",
    )


    # ------------------------------------------------------------------
    # Validate required FreeSurfer outputs
    # ------------------------------------------------------------------

    freesurfer_required = [
        subject_dir / "surf" / "lh.white",
        subject_dir / "surf" / "lh.pial",
        subject_dir / "surf" / "rh.white",
        subject_dir / "surf" / "rh.pial",
        subject_dir / "label" / "lh.aparc.annot",
        subject_dir / "label" / "rh.aparc.annot",
        subject_dir / "mri" / "orig.mgz",
    ]

    for path in freesurfer_required:
        require_file(
            path,
            "Required FreeSurfer output",
        )


    # ------------------------------------------------------------------
    # Required external software
    # ------------------------------------------------------------------

    require_command(
        "bbregister"
    )


    # ------------------------------------------------------------------
    # Existing pipeline scripts
    # ------------------------------------------------------------------

    register_script = require_file(
        repo_root
        / "registration"
        / "register_dwi_to_t1.sh",
        "Registration script",
    )

    build_trajectories_script = require_file(
        repo_root
        / "trajectory_processing"
        / "build_trajectories.py",
        "Trajectory builder",
    )

    transform_trajectories_script = require_file(
        repo_root
        / "trajectory_processing"
        / "transform_trajectories.py",
        "Trajectory transformation script",
    )

    assign_regions_script = require_file(
        repo_root
        / "trajectory_processing"
        / "assign_regions.py",
        "Region assignment script",
    )

    sample_profiles_script = require_file(
        repo_root
        / "profile_sampling"
        / "sample_profiles.py",
        "Profile sampling script",
    )

    clean_bad_trajectories_script = require_file(
        repo_root
        / "profile_sampling"
        / "clean_bad_trajectories.py",
        "Bad-column cleaning script",
    )

    regional_profiles_script = require_file(
        repo_root
        / "profile_sampling"
        / "regional_profiles.py",
        "Regional profile script",
    )


    # ------------------------------------------------------------------
    # Analysis output structure
    # ------------------------------------------------------------------

    analysis_dir = (
        Path(args.analysis_dir)
        .expanduser()
        .resolve()
    )

    registration_dir = (
        analysis_dir
        / "registration"
    )

    trajectory_dir = (
        analysis_dir
        / "trajectories"
    )

    region_dir = (
        analysis_dir
        / "regions"
    )

    profile_dir = analysis_dir / (
        "profiles_before_clean" if args.before_clean else "profiles"
    )

    regional_profile_dir = (
        analysis_dir
        / "regional_profiles"
    )


    for directory in [
        registration_dir,
        trajectory_dir,
        region_dir,
        profile_dir,
        regional_profile_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------

    print_section(
        "CORTICAL COLUMN ANALYSIS"
    )

    print(
        f"Subject:                 {subject_id}"
    )

    print(
        f"Repository:              {repo_root}"
    )

    print(
        f"FreeSurfer SUBJECTS_DIR: {subjects_dir}"
    )

    print(
        f"ADC image:               {adc_image}"
    )

    print(
        f"Registration reference:  {registration_reference}"
    )

    print(
        f"Analysis directory:      {analysis_dir}"
    )


    # ==================================================================
    # STEP 1
    # DWI -> T1 REGISTRATION
    # ==================================================================

    transform = (
        registration_dir
        / f"{subject_id}_DWI2T1.dat"
    )


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


    require_file(
        transform,
        "bbregister transform",
    )


    # ==================================================================
    # STEP 2
    # BUILD CORTICAL TRAJECTORIES
    # ==================================================================

    run_command(
        [
            sys.executable,
            build_trajectories_script,
            "--subject-id",
            subject_id,
            "--subject-dir",
            subject_dir,
            "--output-dir",
            trajectory_dir,
            "--points",
            str(DEPTH_POINTS),
        ],
        "STEP 2: BUILD CORTICAL TRAJECTORIES",
    )


    # Validate expected T1-space outputs.

    pair_lh = require_file(
        trajectory_dir
        / f"{subject_id}_pair_lh.mat",
        "LH cortical pair file",
    )

    pair_rh = require_file(
        trajectory_dir
        / f"{subject_id}_pair_rh.mat",
        "RH cortical pair file",
    )

    trajectory_lh = require_file(
        trajectory_dir
        / f"{subject_id}_trajectory_lh.mat",
        "LH T1-space trajectory file",
    )

    trajectory_rh = require_file(
        trajectory_dir
        / f"{subject_id}_trajectory_rh.mat",
        "RH T1-space trajectory file",
    )


    # ==================================================================
    # STEP 3
    # TRANSFORM TRAJECTORIES TO DWI VOXEL SPACE
    # ==================================================================

    run_command(
        [
            sys.executable,
            transform_trajectories_script,
            "--subject-id",
            subject_id,
            "--trajectory-dir",
            trajectory_dir,
            "--transform",
            transform,
            "--dwi",
            registration_reference,
            "--output-dir",
            trajectory_dir,
        ],
        "STEP 3: TRANSFORM TRAJECTORIES TO DWI SPACE",
    )


    trajectory_dwi_lh = require_file(
        trajectory_dir
        / f"{subject_id}_trajectory_lh_dwi.mat",
        "LH DWI-space trajectory file",
    )

    trajectory_dwi_rh = require_file(
        trajectory_dir
        / f"{subject_id}_trajectory_rh_dwi.mat",
        "RH DWI-space trajectory file",
    )


    # ==================================================================
    # STEP 4
    # REGION ASSIGNMENT
    # ==================================================================

    run_command(
        [
            sys.executable,
            assign_regions_script,
            "--subject-id",
            subject_id,
            "--subjects-dir",
            subjects_dir,
            "--trajectory-dir",
            trajectory_dir,
            "--output-dir",
            region_dir,
        ],
        "STEP 4: ASSIGN CORTICAL REGIONS",
    )


    region_mat_count = count_files(
        region_dir,
        "*.mat",
    )

    region_csv_count = count_files(
        region_dir,
        "*.csv",
    )


    if region_mat_count != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} region MAT files, "
            f"found {region_mat_count}."
        )


    if region_csv_count != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} region CSV files, "
            f"found {region_csv_count}."
        )


    # ==================================================================
    # STEP 5
    # ADC PROFILE SAMPLING
    # ==================================================================

    for metric_name, metric_image in metric_images.items():
        run_command(
            [sys.executable, sample_profiles_script, "--subject-id", subject_id,
             "--region-dir", region_dir, "--metric-image", metric_image,
             "--metric-name", metric_name, "--output-dir", profile_dir],
            f"STEP 5: SAMPLE {metric_name.upper()} PROFILES",
        )
        count = count_files(profile_dir, f"{subject_id}_*_{metric_name}.csv")
        if count != EXPECTED_REGIONS:
            raise RuntimeError(
                f"Expected {EXPECTED_REGIONS} {metric_name.upper()} profile files, "
                f"found {count}."
            )

    adc_profile_count = count_files(profile_dir, f"{subject_id}_*_adc.csv")

    if args.before_clean:
        print_section("STOP BEFORE CLEANING")
        print(f"Raw profiles saved in: {profile_dir}")
        print("No bad-column cleaning was performed.")
        return


    # ==================================================================
    # STEP 6
    # REMOVE BAD COLUMNS
    #
    # A column is considered invalid when any of its 21 ADC samples is:
    #
    #     0
    #     NaN
    #     Inf
    #
    # clean_bad_trajectories.py removes the same trajectory from:
    #
    #     profile CSV
    #     regional DWI coordinate MAT
    #     regional DWI coordinate CSV
    #
    # ==================================================================

    run_command(
        [
            sys.executable,
            clean_bad_trajectories_script,
            "--subject-id",
            subject_id,
            "--region-dir",
            region_dir,
            "--profile-dir",
            profile_dir,
            "--metrics",
            "adc",
        ],
        "STEP 6: REMOVE BAD CORTICAL COLUMNS",
    )


    bad_trajectory_file = require_file(
        profile_dir
        / "bad_trajectories.json",
        "Bad-trajectory summary",
    )


    # ==================================================================
    # STEP 7
    # REGIONAL PROFILE AGGREGATION
    # ==================================================================

    run_command(
        [
            sys.executable,
            regional_profiles_script,
            "--subject-id",
            subject_id,
            "--profile-dir",
            profile_dir,
            "--metric-name",
            "adc",
            "--output-dir",
            regional_profile_dir,
        ],
        "STEP 7: REGIONAL ADC PROFILE AGGREGATION",
    )


    region_mean_dir = require_dir(
        regional_profile_dir
        / "region_mean",
        "Regional mean profile directory",
    )


    regional_mean_count = count_files(
        region_mean_dir,
        f"{subject_id}_*_adc_mean.csv",
    )


    if regional_mean_count != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} regional ADC mean files, "
            f"found {regional_mean_count}."
        )


    qa_file = require_file(
        regional_profile_dir
        / f"{subject_id}_profiles_QA_adc.csv",
        "Regional profile QA file",
    )


    # ==================================================================
    # FINAL OUTPUT CHECK
    # ==================================================================

    pair_count = count_files(
        trajectory_dir,
        f"{subject_id}_pair_*.mat",
    )

    t1_trajectory_count = (
        int(trajectory_lh.is_file())
        +
        int(trajectory_rh.is_file())
    )

    dwi_trajectory_count = (
        int(trajectory_dwi_lh.is_file())
        +
        int(trajectory_dwi_rh.is_file())
    )


    print_section(
        f"{subject_id} DOWNSTREAM COMPLETE"
    )


    print(
        f"Pair files:                 {pair_count}"
    )

    print(
        f"T1 trajectory files:        {t1_trajectory_count}"
    )

    print(
        f"DWI trajectory files:       {dwi_trajectory_count}"
    )

    print(
        f"Region MAT files:           {region_mat_count}"
    )

    print(
        f"Region CSV files:           {region_csv_count}"
    )

    print(
        f"ADC profile files:          {adc_profile_count}"
    )

    print(
        f"Regional ADC mean files:    {regional_mean_count}"
    )

    print(
        f"Bad-column summary:         {bad_trajectory_file}"
    )

    print(
        f"Regional QA summary:        {qa_file}"
    )

    print()
    print(
        f"Transform:\n{transform}"
    )

    print()
    print(
        f"Analysis output:\n{analysis_dir}"
    )


if __name__ == "__main__":
    main()
