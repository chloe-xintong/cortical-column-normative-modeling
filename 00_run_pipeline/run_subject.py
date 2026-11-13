#!/usr/bin/env python3
"""Run the Complete Single-Subject Cortical-Column Pipeline

Description
-----------
Run T1 reconstruction, cohort-specific DWI preprocessing, DWI-to-T1
registration, fsaverage-indexed column construction, metric sampling, and
optional regional profile generation for one subject.

Methods
-------
1. Identify ADRC or ADD from the subject ID prefix.
2. Run or reuse the FreeSurfer cortical reconstruction.
3. Run or reuse DWI preprocessing and tensor-derived metrics.
4. Create the masked DWI registration reference and run bbregister.
5. Construct native-T1 columns indexed by fsaverage vertices.
6. Sample MD, FA, AD, and RD along the columns.
7. Optionally clean profiles and calculate regional means.

Inputs
------
- Subject ID, raw T1 image, raw DWI, bvec, and bval
- Subject output directory and optional external processed-data roots
- ADRC only: reverse-phase b0, phase-encoding direction, and readout time
- Optional template, thread count, and resume flags

Outputs
-------
- FreeSurfer reconstruction and processed diffusion metrics
- DWI registration reference and DWI-to-T1 transform
- Left/right fsaverage-indexed column files
- Raw and optional cleaned vertex-level profiles and regional profiles

Usage
-----
python 00_run_pipeline/run_subject.py     --subject-id D0007     --t1 /path/to/T1.nii.gz     --dwi /path/to/DWI.nii.gz     --bvec /path/to/D0007.bvec     --bval /path/to/D0007.bval     --reverse-phase /path/to/reverse_b0.nii.gz     --pe-dir PA     --readout-time 0.076191     --output-dir /path/to/output/D0007

Notes
-----
- ``--skip-freesurfer`` and ``--skip-dwi-preprocess`` reuse existing outputs.
- ``--before-clean`` stops after raw fsaverage vertex profile generation.
- Required neuroimaging software environments must already be loaded.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RUN_PREPROCESS = (
    PROJECT_ROOT
    / "00_run_pipeline"
    / "run_preprocess.py"
)

RUN_ANALYSIS = (
    PROJECT_ROOT
    / "00_run_pipeline"
    / "run_analysis.py"
)

RUN_FREESURFER = (
    PROJECT_ROOT
    / "01_preprocessing"
    / "t1"
    / "run_freesurfer.sh"
)

PREPARE_REGISTRATION_REFERENCE = (
    PROJECT_ROOT
    / "02_registration"
    / "prepare_registration_reference.sh"
)


# ======================================================================
# UTILITIES
# ======================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require_file(path, label):
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    return path


def require_dir(path, label):
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.is_dir():
        raise FileNotFoundError(
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


def identify_cohort(subject_id):
    """
    Identify cohort from the first character of the subject ID.
    """

    subject_id = (
        subject_id
        .strip()
        .upper()
    )

    if not subject_id:
        raise ValueError(
            "Subject ID is empty."
        )

    if subject_id.startswith("D"):
        return "ADRC"

    if subject_id.startswith("S"):
        return "ADD"

    raise ValueError(
        f"Unknown subject ID: {subject_id}\n"
        f"Expected a D-series or S-series subject."
    )


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete cortical column pipeline "
            "for one subject."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Common inputs
    # ------------------------------------------------------------------

    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject identifier, e.g. D0007 or S00775.",
    )

    parser.add_argument(
        "--t1",
        required=True,
        help="Raw T1-weighted image.",
    )

    parser.add_argument(
        "--dwi",
        required=True,
        help="Raw 4D diffusion image.",
    )

    parser.add_argument(
        "--bvec",
        required=True,
        help="FSL-format diffusion bvec file.",
    )

    parser.add_argument(
        "--bval",
        required=True,
        help="FSL-format diffusion bval file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Subject-level output directory."
        ),
    )

    parser.add_argument(
        "--processed-dwi-root",
        help="Root containing processed DWI subject directories.",
    )
    parser.add_argument(
        "--processed-freesurfer-root",
        help="FreeSurfer SUBJECTS_DIR containing processed subject directories.",
    )
    parser.add_argument(
        "--analysis-root",
        help="Root for subject-level downstream analysis outputs.",
    )

    # ------------------------------------------------------------------
    # D-series / ADRC only
    # ------------------------------------------------------------------

    parser.add_argument(
        "--reverse-phase",
        default=None,
        help=(
            "Reverse phase-encoded image. "
            "Required for D-series subjects."
        ),
    )

    parser.add_argument(
        "--pe-dir",
        default=None,
        help=(
            "Phase-encoding direction. "
            "Required for D-series subjects."
        ),
    )

    parser.add_argument(
        "--readout-time",
        type=float,
        default=None,
        help=(
            "Total readout time in seconds. "
            "Required for D-series subjects."
        ),
    )

    parser.add_argument(
        "--before-clean",
        action="store_true",
        help="Generate all four raw metric profiles and stop before cleaning.",
    )

    parser.add_argument(
        "--template",
        choices=("fsaverage", "fsaverage6"),
        default="fsaverage",
        help="Template whose vertices define the subject-specific columns.",
    )

    # ------------------------------------------------------------------
    # FreeSurfer
    # ------------------------------------------------------------------

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help=(
            "Number of FreeSurfer OpenMP threads. "
            "Default: 8."
        ),
    )

    # ------------------------------------------------------------------
    # Resume options
    # ------------------------------------------------------------------

    parser.add_argument(
        "--skip-freesurfer",
        action="store_true",
        help=(
            "Skip FreeSurfer and use an existing "
            "reconstruction in preprocess/freesurfer/."
        ),
    )

    parser.add_argument(
        "--skip-dwi-preprocess",
        action="store_true",
        help=(
            "Skip diffusion preprocessing and use existing "
            "outputs in preprocess/diffusion/."
        ),
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    subject_id = (
        args.subject_id
        .strip()
        .upper()
    )

    cohort = identify_cohort(
        subject_id
    )


    # ==================================================================
    # VALIDATE INPUTS
    # ==================================================================

    t1 = require_file(
        args.t1,
        "T1 image",
    )

    dwi = require_file(
        args.dwi,
        "Raw DWI",
    )

    bvec = require_file(
        args.bvec,
        "bvec",
    )

    bval = require_file(
        args.bval,
        "bval",
    )

    require_file(
        RUN_PREPROCESS,
        "Preprocessing runner",
    )

    require_file(
        RUN_ANALYSIS,
        "Analysis runner",
    )

    require_file(
        RUN_FREESURFER,
        "FreeSurfer script",
    )

    require_file(
        PREPARE_REGISTRATION_REFERENCE,
        "Registration-reference preparation script",
    )


    # ------------------------------------------------------------------
    # D-series-specific validation
    # ------------------------------------------------------------------

    reverse_phase = None

    if cohort == "ADRC":

        if args.reverse_phase is None:
            raise ValueError(
                "D-series subjects require --reverse-phase."
            )

        if args.pe_dir is None:
            raise ValueError(
                "D-series subjects require --pe-dir."
            )

        if args.readout_time is None:
            raise ValueError(
                "D-series subjects require --readout-time."
            )

        reverse_phase = require_file(
            args.reverse_phase,
            "Reverse phase image",
        )


    # ==================================================================
    # OUTPUT STRUCTURE
    # ==================================================================

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    preprocess_dir = output_dir / "preprocess"
    diffusion_dir = (
        Path(args.processed_dwi_root).expanduser().resolve() / subject_id
        if args.processed_dwi_root
        else preprocess_dir / "diffusion"
    )
    subjects_dir = (
        Path(args.processed_freesurfer_root).expanduser().resolve()
        if args.processed_freesurfer_root
        else preprocess_dir / "freesurfer"
    )
    analysis_dir = (
        Path(args.analysis_root).expanduser().resolve() / subject_id
        if args.analysis_root
        else output_dir / "analysis"
    )

    registration_dir = (
        analysis_dir
        / "registration"
    )


    for directory in [
        output_dir,
        preprocess_dir,
        diffusion_dir,
        subjects_dir,
        analysis_dir,
        registration_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ==================================================================
    # RUN SUMMARY
    # ==================================================================

    print_section(
        "CORTICAL COLUMN SUBJECT PIPELINE"
    )

    print(
        f"Subject:          {subject_id}"
    )

    print(
        f"Cohort:           {cohort}"
    )

    print(
        f"T1:               {t1}"
    )

    print(
        f"Raw DWI:          {dwi}"
    )

    print(
        f"bvec:             {bvec}"
    )

    print(
        f"bval:             {bval}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"FreeSurfer dir:   {subjects_dir}"
    )

    print(
        f"Diffusion dir:    {diffusion_dir}"
    )

    print(
        f"Analysis dir:     {analysis_dir}"
    )


    # ==================================================================
    # STEP 1
    # FREESURFER
    # ==================================================================

    if args.skip_freesurfer:

        print_section(
            "STEP 1: FREESURFER — SKIPPED"
        )

        print(
            "Using existing FreeSurfer reconstruction."
        )

    else:

        run_command(
            [
                "bash",
                RUN_FREESURFER,
                subject_id,
                t1,
                subjects_dir,
                args.threads,
            ],
            "STEP 1: FREESURFER RECONSTRUCTION",
        )


    # ------------------------------------------------------------------
    # Validate FreeSurfer output.
    # ------------------------------------------------------------------

    fs_subject_dir = require_dir(
        subjects_dir / subject_id,
        "FreeSurfer subject directory",
    )

    required_fs_files = [
        fs_subject_dir / "surf" / "lh.white",
        fs_subject_dir / "surf" / "lh.pial",
        fs_subject_dir / "surf" / "rh.white",
        fs_subject_dir / "surf" / "rh.pial",
        fs_subject_dir / "label" / "lh.aparc.annot",
        fs_subject_dir / "label" / "rh.aparc.annot",
        fs_subject_dir / "mri" / "orig.mgz",
    ]

    for path in required_fs_files:
        require_file(
            path,
            "Required FreeSurfer output",
        )


    # ==================================================================
    # STEP 2
    # DIFFUSION PREPROCESSING + TENSOR FITTING
    # ==================================================================

    if args.skip_dwi_preprocess:

        print_section(
            "STEP 2: DWI PREPROCESSING — SKIPPED"
        )

        print(
            "Using existing diffusion preprocessing outputs."
        )

    else:

        preprocess_command = [
            sys.executable,
            RUN_PREPROCESS,
            "--subject-id",
            subject_id,
            "--dwi",
            dwi,
            "--bvec",
            bvec,
            "--bval",
            bval,
            "--output-dir",
            output_dir,
            "--diffusion-dir",
            diffusion_dir,
        ]

        if cohort == "ADRC":

            preprocess_command.extend(
                [
                    "--reverse-phase",
                    reverse_phase,
                    "--pe-dir",
                    args.pe_dir,
                    "--readout-time",
                    args.readout_time,
                ]
            )

        run_command(
            preprocess_command,
            "STEP 2: DIFFUSION PREPROCESSING AND TENSOR FITTING",
        )


    # ==================================================================
    # VALIDATE DIFFUSION METRIC OUTPUTS
    # ==================================================================

    adc_path = diffusion_dir / f"{subject_id}_adc.nii.gz"
    legacy_md_path = diffusion_dir / f"{subject_id}_md.nii.gz"

    if adc_path.is_file():
        adc_image = require_file(
            adc_path,
            "MD image stored in the ADC output file",
        )
    elif legacy_md_path.is_file():
        adc_image = require_file(
            legacy_md_path,
            "legacy MD image",
        )
        print(
            "Using legacy MD filename for existing preprocessing output: "
            f"{legacy_md_path}"
        )
    else:
        adc_image = require_file(
            adc_path,
            "MD image stored in the ADC output file",
        )

    metric_images = {
        "md": adc_image,
        "fa": require_file(
            diffusion_dir / f"{subject_id}_fa.nii.gz",
            "FA image",
        ),
        "ad": require_file(
            diffusion_dir / f"{subject_id}_ad.nii.gz",
            "AD image",
        ),
        "rd": require_file(
            diffusion_dir / f"{subject_id}_rd.nii.gz",
            "RD image",
        ),
    }


    # ==================================================================
    # STEP 3
    # SELECT REGISTRATION INPUTS
    # ==================================================================

    print_section(
        "STEP 3: PREPARE REGISTRATION INPUTS"
    )


    # ------------------------------------------------------------------
    # ADRC / D-series
    #
    # Registration uses the final bias-corrected DWI and the BET f=0.30
    # eddy mask produced during ADRC preprocessing.
    # ------------------------------------------------------------------

    if cohort == "ADRC":

        registration_dwi = require_file(
            diffusion_dir
            / f"{subject_id}_05_biascorrected.mif",
            "ADRC preprocessed DWI",
        )

        registration_mask = require_file(
            diffusion_dir
            / f"{subject_id}_corrected_mean_b0_brain_mask.nii.gz",
            "ADRC registration mask",
        )


    # ------------------------------------------------------------------
    # ADD / S-series
    #
    # The current ADD workflow writes the final bias-corrected DWI as MIF,
    # with gradients embedded, plus a mask in the same grid.  Use those
    # products directly; the removed legacy workflow expected 0.5-mm files
    # that the current preprocessor never creates.
    # ------------------------------------------------------------------

    else:

        registration_dwi = require_file(
            diffusion_dir
            / f"{subject_id}_05_biascorrected.mif",
            "ADD bias-corrected DWI",
        )

        registration_mask = require_file(
            diffusion_dir
            / f"{subject_id}_corrected_mean_b0_brain_mask.nii.gz",
            "ADD registration mask",
        )


    print()
    print(
        f"Registration DWI:  {registration_dwi}"
    )

    print(
        f"Registration mask: {registration_mask}"
    )


    # ==================================================================
    # STEP 4
    # CREATE 3D REGISTRATION REFERENCE
    # ==================================================================

    run_command(
        [
            "bash",
            PREPARE_REGISTRATION_REFERENCE,
            registration_dwi,
            registration_dir,
            subject_id,
            registration_mask,
        ],
        "STEP 4: CREATE REGISTRATION REFERENCE",
    )


    registration_reference = require_file(
        registration_dir
        / f"{subject_id}_dwi_masked.nii.gz",
        "3D registration reference",
    )


    # ==================================================================
    # STEP 5
    # DOWNSTREAM CORTICAL COLUMN ANALYSIS
    # ==================================================================

    run_command(
        [
            sys.executable,
            RUN_ANALYSIS,
            "--subject-id",
            subject_id,
            "--subjects-dir",
            subjects_dir,
            "--registration-reference",
            registration_reference,
            "--analysis-dir",
            analysis_dir,
            "--template",
            args.template,
            "--md-image",
            metric_images["md"],
            "--fa-image",
            metric_images["fa"],
            "--ad-image",
            metric_images["ad"],
            "--rd-image",
            metric_images["rd"],
            *(["--before-clean"] if args.before_clean else []),
        ],
        "STEP 5: CORTICAL COLUMN ANALYSIS",
    )


    # ==================================================================
    # FINAL CHECK
    # ==================================================================

    transform = require_file(
        registration_dir
        / f"{subject_id}_DWI2T1.dat",
        "Final DWI-to-T1 transform",
    )

    if args.before_clean:
        raw_profile_dir = require_dir(
            analysis_dir / "profiles_before_clean",
            "Raw fsaverage vertex profile directory",
        )
        for metric_name in metric_images:
            for hemisphere in ("lh", "rh"):
                require_file(
                    raw_profile_dir
                    / f"{subject_id}_{hemisphere}_{metric_name}_profiles.npz",
                    "Raw fsaverage vertex profile",
                )
        regional_profile_dir = None
    else:
        regional_profile_dir = require_dir(
            analysis_dir / "regional_profiles",
            "Regional profile directory",
        )


    # ==================================================================
    # COMPLETE
    # ==================================================================

    print_section(
        f"{subject_id} PIPELINE COMPLETE"
    )

    print(
        f"Subject: {subject_id}"
    )

    print(
        f"Cohort:  {cohort}"
    )

    print()

    print(
        "ADC / MD:"
    )

    print(
        adc_image
    )

    print()

    print(
        "Registration reference:"
    )

    print(
        registration_reference
    )

    print()

    print(
        "DWI -> T1 transform:"
    )

    print(
        transform
    )

    print()

    print(
        "Preprocessing output:"
    )

    print(
        preprocess_dir
    )

    print()

    print(
        "Analysis output:"
    )

    print(
        analysis_dir
    )

    print()

    if regional_profile_dir is not None:
        print("Regional profiles:")
        print(regional_profile_dir)
    else:
        print("Raw fsaverage vertex profiles:")
        print(raw_profile_dir)


if __name__ == "__main__":
    main()
