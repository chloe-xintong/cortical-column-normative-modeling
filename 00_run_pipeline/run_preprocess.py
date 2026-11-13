#!/usr/bin/env python3
"""Run Cohort-Specific DWI Preprocessing

Description
-----------
Dispatch one raw diffusion scan to the ADRC or ADD preprocessing workflow and
validate the final tensor-derived metric images.

Methods
-------
1. Identify the cohort from the subject ID prefix (D=ADRC, S=ADD).
2. Validate the raw DWI and FSL gradient files.
3. Run denoising, Gibbs-ringing correction, eddy correction, masking, and
   bias-field correction with the cohort-specific script.
4. Use reverse-phase topup for ADRC; use ``rpe_none`` for ADD.
5. Fit the diffusion tensor and verify ADC/MD, FA, AD, and RD outputs.

Inputs
------
- Subject ID, raw 4D DWI, bvec, and bval files
- Output directory or exact diffusion output directory
- ADRC only: reverse-phase b0, phase-encoding direction, and readout time

Outputs
-------
- Bias-corrected DWI and corrected mean-b0 brain mask
- Diffusion tensor image
- ADC/MD, FA, AD, and RD scalar images

Usage
-----
python 00_run_pipeline/run_preprocess.py     --subject-id D0007     --dwi /path/to/D0007_dwi.nii.gz     --bvec /path/to/D0007.bvec     --bval /path/to/D0007.bval     --reverse-phase /path/to/D0007_reverse_b0.nii.gz     --pe-dir PA     --readout-time 0.076191     --output-dir /path/to/D0007

Notes
-----
- ADD uses fixed protocol values PA and 0.119063 seconds in its shell script.
- FSL, MRtrix, and ANTs must already be available in the environment.
- This runner does not perform FreeSurfer reconstruction or column building.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DWI_PREPROCESS_ROOT = (
    PROJECT_ROOT
    / "01_preprocessing"
    / "dwi"
)

ADRC_PREPROCESS_SCRIPT = (
    DWI_PREPROCESS_ROOT
    / "ADRC"
    / "preprocess_dwi.sh"
)

ADD_PREPROCESS_SCRIPT = (
    DWI_PREPROCESS_ROOT
    / "ADD"
    / "preprocess_dwi.sh"
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
    path = Path(path).expanduser().resolve()

    if not path.is_file():
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
    Identify the preprocessing cohort from the first character
    of the subject ID.
    """

    subject_id = subject_id.strip().upper()

    if not subject_id:
        raise ValueError(
            "Subject ID is empty."
        )

    prefix = subject_id[0]

    if prefix == "D":
        return "ADRC"

    if prefix == "S":
        return "ADD"

    raise ValueError(
        f"Unknown subject prefix: {prefix}\n"
        f"Expected a D-series or S-series subject ID."
    )


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run cohort-specific diffusion preprocessing "
            "and tensor metric generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--subject-id",
        required=True,
        help=(
            "Subject identifier. "
            "D-series subjects use ADRC preprocessing; "
            "S-series subjects use ADD preprocessing."
        ),
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
            "Subject output directory. "
            "Preprocessing outputs are written to "
            "<output-dir>/preprocess/diffusion/."
        ),
    )
    parser.add_argument(
        "--diffusion-dir",
        help=(
            "Optional exact diffusion output directory. When omitted, use "
            "<output-dir>/preprocess/diffusion/."
        ),
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

    return parser.parse_args()


# ======================================================================
# ADRC / D-SERIES PREPROCESSING
# ======================================================================

def run_adrc_preprocessing(
    args,
    diffusion_dir,
):

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

    # ------------------------------------------------------------------
    # Run the existing ADRC preprocessing workflow.
    # ------------------------------------------------------------------

    run_command(
        [
            "bash",
            ADRC_PREPROCESS_SCRIPT,
            args.dwi,
            args.bvec,
            args.bval,
            reverse_phase,
            diffusion_dir,
            args.subject_id,
            args.pe_dir,
            args.readout_time,
        ],
        "DWI PREPROCESSING: ADRC / D-SERIES",
    )

# ======================================================================
# ADD / S-SERIES PREPROCESSING
# ======================================================================

def run_add_preprocessing(
    args,
    diffusion_dir,
):

    # ------------------------------------------------------------------
    # Run the modern ADD workflow without reverse-b0 or topup.
    # ------------------------------------------------------------------

    run_command(
        [
            "bash",
            ADD_PREPROCESS_SCRIPT,
            args.dwi,
            args.bvec,
            args.bval,
            diffusion_dir,
            args.subject_id,
        ],
        "DWI PREPROCESSING: ADD / S-SERIES",
    )



# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    args.subject_id = (
        args.subject_id
        .strip()
        .upper()
    )

    # ------------------------------------------------------------------
    # Validate common inputs.
    # ------------------------------------------------------------------

    args.dwi = require_file(
        args.dwi,
        "Raw DWI",
    )

    args.bvec = require_file(
        args.bvec,
        "bvec",
    )

    args.bval = require_file(
        args.bval,
        "bval",
    )

    require_file(
        ADRC_PREPROCESS_SCRIPT,
        "ADRC preprocessing script",
    )

    require_file(
        ADD_PREPROCESS_SCRIPT,
        "ADD preprocessing script",
    )

    require_command("bash")

    # ------------------------------------------------------------------
    # Identify cohort automatically from subject ID.
    # ------------------------------------------------------------------

    cohort = identify_cohort(
        args.subject_id
    )

    # ------------------------------------------------------------------
    # Create standard preprocessing output directory.
    # ------------------------------------------------------------------

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    diffusion_dir = (
        Path(args.diffusion_dir).expanduser().resolve()
        if args.diffusion_dir
        else output_dir / "preprocess" / "diffusion"
    )

    diffusion_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Print run information.
    # ------------------------------------------------------------------

    print_section(
        "CORTICAL COLUMN PREPROCESSING"
    )

    print(f"Subject:      {args.subject_id}")
    print(f"Cohort:       {cohort}")
    print(f"Raw DWI:      {args.dwi}")
    print(f"bvec:         {args.bvec}")
    print(f"bval:         {args.bval}")
    print(f"Output:       {diffusion_dir}")

    # ------------------------------------------------------------------
    # Run cohort-specific preprocessing.
    # ------------------------------------------------------------------

    if cohort == "ADRC":

        run_adrc_preprocessing(
            args,
            diffusion_dir,
        )

    elif cohort == "ADD":

        run_add_preprocessing(
            args,
            diffusion_dir,
        )

    # ------------------------------------------------------------------
    # Final output check.
    # ------------------------------------------------------------------

    adc_image = (
        diffusion_dir
        / f"{args.subject_id}_adc.nii.gz"
    )

    fa_image = (
        diffusion_dir
        / f"{args.subject_id}_fa.nii.gz"
    )

    ad_image = (
        diffusion_dir
        / f"{args.subject_id}_ad.nii.gz"
    )

    rd_image = (
        diffusion_dir
        / f"{args.subject_id}_rd.nii.gz"
    )

    require_file(
        adc_image,
        "ADC image",
    )

    require_file(
        fa_image,
        "FA image",
    )

    require_file(
        ad_image,
        "AD image",
    )

    require_file(
        rd_image,
        "RD image",
    )

    print_section(
        "PREPROCESSING COMPLETED"
    )

    print("ADC / MD:")
    print(adc_image)

    print()
    print("FA:")
    print(fa_image)

    print()
    print("AD:")
    print(ad_image)

    print()
    print("RD:")
    print(rd_image)


if __name__ == "__main__":
    main()
