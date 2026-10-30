#!/usr/bin/env python3
"""Resample native FreeSurfer curvature to fsaverage with mri_surf2surf."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_surface_resampling(
    source: Path,
    target: Path,
    subject_id: str,
    hemisphere: str,
    subjects_dir: Path,
) -> None:
    command = [
        "mri_surf2surf",
        "--srcsubject",
        subject_id,
        "--trgsubject",
        "fsaverage",
        "--hemi",
        hemisphere,
        "--sval",
        str(source),
        "--tval",
        str(target),
        "--mapmethod",
        "nnfr",
    ]
    environment = os.environ.copy()
    environment["SUBJECTS_DIR"] = str(subjects_dir)
    subprocess.run(
        command,
        check=True,
        env=environment,
    )


def main() -> None:
    args = parse_args()
    source = args.subjects_dir / args.subject_id / "surf" / (
        f"{args.hemisphere}.curv"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signed_output = args.output_dir / f"{args.hemisphere}.signed_curvature.mgh"
    absolute_output = args.output_dir / f"{args.hemisphere}.absolute_curvature.mgh"

    run_surface_resampling(
        source,
        signed_output,
        args.subject_id,
        args.hemisphere,
        args.subjects_dir,
    )
    native_curvature = np.asarray(
        nib.freesurfer.io.read_morph_data(str(source)),
        dtype=float,
    )
    with tempfile.TemporaryDirectory(prefix="o2_fs_curvature_") as directory:
        absolute_source = Path(directory) / f"{args.hemisphere}.abs.curv"
        nib.freesurfer.io.write_morph_data(
            str(absolute_source),
            np.abs(native_curvature),
        )
        run_surface_resampling(
            absolute_source,
            absolute_output,
            args.subject_id,
            args.hemisphere,
            args.subjects_dir,
        )


if __name__ == "__main__":
    main()
