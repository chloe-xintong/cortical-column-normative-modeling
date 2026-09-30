#!/usr/bin/env python3
"""Resample native FreeSurfer thickness to a template surface."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--template", choices=["fsaverage", "fsaverage6"], required=True)
    parser.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if shutil.which("mri_surf2surf") is None:
        raise FileNotFoundError("mri_surf2surf is not available in PATH")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "mri_surf2surf",
        "--srcsubject", args.subject_id,
        "--trgsubject", args.template,
        "--hemi", args.hemisphere,
        "--sval", str(args.subjects_dir / args.subject_id / "surf" / f"{args.hemisphere}.thickness"),
        "--tval", str(args.output),
    ]
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)
    values = np.asarray(nib.load(str(args.output)).dataobj).reshape(-1)
    expected = 40962 if args.template == "fsaverage6" else 163842
    if values.size != expected:
        raise ValueError(f"Expected {expected} thickness vertices, got {values.size}")
    logging.info("Validated thickness output: %s", args.output)


if __name__ == "__main__":
    main()
