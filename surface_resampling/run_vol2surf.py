#!/usr/bin/env python3
"""Sample a diffusion metric volume onto a FreeSurfer template surface."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np


EXPECTED_VERTICES = {"fsaverage": 163842, "fsaverage6": 40962}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--srcsubject")
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--metric-image", type=Path, required=True)
    parser.add_argument("--reg", type=Path, required=True)
    parser.add_argument("--template", choices=EXPECTED_VERTICES, required=True)
    parser.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interp", default="trilinear")
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> None:
    logging.info("Running: %s", " ".join(command))
    with log_path.open("w") as log_file:
        result = subprocess.run(command, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}: {log_path}")


def validate_mgh(path: Path, expected_vertices: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty output: {path}")
    values = np.asarray(nib.load(str(path)).dataobj).reshape(-1)
    if values.size != expected_vertices:
        raise ValueError(f"Expected {expected_vertices} vertices in {path}, got {values.size}")


def main() -> None:
    args = parse_args()
    if args.srcsubject is None:
        args.srcsubject = args.subject_id
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    depths = [float(value) for value in args.depths.split(",")]
    expected = EXPECTED_VERTICES[args.template]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fsaverage_dir = args.subjects_dir / args.template
    if not fsaverage_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {fsaverage_dir}")
    if shutil.which("mri_vol2surf") is None:
        raise FileNotFoundError("mri_vol2surf is not available in PATH")

    manifest = []
    for index, depth in enumerate(depths, start=1):
        stem = f"{args.subject_id}_{args.hemisphere}_md_depth-{depth:.2f}"
        output_path = args.output_dir / f"{stem}.mgh"
        log_path = args.output_dir / f"{stem}.log"
        started = time.time()
        if output_path.exists():
            try:
                validate_mgh(output_path, expected)
                logging.info("[%d/%d] Existing valid output: %s", index, len(depths), output_path)
                status = "skipped_existing"
            except (OSError, ValueError):
                output_path.unlink()
                status = "rerun"
        else:
            status = "completed"
        if not output_path.exists():
            command = [
                "mri_vol2surf",
                "--mov", str(args.metric_image),
                "--reg", str(args.reg),
                "--srcsubject", args.srcsubject,
                "--hemi", args.hemisphere,
                "--trgsubject", args.template,
                "--projfrac", f"{depth:.2f}",
                "--interp", args.interp,
                "--o", str(output_path),
            ]
            run_command(command, log_path)
        validate_mgh(output_path, expected)
        manifest.append({"depth": depth, "output": str(output_path), "status": status, "elapsed_s": time.time() - started})
        logging.info("[%d/%d] Validated %s", index, len(depths), output_path.name)

    metadata = {
        "subject_id": args.subject_id,
        "template": args.template,
        "hemisphere": args.hemisphere,
        "metric": "md",
        "metric_image": str(args.metric_image),
        "registration": str(args.reg),
        "interp": args.interp,
        "expected_vertices": expected,
        "outputs": manifest,
    }
    (args.output_dir / "vol2surf_manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
