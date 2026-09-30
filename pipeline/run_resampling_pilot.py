#!/usr/bin/env python3
"""Run one subject of the surface-resampling pilot with resume support."""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
from pathlib import Path


DEPTHS = "0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = rows[args.row_index]
    subject = row["subject_id"]
    total = 84
    done = 0
    for template in ["fsaverage6", "fsaverage"]:
        for hemi in ["lh", "rh"]:
            output_dir = args.output_root / subject / template / hemi
            output_dir.mkdir(parents=True, exist_ok=True)
            for depth in DEPTHS.split(","):
                subprocess.run(["python", "-u", "surface_resampling/run_vol2surf.py", "--subject-id", subject, "--srcsubject", subject, "--subjects-dir", str(args.subjects_dir), "--metric-image", row["md_path"], "--reg", row["registration_path"], "--template", template, "--hemisphere", hemi, "--depths", depth, "--output-dir", str(output_dir)], check=True)
                done += 1
                logging.info("PROGRESS %d/%d subject=%s template=%s hemi=%s depth=%s", done, total, subject, template, hemi, depth)
            subprocess.run(["python", "-u", "surface_resampling/stack_depth_profiles.py", "--input-dir", str(output_dir), "--subject-id", subject, "--template", template, "--hemisphere", hemi, "--depths", DEPTHS, "--output", str(output_dir / f"{subject}_{hemi}_md_profiles.npz")], check=True)
    logging.info("Completed subject %s", subject)


if __name__ == "__main__":
    main()
