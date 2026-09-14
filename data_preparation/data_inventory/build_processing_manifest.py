#!/usr/bin/env python3
"""Create a subject manifest from the data inventory for V2 processing."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument(
        "--before-clean-only", action="store_true",
        help="Select subjects with complete preprocessing but without profiles_before_clean.",
    )
    parser.add_argument(
        "--require-label", action="store_true",
        help="Require at least one diagnosis label (for downstream analysis).",
    )
    args = parser.parse_args()

    with args.inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        subject = row["subject"]
        if row.get("preprocessing_ready") != "True":
            continue
        if args.require_label and row.get("has_any_diagnosis_label") != "True":
            continue
        if args.before_clean_only:
            if not args.output_root or not args.analysis_root:
                raise SystemExit("--before-clean-only requires --output-root and --analysis-root")
            dwi = args.output_root / subject
            complete = all((dwi / f"{subject}_{name}").is_file() for name in (
                "04_preprocessed.mif", "fa.nii.gz", "md.nii.gz", "ad.nii.gz", "rd.nii.gz"
            ))
            already_done = (args.analysis_root / subject / "profiles_before_clean").is_dir()
            if not complete or already_done:
                continue
        elif args.output_root and all((args.output_root / subject / f"{subject}_{name}").is_file() for name in (
            "rd.nii.gz", "fa.nii.gz", "md.nii.gz", "ad.nii.gz", "04_preprocessed.mif"
        )):
            continue
        selected.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject"])
        writer.writerows([[row["subject"]] for row in selected])
    print(f"Wrote {len(selected)} subjects to {args.output}")


if __name__ == "__main__":
    main()
