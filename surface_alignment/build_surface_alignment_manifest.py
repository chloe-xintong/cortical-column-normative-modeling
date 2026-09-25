#!/usr/bin/env python3
"""Build a strict manifest for native-column to fsaverage batch mapping."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = ("md", "fa", "ad", "rd")
HEMISPHERES = ("lh", "rh")


def region_set(profile_dir: Path, subject: str, hemi: str, metric: str) -> set[str]:
    prefix = f"{subject}_{hemi}_"
    suffix = f"_{metric}.csv"
    return {p.name[len(prefix):-len(suffix)] for p in profile_dir.glob(f"{prefix}*{suffix}")}


def problem_for_subject(subject: str, subjects_dir: Path, profiles_root: Path, mapping_root: Path) -> str:
    if (mapping_root / subject / f"{subject}_mapping_qc.csv").is_file():
        return "mapping_output_already_exists"
    subject_dir = subjects_dir / subject
    profile_dir = profiles_root / subject / "profiles"
    required = [profile_dir / "bad_trajectory_mask.json"]
    for hemi in HEMISPHERES:
        required += [
            subject_dir / "surf" / f"{hemi}.white",
            subject_dir / "surf" / f"{hemi}.pial",
            subject_dir / "surf" / f"{hemi}.sphere.reg",
            subject_dir / "label" / f"{hemi}.aparc.annot",
        ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return f"missing_required_input:{missing[0]}"
    for hemi in HEMISPHERES:
        md_regions = region_set(profile_dir, subject, hemi, "md")
        if not md_regions:
            return f"no_md_profiles:{hemi}"
        if len(md_regions) != 34:
            return f"unexpected_md_region_count:{hemi}:{len(md_regions)}"
        for metric in METRICS[1:]:
            if region_set(profile_dir, subject, hemi, metric) != md_regions:
                return f"metric_region_set_mismatch:{hemi}:{metric}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    args = parser.parse_args()
    included, excluded = [], []
    for p in sorted(args.profiles_root.glob("D*/profiles")):
        subject = p.parent.name
        issue = problem_for_subject(subject, args.subjects_dir, args.profiles_root, args.mapping_root)
        (excluded if issue else included).append({"subject_id": subject, "reason": issue} if issue else {"subject_id": subject})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id"])
        writer.writeheader(); writer.writerows(included)
    with args.excluded.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "reason"])
        writer.writeheader(); writer.writerows(excluded)
    print(f"Eligible: {len(included)}")
    print(f"Excluded: {len(excluded)}")
    print(f"Manifest: {args.manifest}")
    print(f"Excluded report: {args.excluded}")


if __name__ == "__main__":
    main()
