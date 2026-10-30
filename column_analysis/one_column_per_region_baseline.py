#!/usr/bin/env python3
"""Build one regional baseline column per subject and hemisphere.

For every subject, hemisphere, and FreeSurfer cortical region, this script
computes the mean MD profile across valid native columns and a regional
cortical-thickness summary.  The profile contains 21 depth values, with
depth 0 at the white-matter surface and depth 20 at the pial surface.

The script validates the relationship between the region column file and
the MD profile file.  It does not assume that files from different regions or
hemispheres can be combined.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_REGIONS = (
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    " lingual".strip(),
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
)

DEPTH_POINTS = 21
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--freesurfer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects-file", type=Path)
    parser.add_argument("--subject-id")
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--profile-dir-name", default="profiles_before_clean")
    parser.add_argument("--depth-points", type=int, default=DEPTH_POINTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_subjects(args: argparse.Namespace) -> list[str]:
    if args.subject_id:
        return [args.subject_id]
    if args.subjects_file:
        return [
            line.strip()
            for line in args.subjects_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    subjects = sorted(path.name for path in args.analysis_root.iterdir() if path.is_dir())
    if not subjects:
        raise FileNotFoundError(f"No subject directories found in {args.analysis_root}")
    return subjects


def load_profile(path: Path, depth_points: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",")
    values = np.atleast_2d(values).astype(float)
    if values.shape[1] != depth_points:
        raise ValueError(f"Expected {depth_points} depths in {path}, got {values.shape[1]}")
    values[~np.isfinite(values)] = np.nan
    values[values <= 0] = np.nan
    return values


def load_region_columns(path: Path, n_columns: int, depth_points: int) -> np.ndarray:
    columns = np.loadtxt(path, delimiter=",")
    if columns.ndim != 2 or columns.shape[0] != 4:
        raise ValueError(f"Expected a 4 x N region column array in {path}")
    expected_points = n_columns * depth_points
    if columns.shape[1] != expected_points:
        raise ValueError(
            f"Profile/column mismatch for {path}: "
            f"{n_columns} x {depth_points} != {columns.shape[1]}"
        )
    return columns[:3].reshape(3, n_columns, depth_points).transpose(1, 2, 0)


def load_surface_thickness(
    subject_dir: Path,
    hemisphere: str,
) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    white_path = subject_dir / "surf" / f"{hemisphere}.white"
    thickness_path = subject_dir / "surf" / f"{hemisphere}.thickness"
    white_coords, _ = nib.freesurfer.io.read_geometry(str(white_path))
    thickness = nib.freesurfer.io.read_morph_data(str(thickness_path)).astype(float)
    if len(white_coords) != len(thickness):
        raise ValueError(f"Surface/thickness vertex mismatch for {subject_dir} {hemisphere}")
    return white_coords, thickness, cKDTree(white_coords)


def thickness_from_columns(
    coordinates: np.ndarray,
    white_coords: np.ndarray,
    thickness_values: np.ndarray,
    white_tree: cKDTree,
) -> tuple[np.ndarray, np.ndarray]:
    white_points = coordinates[:, 0, :]
    _, vertex_ids = white_tree.query(white_points)
    freesurfer_thickness = thickness_values[vertex_ids]
    column_thickness = np.linalg.norm(np.diff(coordinates, axis=1), axis=2).sum(axis=1)
    return freesurfer_thickness, column_thickness


def build_row(
    subject: str,
    hemisphere: str,
    region: str,
    profiles: np.ndarray,
    coordinates: np.ndarray,
    surface_data: tuple[np.ndarray, np.ndarray, cKDTree],
) -> dict[str, object]:
    valid_profile = np.isfinite(profiles).any(axis=1)
    valid_values = profiles[valid_profile]
    if len(valid_values) == 0:
        raise ValueError("No valid MD profiles")

    white_coords, thickness_values, white_tree = surface_data
    freesurfer_thickness, column_thickness = thickness_from_columns(
        coordinates[valid_profile],
        white_coords,
        thickness_values,
        white_tree,
    )

    row: dict[str, object] = {
        "subject_id": subject,
        "hemisphere": hemisphere,
        "region": region,
        "n_columns_total": int(len(profiles)),
        "n_columns_valid": int(len(valid_values)),
        "freesurfer_thickness_mean_mm": float(np.nanmean(freesurfer_thickness)),
        "freesurfer_thickness_sd_mm": float(np.nanstd(freesurfer_thickness, ddof=1))
        if len(freesurfer_thickness) > 1
        else np.nan,
        "column_thickness_mean_mm": float(np.nanmean(column_thickness)),
        "column_thickness_sd_mm": float(np.nanstd(column_thickness, ddof=1))
        if len(column_thickness) > 1
        else np.nan,
    }
    for depth in range(profiles.shape[1]):
        row[f"md_mean_depth_{depth:02d}"] = float(np.nanmean(valid_values[:, depth]))
        row[f"md_valid_depth_{depth:02d}"] = int(np.isfinite(valid_values[:, depth]).sum())
    return row


def iter_cases(
    subjects: Iterable[str],
    hemispheres: Iterable[str],
    regions: Iterable[str],
) -> Iterable[tuple[str, str, str]]:
    for subject in subjects:
        for hemisphere in hemispheres:
            for region in regions:
                yield subject, hemisphere, region


def main() -> None:
    args = parse_args()
    configure_logging()
    subjects = read_subjects(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "one_column_per_region_baseline.csv"
    failure_path = args.output_dir / "one_column_per_region_baseline_failures.csv"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_path}")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    cases = list(iter_cases(subjects, args.hemispheres, args.regions))
    logging.info("Starting %d subject-hemi-region cases", len(cases))
    surface_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, cKDTree]] = {}

    for index, (subject, hemisphere, region) in enumerate(cases, start=1):
        try:
            subject_root = args.analysis_root / subject
            clean_marker = (
                subject_root
                / args.profile_dir_name
                / "bad_columns.json"
            )
            if not clean_marker.is_file():
                clean_marker = (
                    subject_root
                    / args.profile_dir_name
                    / "bad_trajectories.json"
                )
            if not clean_marker.is_file():
                raise FileNotFoundError(
                    "Bad-point cleaning marker is missing; subject was not "
                    "included in the baseline."
                )
            profile_path = (
                subject_root
                / args.profile_dir_name
                / f"{subject}_{hemisphere}_{region}_md.csv"
            )
            region_path = subject_root / "regions" / f"{hemisphere}_{region}.csv"
            profiles = load_profile(profile_path, args.depth_points)
            coordinates = load_region_columns(
                region_path,
                n_columns=profiles.shape[0],
                depth_points=args.depth_points,
            )
            cache_key = (subject, hemisphere)
            if cache_key not in surface_cache:
                surface_cache[cache_key] = load_surface_thickness(
                    args.freesurfer_root / subject,
                    hemisphere,
                )
            rows.append(
                build_row(
                    subject,
                    hemisphere,
                    region,
                    profiles,
                    coordinates,
                    surface_cache[cache_key],
                )
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "region": region,
                    "reason": str(error),
                }
            )

        if index % args.log_every == 0 or index == len(cases):
            logging.info(
                "Progress %d/%d: successful=%d failures=%d",
                index,
                len(cases),
                len(rows),
                len(failures),
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    pd.DataFrame(failures).to_csv(failure_path, index=False)
    logging.info("Wrote %d baseline rows to %s", len(rows), output_path)
    logging.info("Wrote %d failure rows to %s", len(failures), failure_path)


if __name__ == "__main__":
    main()
