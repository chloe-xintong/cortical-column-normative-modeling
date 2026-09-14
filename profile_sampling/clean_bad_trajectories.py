#!/usr/bin/env python

# Purpose:
# Remove invalid cortical trajectories consistently across diffusion metrics.
#
# A trajectory is marked as bad if any of its 21 depth samples
# is zero, NaN, or Inf in any requested metric.
#
# The same trajectory is then removed from:
# 1. all metric profile CSV files
# 2. the corresponding regional DWI coordinate MAT
# 3. the corresponding regional DWI coordinate CSV
#
# This reproduces the cross-contrast cleaning behavior
# of the original Brain_column_builder pipeline.
#
# Usage:
# python profile_sampling/clean_bad_trajectories.py \
#   --subject-id <ID> \
#   --region-dir <regional coordinate directory> \
#   --profile-dir <profile directory> \
#   --metrics md fa ad rd

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


DEPTH_SAMPLES = 21


def load_profile(profile_path):
    profile_path = Path(profile_path)

    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Profile file not found: {profile_path}"
        )

    if profile_path.stat().st_size == 0:
        return np.empty(
            (0, DEPTH_SAMPLES),
            dtype=float,
        )

    values = np.loadtxt(
        profile_path,
        delimiter=",",
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim == 1:
        values = values.reshape(
            1,
            -1,
        )

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2D profile data. Got {values.shape}."
        )

    if values.shape[1] != DEPTH_SAMPLES:
        raise ValueError(
            f"Expected {DEPTH_SAMPLES} depth samples. "
            f"Got {values.shape} in {profile_path}."
        )

    return values


def get_region_key(
    profile_path,
    subject_id,
    metric_name,
):
    filename = profile_path.stem

    prefix = f"{subject_id}_"
    suffix = f"_{metric_name}"

    if not filename.startswith(prefix):
        raise ValueError(
            f"Unexpected filename: {profile_path.name}"
        )

    if not filename.endswith(suffix):
        raise ValueError(
            f"Unexpected filename: {profile_path.name}"
        )

    return filename[
        len(prefix):
        -len(suffix)
    ]


def find_profile_files(
    subject_id,
    profile_dir,
    metric_name,
):
    profile_dir = Path(profile_dir)

    return sorted(
        profile_dir.glob(
            f"{subject_id}_*_{metric_name}.csv"
        )
    )


def find_bad_trajectories(values):
    invalid = (
        ~np.isfinite(values)
        | (values == 0)
    )

    bad_indices = np.flatnonzero(
        invalid.any(axis=1)
    )

    return set(
        int(index)
        for index in bad_indices
    )


def collect_bad_trajectories(
    subject_id,
    profile_dir,
    metrics,
):
    bad_by_region = {}

    for metric_name in metrics:
        profile_files = find_profile_files(
            subject_id,
            profile_dir,
            metric_name,
        )

        if not profile_files:
            raise RuntimeError(
                f"No profile files found for metric "
                f"{metric_name} in {profile_dir}"
            )

        print()
        print(
            f"Scanning {metric_name}: "
            f"{len(profile_files)} regions"
        )

        for profile_path in profile_files:
            region_key = get_region_key(
                profile_path,
                subject_id,
                metric_name,
            )

            values = load_profile(
                profile_path
            )

            bad_indices = find_bad_trajectories(
                values
            )

            if not bad_indices:
                continue

            bad_by_region.setdefault(
                region_key,
                set(),
            ).update(
                bad_indices
            )

            print(
                f"{region_key}: "
                f"{len(bad_indices)} bad trajectories"
            )

    return bad_by_region


def validate_profile_files(
    subject_id,
    profile_dir,
    metrics,
    bad_by_region,
):
    for region_key, bad_indices in bad_by_region.items():
        row_counts = []

        for metric_name in metrics:
            profile_path = (
                Path(profile_dir)
                / (
                    f"{subject_id}_"
                    f"{region_key}_"
                    f"{metric_name}.csv"
                )
            )

            values = load_profile(
                profile_path
            )

            row_counts.append(
                values.shape[0]
            )

            invalid_indices = [
                index
                for index in bad_indices
                if (
                    index < 0
                    or index >= values.shape[0]
                )
            ]

            if invalid_indices:
                raise IndexError(
                    f"Bad trajectory indices exceed "
                    f"row count in {profile_path}: "
                    f"{invalid_indices[:10]}"
                )

        if len(set(row_counts)) != 1:
            raise ValueError(
                f"Metric profile row counts do not "
                f"match for {region_key}: "
                f"{row_counts}"
            )


def validate_region_coordinates(
    region_dir,
    bad_by_region,
):
    region_dir = Path(region_dir)

    for region_key, bad_indices in bad_by_region.items():
        hemisphere = region_key.split(
            "_",
            1,
        )[0]

        mat_path = (
            region_dir
            / f"{region_key}.mat"
        )

        if not mat_path.is_file():
            raise FileNotFoundError(
                f"Regional coordinate file "
                f"not found: {mat_path}"
            )

        data = loadmat(
            str(mat_path)
        )

        variable_name = (
            f"{hemisphere}_trajectory_dwi"
        )

        if variable_name not in data:
            raise KeyError(
                f"{variable_name} not found "
                f"in {mat_path}"
            )

        coordinates = np.asarray(
            data[variable_name],
            dtype=float,
        )

        if (
            coordinates.ndim != 2
            or coordinates.shape[0] not in [3, 4]
        ):
            raise ValueError(
                f"Unexpected coordinate shape "
                f"in {mat_path}: "
                f"{coordinates.shape}"
            )

        if (
            coordinates.shape[1]
            % DEPTH_SAMPLES
            != 0
        ):
            raise ValueError(
                f"Coordinate count is not divisible "
                f"by {DEPTH_SAMPLES}: "
                f"{coordinates.shape}"
            )

        n_trajectories = (
            coordinates.shape[1]
            // DEPTH_SAMPLES
        )

        invalid_indices = [
            index
            for index in bad_indices
            if (
                index < 0
                or index >= n_trajectories
            )
        ]

        if invalid_indices:
            raise IndexError(
                f"Bad trajectory indices exceed "
                f"coordinate trajectory count "
                f"for {region_key}: "
                f"{invalid_indices[:10]}"
            )


def clean_profile(
    profile_path,
    bad_indices,
):
    values = load_profile(
        profile_path
    )

    n_before = values.shape[0]

    keep = np.ones(
        n_before,
        dtype=bool,
    )

    if bad_indices:
        indices = np.asarray(
            sorted(bad_indices),
            dtype=int,
        )

        keep[indices] = False

    cleaned = values[
        keep,
        :
    ]

    np.savetxt(
        profile_path,
        cleaned,
        delimiter=",",
    )

    return (
        n_before,
        cleaned.shape[0],
    )


def clean_region_coordinates(
    region_dir,
    region_key,
    bad_indices,
):
    region_dir = Path(region_dir)

    hemisphere = region_key.split(
        "_",
        1,
    )[0]

    mat_path = (
        region_dir
        / f"{region_key}.mat"
    )

    csv_path = (
        region_dir
        / f"{region_key}.csv"
    )

    data = loadmat(
        str(mat_path)
    )

    variable_name = (
        f"{hemisphere}_trajectory_dwi"
    )

    coordinates = np.asarray(
        data[variable_name],
        dtype=float,
    )

    n_trajectories = (
        coordinates.shape[1]
        // DEPTH_SAMPLES
    )

    keep_trajectories = np.ones(
        n_trajectories,
        dtype=bool,
    )

    if bad_indices:
        indices = np.asarray(
            sorted(bad_indices),
            dtype=int,
        )

        keep_trajectories[
            indices
        ] = False

    # Each trajectory contains 21 consecutive points.
    keep_points = np.repeat(
        keep_trajectories,
        DEPTH_SAMPLES,
    )

    cleaned = coordinates[
        :,
        keep_points,
    ]

    savemat(
        str(mat_path),
        {
            variable_name: cleaned,
        },
    )

    np.savetxt(
        str(csv_path),
        cleaned,
        delimiter=",",
    )

    return (
        n_trajectories,
        cleaned.shape[1]
        // DEPTH_SAMPLES,
    )


def save_bad_trajectory_list(
    subject_id,
    profile_dir,
    metrics,
    bad_by_region,
):
    output_path = (
        Path(profile_dir)
        / "bad_trajectories.json"
    )

    data = {
        "subject_id": subject_id,
        "metrics": list(metrics),
        "depth_samples": DEPTH_SAMPLES,
        "indexing": (
            "0-based trajectory row indices "
            "at cleaning time"
        ),
        "bad_trajectories": {
            region: sorted(indices)
            for region, indices
            in sorted(
                bad_by_region.items()
            )
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )

    print()
    print(
        f"Bad trajectory list saved: "
        f"{output_path}"
    )


def save_bad_trajectory_mask(subject_id, profile_dir, region_dir, bad_by_region):
    """Save complete binary masks before profile/coordinate cleaning."""
    data = {
        "subject_id": subject_id,
        "depth_samples": DEPTH_SAMPLES,
        "indexing": "0-based trajectory row indices before cleaning",
        "regions": {},
    }
    for region_key, bad_indices in sorted(bad_by_region.items()):
        hemisphere = region_key.split("_", 1)[0]
        mat_path = Path(region_dir) / f"{region_key}.mat"
        mat = loadmat(str(mat_path))
        coordinates = np.asarray(mat[f"{hemisphere}_trajectory_dwi"], dtype=float)
        n_trajectories = coordinates.shape[1] // DEPTH_SAMPLES
        mask = np.zeros(n_trajectories, dtype=np.uint8)
        mask[sorted(bad_indices)] = 1
        data["regions"][region_key] = {
            "n_trajectories_before": int(n_trajectories),
            "bad_count": int(mask.sum()),
            "keep_count": int((mask == 0).sum()),
            "bad_mask": mask.tolist(),
            "bad_indices": sorted(int(i) for i in bad_indices),
        }
    output_path = Path(profile_dir) / "bad_trajectory_mask.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    print(f"Bad trajectory mask saved: {output_path}")


def clean_all_profiles(
    subject_id,
    profile_dir,
    metrics,
    bad_by_region,
):
    for metric_name in metrics:
        print()
        print(
            f"Cleaning metric: "
            f"{metric_name}"
        )

        profile_files = find_profile_files(
            subject_id,
            profile_dir,
            metric_name,
        )

        for profile_path in profile_files:
            region_key = get_region_key(
                profile_path,
                subject_id,
                metric_name,
            )

            bad_indices = bad_by_region.get(
                region_key
            )

            if not bad_indices:
                continue

            before, after = clean_profile(
                profile_path,
                bad_indices,
            )

            print(
                f"{region_key}: "
                f"{before} -> {after}"
            )


def clean_all_coordinates(
    region_dir,
    bad_by_region,
):
    print()
    print(
        "Cleaning regional coordinates..."
    )

    for region_key, bad_indices in sorted(
        bad_by_region.items()
    ):
        before, after = (
            clean_region_coordinates(
                region_dir,
                region_key,
                bad_indices,
            )
        )

        print(
            f"{region_key}: "
            f"{before} -> {after}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove invalid trajectories "
            "consistently across diffusion metrics."
        )
    )

    parser.add_argument(
        "--subject-id",
        required=True,
    )

    parser.add_argument(
        "--region-dir",
        required=True,
        help=(
            "Directory containing "
            "lh_<region>.mat / csv and "
            "rh_<region>.mat / csv."
        ),
    )

    parser.add_argument(
        "--profile-dir",
        required=True,
        help=(
            "Directory containing sampled "
            "MD/FA/AD/RD profile CSV files."
        ),
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "md",
            "fa",
            "ad",
            "rd",
        ],
    )

    args = parser.parse_args()

    print(
        "Cross-contrast trajectory cleaning"
    )

    bad_by_region = collect_bad_trajectories(
        subject_id=args.subject_id,
        profile_dir=args.profile_dir,
        metrics=args.metrics,
    )

    save_bad_trajectory_list(
        subject_id=args.subject_id,
        profile_dir=args.profile_dir,
        metrics=args.metrics,
        bad_by_region=bad_by_region,
    )

    save_bad_trajectory_mask(
        subject_id=args.subject_id,
        profile_dir=args.profile_dir,
        region_dir=args.region_dir,
        bad_by_region=bad_by_region,
    )

    if not bad_by_region:
        print()
        print(
            "No zero or non-finite "
            "trajectories found."
        )
        return

    # Validate everything before modifying files.
    validate_profile_files(
        subject_id=args.subject_id,
        profile_dir=args.profile_dir,
        metrics=args.metrics,
        bad_by_region=bad_by_region,
    )

    validate_region_coordinates(
        region_dir=args.region_dir,
        bad_by_region=bad_by_region,
    )

    clean_all_profiles(
        subject_id=args.subject_id,
        profile_dir=args.profile_dir,
        metrics=args.metrics,
        bad_by_region=bad_by_region,
    )

    clean_all_coordinates(
        region_dir=args.region_dir,
        bad_by_region=bad_by_region,
    )

    print()
    print(
        "Cross-contrast trajectory "
        "cleaning complete."
    )


if __name__ == "__main__":
    main()
