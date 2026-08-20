#!/usr/bin/env python

# Purpose:
# Sample diffusion metric values along cortical trajectories.
#
# Usage:
# python profile_sampling/sample_profiles.py \
#   --subject-id <ID> \
#   --region-dir <region coordinate dir> \
#   --metric-image <MD/FA/AD/RD nifti> \
#   --metric-name <md/fa/ad/rd> \
#   --output-dir <output dir>
#
# Inputs:
# Region-wise DWI voxel coordinates and one diffusion metric image.
#
# Outputs:
# One CSV per hemisphere and cortical region.
# Rows are trajectories and columns are 21 cortical depth samples.

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates


DEPTH_SAMPLES = 21


def load_metric_image(metric_path):
    metric_path = Path(metric_path)

    if not metric_path.is_file():
        raise FileNotFoundError(
            f"Metric image not found: {metric_path}"
        )

    image = nib.load(str(metric_path))
    volume = image.get_fdata()

    if volume.ndim == 4:
        if volume.shape[3] != 1:
            raise ValueError(
                f"Expected a 3D metric image. Got {volume.shape}."
            )
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(
            f"Metric image must be 3D. Got {volume.shape}."
        )

    print(f"Metric image: {metric_path}")
    print(f"Shape: {volume.shape}")

    return volume


def load_region_coordinates(mat_path, hemisphere):
    mat_path = Path(mat_path)

    data = loadmat(str(mat_path))

    key_candidates = [
        f"{hemisphere}_trajectory_dwi",
        f"{hemisphere}_cp_dwi",
        "cp_dwi",
    ]

    coordinates = None

    for key in key_candidates:
        if key in data:
            coordinates = np.asarray(
                data[key],
                dtype=float,
            )
            break

    if coordinates is None:
        raise KeyError(
            f"No trajectory coordinate variable found in {mat_path}. "
            f"Tried: {key_candidates}"
        )

    if coordinates.ndim != 2:
        raise ValueError(
            f"Coordinates must be 2D. Got {coordinates.shape}."
        )

    if coordinates.shape[0] not in [3, 4]:
        raise ValueError(
            "Coordinates must have 3 or 4 rows. "
            f"Got {coordinates.shape}."
        )

    return coordinates


def sample_trajectories(
    volume,
    coordinates,
    depth_samples=DEPTH_SAMPLES,
):
    if coordinates.shape[0] == 4:
        coords = coordinates[:3, :]
    else:
        coords = coordinates

    n_points = coords.shape[1]

    if n_points % depth_samples != 0:
        raise ValueError(
            f"{n_points} coordinates cannot be divided "
            f"into trajectories with {depth_samples} points."
        )

    n_trajectories = (
        n_points // depth_samples
    )

    values = np.zeros(
        (n_trajectories, depth_samples),
        dtype=float,
    )

    for depth in range(depth_samples):
        point_indices = (
            depth
            + np.arange(n_trajectories)
            * depth_samples
        )

        sample_coords = coords[
            :,
            point_indices,
        ].astype(float)

        # The original Brain_column_builder stores
        # DWI voxel coordinates using 1-based indexing.
        # Convert to Python 0-based array indexing.
        sample_coords -= 1.0

        sampled_values = map_coordinates(
            volume,
            sample_coords,
            order=1,
            mode="nearest",
        )

        values[:, depth] = sampled_values

    return values


def process_region_file(
    mat_path,
    hemisphere,
    metric_volume,
    subject_id,
    metric_name,
    output_dir,
):
    region_name = mat_path.stem[
        len(hemisphere) + 1:
    ]

    coordinates = load_region_coordinates(
        mat_path,
        hemisphere,
    )

    values = sample_trajectories(
        metric_volume,
        coordinates,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{subject_id}_"
            f"{hemisphere}_"
            f"{region_name}_"
            f"{metric_name}.csv"
        )
    )

    np.savetxt(
        output_path,
        values,
        delimiter=",",
    )

    print(
        f"{hemisphere}_{region_name}: "
        f"{values.shape[0]} trajectories × "
        f"{values.shape[1]} depths"
    )

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample diffusion metric values "
            "along cortical trajectories."
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
            "lh_<region>.mat and rh_<region>.mat."
        ),
    )

    parser.add_argument(
        "--metric-image",
        required=True,
        help=(
            "3D diffusion metric image such as "
            "MD, FA, AD, or RD."
        ),
    )

    parser.add_argument(
        "--metric-name",
        required=True,
        help="Name used in output filenames.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    metric_volume = load_metric_image(
        args.metric_image
    )

    region_dir = Path(
        args.region_dir
    )

    if not region_dir.is_dir():
        raise FileNotFoundError(
            f"Region directory not found: {region_dir}"
        )

    for hemisphere in [
        "lh",
        "rh",
    ]:
        region_files = sorted(
            region_dir.glob(
                f"{hemisphere}_*.mat"
            )
        )

        print()
        print(
            f"{hemisphere.upper()} regions: "
            f"{len(region_files)}"
        )

        for mat_path in region_files:
            process_region_file(
                mat_path=mat_path,
                hemisphere=hemisphere,
                metric_volume=metric_volume,
                subject_id=args.subject_id,
                metric_name=args.metric_name,
                output_dir=args.output_dir,
            )

    print()
    print("Profile sampling complete.")


if __name__ == "__main__":
    main()