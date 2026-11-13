#!/usr/bin/env python3
"""Sample a Scalar Image Along Fsaverage-Indexed Columns

Description
-----------
Sample any 3D scalar image along native-T1 cortical columns after transforming
their coordinates into the registered DWI voxel grid.

Methods
-------
1. Load hemisphere column coordinates and mapping metadata.
2. Verify that the metric and DWI-reference grids and affines match.
3. Transform native-T1 column points to DWI voxel coordinates.
4. Perform trilinear interpolation at every in-bounds depth point.
5. Save profiles, coordinates, validity masks, and column metadata.

Inputs
------
- Subject column NPZ files and arbitrary 3D scalar metric image
- Metric name, DWI-to-T1 registration, and DWI reference image
- Output directory and optional hemisphere selection

Outputs
-------
- One vertex-by-depth metric profile NPZ per hemisphere
- DWI voxel coordinates and sampling-validity masks

Usage
-----
python 04_profile_sampling/sample_template_columns.py     --subject-id D0007     --columns-dir /path/to/columns     --metric-image /path/to/metric.nii.gz     --metric-name md     --registration /path/to/D0007_DWI2T1.dat     --dwi /path/to/D0007_dwi_masked.nii.gz     --output-dir /path/to/profiles_before_clean

Notes
-----
- The metric is not restricted to MD.
- Metric and reference affine mismatches are rejected before sampling.
- Output rows retain fsaverage vertex IDs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from transform_t1_columns_to_dwi import (
    load_dwi_geometry,
    load_transform_matrix,
    transform_points_to_dwi,
)


SAMPLING_ORDER = 1
SAMPLING_MODE = "nearest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--columns-dir", type=Path, required=True)
    parser.add_argument("--metric-image", type=Path, required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    return parser.parse_args()


def load_metric(path: Path, dwi_path: Path) -> np.ndarray:
    """Load a scalar image and require the registration-reference grid."""

    metric_image = nib.load(str(path))
    dwi_image = nib.load(str(dwi_path))
    metric = metric_image.get_fdata(dtype=np.float64)
    if metric.ndim == 4:
        if metric.shape[3] != 1:
            raise ValueError(f"Expected a 3D metric image, got {metric.shape}")
        metric = metric[..., 0]
    if metric.ndim != 3:
        raise ValueError(f"Expected a 3D metric image, got {metric.shape}")
    if metric.shape != dwi_image.shape[:3]:
        raise ValueError(
            "Metric and DWI-reference dimensions differ: "
            f"{metric.shape} != {dwi_image.shape[:3]}"
        )
    if not np.allclose(metric_image.affine, dwi_image.affine, atol=1e-4):
        raise ValueError(
            "Metric and DWI-reference affines differ; resample the metric "
            "to the registration-reference grid before column sampling."
        )
    return metric


def sample_columns(
    column_points: np.ndarray,
    mapping_valid: np.ndarray,
    metric: np.ndarray,
    transform: np.ndarray,
    t_mov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_columns, n_depths, _ = column_points.shape
    dwi_points = transform_points_to_dwi(
        column_points.reshape(-1, 3),
        transform,
        t_mov,
    ).reshape(n_columns, n_depths, 3)
    dwi_points -= 1.0
    shape = np.asarray(metric.shape, dtype=float)
    in_bounds = mapping_valid[:, None] & np.all(
        (dwi_points >= 0.0) & (dwi_points <= shape[None, None, :] - 1.0),
        axis=2,
    )
    profiles = np.full((n_columns, n_depths), np.nan, dtype=float)
    for depth_index in range(n_depths):
        valid = mapping_valid & in_bounds[:, depth_index]
        if not valid.any():
            continue
        profiles[valid, depth_index] = map_coordinates(
            metric,
            dwi_points[valid, depth_index, :].T,
            order=SAMPLING_ORDER,
            mode=SAMPLING_MODE,
        )
    sampled_finite = np.isfinite(profiles)
    sampling_valid = (
        mapping_valid
        & in_bounds.all(axis=1)
        & sampled_finite.all(axis=1)
    )
    return profiles, dwi_points, in_bounds, sampled_finite, sampling_valid


def process_hemisphere(args: argparse.Namespace, hemisphere: str) -> None:
    column_path = args.columns_dir / f"{args.subject_id}_{hemisphere}_columns.npz"
    if not column_path.is_file():
        raise FileNotFoundError(f"Column file not found: {column_path}")
    with np.load(column_path, allow_pickle=True) as data:
        column_data = {key: data[key] for key in data.files}

    metric = load_metric(args.metric_image, args.dwi)
    transform = load_transform_matrix(args.registration)
    _, t_mov = load_dwi_geometry(args.dwi)
    profiles, dwi_points, in_bounds, sampled_finite, sampling_valid = sample_columns(
        column_data["column_points"],
        column_data["mapping_valid"].astype(bool),
        metric,
        transform,
        t_mov,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{args.subject_id}_{hemisphere}_{args.metric_name}_profiles.npz"
    )
    output_data = {
        "subject_id": args.subject_id,
        "hemisphere": hemisphere,
        "metric_name": args.metric_name,
        "fsaverage_vertex_id": column_data["fsaverage_vertex_id"],
        "target_vertex_id": column_data["target_vertex_id"],
        "profiles": profiles,
        "dwi_voxel_coordinates": dwi_points,
        "coordinate_in_bounds_mask": in_bounds,
        "sampled_finite_mask": sampled_finite,
        "sampling_valid": sampling_valid,
        "mapping_valid": column_data["mapping_valid"],
        "depth_fractions": column_data["depth_fractions"],
        "depth_direction": column_data["depth_direction"],
        "template": column_data["template"],
        "column_file": str(column_path),
        "metric_image": str(args.metric_image),
        "registration": str(args.registration),
        "sampling_order": SAMPLING_ORDER,
        "sampling_mode": SAMPLING_MODE,
        "coordinate_indexing": (
            "0-based after subtracting 1 from transform output"
        ),
    }
    for key in (
        "template_label_id",
        "template_label_name",
        "fsaverage_label",
        "template_annotation_path",
        "in_analysis_mask",
        "column_length",
        "cortical_thickness",
        "is_zero_length",
        "column_valid",
        "native_triangle_id",
        "native_triangle_vertex_ids",
        "barycentric_weights",
        "white_coordinates_t1",
        "pial_coordinates_t1",
    ):
        if key in column_data:
            output_data[key] = column_data[key]
    np.savez_compressed(output_path, **output_data)
    logging.info(
        "[%s] %s: %d columns x %d depths; complete finite %.2f%%; in-bounds %.2f%%",
        hemisphere,
        args.metric_name,
        profiles.shape[0],
        profiles.shape[1],
        sampled_finite.all(axis=1).mean() * 100,
        in_bounds.all(axis=1).mean() * 100,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for hemisphere in args.hemispheres:
        process_hemisphere(args, hemisphere)


if __name__ == "__main__":
    main()
