#!/usr/bin/env python
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat


def load_transform_matrix(transform_path):
    transform_path = Path(transform_path)

    if not transform_path.is_file():
        raise FileNotFoundError(
            f"Transform file not found: {transform_path}"
        )

    rows = []

    with open(transform_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 4:
                continue

            try:
                row = [float(value) for value in values]
            except ValueError:
                continue

            rows.append(row)

    if len(rows) < 4:
        raise ValueError(
            f"Could not find a 4x4 transform matrix in {transform_path}"
        )

    transform = np.asarray(
        rows[-4:],
        dtype=float,
    )

    if transform.shape != (4, 4):
        raise ValueError(
            f"Transform must have shape (4, 4), got {transform.shape}"
        )

    return transform


def vox2ras_tkreg(shape, voxel_size):
    nx, ny, nz = shape
    dx, dy, dz = voxel_size

    matrix = np.array(
        [
            [-dx, 0.0, 0.0, dx * nx / 2.0],
            [0.0, 0.0, dz, -dz * nz / 2.0],
            [0.0, -dy, 0.0, dy * ny / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return matrix


def load_dwi_geometry(dwi_path):
    dwi_path = Path(dwi_path)

    if not dwi_path.is_file():
        raise FileNotFoundError(
            f"DWI image not found: {dwi_path}"
        )

    image = nib.load(str(dwi_path))

    shape = image.shape[:3]
    voxel_size = image.header.get_zooms()[:3]

    t_mov = vox2ras_tkreg(
        shape,
        voxel_size,
    )

    return shape, voxel_size, t_mov


def load_trajectory_points(trajectory_path, hemisphere):
    trajectory_path = Path(trajectory_path)

    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Trajectory file not found: {trajectory_path}"
        )

    data = loadmat(str(trajectory_path))

    variable_name = f"{hemisphere}_trajectory_points"

    if variable_name not in data:
        raise KeyError(
            f"{variable_name} not found in {trajectory_path}"
        )

    trajectory_points = np.asarray(
        data[variable_name],
        dtype=float,
    )

    if (
        trajectory_points.ndim != 2
        or trajectory_points.shape[1] != 3
    ):
        raise ValueError(
            "Trajectory points must have shape (N, 3). "
            f"Got {trajectory_points.shape}."
        )

    return trajectory_points


def transform_points_to_dwi(
    trajectory_points,
    transform,
    t_mov,
):
    n_points = trajectory_points.shape[0]

    homogeneous = np.ones(
        (4, n_points),
        dtype=float,
    )

    homogeneous[0:3, :] = trajectory_points.T

    dwi_ras = transform @ homogeneous

    dwi_voxel = np.linalg.inv(t_mov) @ dwi_ras

    return dwi_voxel


def process_hemisphere(
    subject_id,
    hemisphere,
    trajectory_dir,
    output_dir,
    transform,
    t_mov,
):
    trajectory_path = (
        Path(trajectory_dir)
        / f"{subject_id}_trajectory_{hemisphere}.mat"
    )

    trajectory_points = load_trajectory_points(
        trajectory_path,
        hemisphere,
    )

    trajectory_dwi = transform_points_to_dwi(
        trajectory_points,
        transform,
        t_mov,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{subject_id}_trajectory_{hemisphere}_dwi.mat"
    )

    variable_name = (
        f"{hemisphere}_trajectory_dwi"
    )

    savemat(
        str(output_path),
        {
            variable_name: trajectory_dwi,
        },
    )

    print()
    print(f"Hemisphere: {hemisphere}")
    print(
        f"Input trajectory shape: "
        f"{trajectory_points.shape}"
    )
    print(
        f"DWI trajectory shape: "
        f"{trajectory_dwi.shape}"
    )

    print(
        "Voxel coordinate ranges:"
    )

    print(
        f"i: {trajectory_dwi[0].min():.2f} "
        f"to {trajectory_dwi[0].max():.2f}"
    )

    print(
        f"j: {trajectory_dwi[1].min():.2f} "
        f"to {trajectory_dwi[1].max():.2f}"
    )

    print(
        f"k: {trajectory_dwi[2].min():.2f} "
        f"to {trajectory_dwi[2].max():.2f}"
    )

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transform cortical trajectory points "
            "from FreeSurfer/T1 space to DWI voxel coordinates."
        )
    )

    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject identifier.",
    )

    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory containing trajectory MAT files.",
    )

    parser.add_argument(
        "--transform",
        required=True,
        help="T1-to-DWI transform file.",
    )

    parser.add_argument(
        "--dwi",
        required=True,
        help="Reference DWI NIfTI image.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for transformed trajectories.",
    )

    args = parser.parse_args()

    transform = load_transform_matrix(
        args.transform
    )

    shape, voxel_size, t_mov = load_dwi_geometry(
        args.dwi
    )

    print(f"Subject: {args.subject_id}")
    print(f"DWI shape: {shape}")
    print(f"DWI voxel size: {voxel_size}")

    print()
    print("T1-to-DWI transform:")
    print(transform)

    print()
    print("T_mov:")
    print(t_mov)

    for hemisphere in ["lh", "rh"]:
        process_hemisphere(
            subject_id=args.subject_id,
            hemisphere=hemisphere,
            trajectory_dir=args.trajectory_dir,
            output_dir=args.output_dir,
            transform=transform,
            t_mov=t_mov,
        )

    print()
    print("Trajectory transformation complete.")


if __name__ == "__main__":
    main()