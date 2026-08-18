#!/usr/bin/env python

import argparse
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.io import savemat


POINTS_NUM = 21


def load_surface(surface_path):
    surface_path = Path(surface_path)

    if not surface_path.is_file():
        raise FileNotFoundError(
            f"Surface file not found: {surface_path}"
        )

    vertices, faces = fsio.read_geometry(str(surface_path))

    print(
        f"Loaded {surface_path.name}: "
        f"{vertices.shape[0]} vertices, "
        f"{faces.shape[0]} faces"
    )

    return vertices


def build_pair(white_vertices, pial_vertices):
    if white_vertices.shape != pial_vertices.shape:
        raise ValueError(
            "White and pial surfaces must have the same shape. "
            f"Got {white_vertices.shape} and {pial_vertices.shape}."
        )

    pair = np.hstack(
        [
            white_vertices,
            pial_vertices,
        ]
    )

    return pair


def generate_trajectory_points(pair, points_num=POINTS_NUM):
    pair = np.asarray(pair, dtype=float)

    if pair.ndim != 2 or pair.shape[1] != 6:
        raise ValueError(
            f"Pair must have shape (N, 6), got {pair.shape}."
        )

    if points_num < 2:
        raise ValueError(
            "points_num must be at least 2."
        )

    white = pair[:, 0:3]
    pial = pair[:, 3:6]

    depth = np.linspace(
        0.0,
        1.0,
        points_num,
        dtype=float,
    )

    points = (
        white[:, None, :]
        * (1.0 - depth)[None, :, None]
        + pial[:, None, :]
        * depth[None, :, None]
    )

    trajectory_points = points.reshape(
        pair.shape[0] * points_num,
        3,
    )

    return trajectory_points


def process_hemisphere(
    subject_id,
    subject_dir,
    output_dir,
    hemisphere,
    points_num,
):
    subject_dir = Path(subject_dir)
    output_dir = Path(output_dir)

    surf_dir = subject_dir / "surf"

    white_path = surf_dir / f"{hemisphere}.white"
    pial_path = surf_dir / f"{hemisphere}.pial"

    print()
    print(f"Processing hemisphere: {hemisphere}")

    white_vertices = load_surface(
        white_path
    )

    pial_vertices = load_surface(
        pial_path
    )

    pair = build_pair(
        white_vertices,
        pial_vertices,
    )

    trajectory_points = generate_trajectory_points(
        pair,
        points_num=points_num,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_output = (
        output_dir
        / f"{subject_id}_pair_{hemisphere}.mat"
    )

    trajectory_output = (
        output_dir
        / f"{subject_id}_trajectory_{hemisphere}.mat"
    )

    pair_variable = f"{hemisphere}_pair"

    trajectory_variable = (
        f"{hemisphere}_trajectory_points"
    )

    savemat(
        str(pair_output),
        {
            pair_variable: pair,
        },
    )

    savemat(
        str(trajectory_output),
        {
            trajectory_variable: trajectory_points,
        },
    )

    print(f"Pair shape: {pair.shape}")
    print(
        f"Trajectory points shape: "
        f"{trajectory_points.shape}"
    )
    print(
        f"Points per trajectory: {points_num}"
    )

    print(f"Saved pair: {pair_output}")
    print(
        f"Saved trajectory: "
        f"{trajectory_output}"
    )


def build_trajectories(
    subject_id,
    subject_dir,
    output_dir,
    points_num=POINTS_NUM,
):
    subject_dir = Path(subject_dir)

    if not subject_dir.is_dir():
        raise NotADirectoryError(
            f"FreeSurfer subject directory not found: "
            f"{subject_dir}"
        )

    print(f"Subject: {subject_id}")
    print(
        f"FreeSurfer subject directory: "
        f"{subject_dir}"
    )
    print(f"Output directory: {output_dir}")
    print(
        f"Points per trajectory: "
        f"{points_num}"
    )

    for hemisphere in ["lh", "rh"]:
        process_hemisphere(
            subject_id=subject_id,
            subject_dir=subject_dir,
            output_dir=output_dir,
            hemisphere=hemisphere,
            points_num=points_num,
        )

    print()
    print("Trajectory construction complete.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build cortical trajectories from "
            "FreeSurfer white and pial surfaces."
        )
    )

    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject identifier.",
    )

    parser.add_argument(
        "--subject-dir",
        required=True,
        help=(
            "FreeSurfer subject directory "
            "containing the surf directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for trajectory outputs."
        ),
    )

    parser.add_argument(
        "--points",
        type=int,
        default=POINTS_NUM,
        help=(
            "Number of points per trajectory, "
            "including white and pial endpoints. "
            "Default: 21."
        ),
    )

    args = parser.parse_args()

    build_trajectories(
        subject_id=args.subject_id,
        subject_dir=args.subject_dir,
        output_dir=args.output_dir,
        points_num=args.points,
    )


if __name__ == "__main__":
    main()