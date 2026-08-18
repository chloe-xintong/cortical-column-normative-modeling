#!/usr/bin/env python

"""
Check cortical trajectory outputs generated from FreeSurfer surfaces.

This script verifies that the generated pair and trajectory files are
internally consistent for both hemispheres.

For each hemisphere, it checks:
- pair data have shape (N, 6)
- trajectory points have shape (N * points_num, 3)
- trajectory points can be reshaped to (N, points_num, 3)
- the first trajectory point matches the white endpoint
- the last trajectory point matches the pial endpoint
- the midpoint matches the expected linear interpolation
- all trajectory points follow the expected straight-line interpolation

These checks confirm that each cortical trajectory was constructed correctly
from the corresponding white and pial surface vertices.

Usage:
    python qc/check_trajectories.py \
        --trajectory-dir <trajectory_output_dir> \
        --subject-id <subject_id>

Optional:
    --points <number_of_points>

Default:
    21 points per trajectory
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


POINTS_NUM = 21
TOLERANCE = 1e-10


def check_file(path):
    if not path.is_file():
        print(f"  [FAIL] Missing file: {path}")
        return False

    if path.stat().st_size == 0:
        print(f"  [FAIL] Empty file: {path}")
        return False

    return True


def check_hemisphere(
    trajectory_dir,
    subject_id,
    hemisphere,
    points_num,
):
    print()
    print(f"Checking {hemisphere.upper()} hemisphere")

    pair_file = (
        trajectory_dir
        / f"{subject_id}_pair_{hemisphere}.mat"
    )

    trajectory_file = (
        trajectory_dir
        / f"{subject_id}_trajectory_{hemisphere}.mat"
    )

    files_ok = True

    for path in [pair_file, trajectory_file]:
        if not check_file(path):
            files_ok = False

    if not files_ok:
        print(f"{hemisphere.upper()}: FAIL")
        return False

    pair_variable = f"{hemisphere}_pair"
    trajectory_variable = (
        f"{hemisphere}_trajectory_points"
    )

    try:
        pair_data = loadmat(pair_file)
        trajectory_data = loadmat(trajectory_file)

    except Exception as exc:
        print(f"  [FAIL] Could not read MAT files: {exc}")
        print(f"{hemisphere.upper()}: FAIL")
        return False

    if pair_variable not in pair_data:
        print(
            f"  [FAIL] Variable '{pair_variable}' "
            f"not found in {pair_file.name}"
        )
        print(f"{hemisphere.upper()}: FAIL")
        return False

    if trajectory_variable not in trajectory_data:
        print(
            f"  [FAIL] Variable '{trajectory_variable}' "
            f"not found in {trajectory_file.name}"
        )
        print(f"{hemisphere.upper()}: FAIL")
        return False

    pair = np.asarray(
        pair_data[pair_variable],
        dtype=float,
    )

    trajectory = np.asarray(
        trajectory_data[trajectory_variable],
        dtype=float,
    )

    pair_shape_ok = (
        pair.ndim == 2
        and pair.shape[1] == 6
    )

    if not pair_shape_ok:
        print(
            f"  [FAIL] Pair must have shape (N, 6), "
            f"got {pair.shape}"
        )
        print(f"{hemisphere.upper()}: FAIL")
        return False

    n_trajectories = pair.shape[0]

    expected_trajectory_shape = (
        n_trajectories * points_num,
        3,
    )

    trajectory_shape_ok = (
        trajectory.shape
        == expected_trajectory_shape
    )

    print(f"  pair shape:              {pair.shape}")
    print(f"  trajectory shape:        {trajectory.shape}")
    print(f"  number of trajectories:  {n_trajectories}")
    print(f"  points per trajectory:   {points_num}")

    print(
        "  pair shape:",
        "PASS" if pair_shape_ok else "FAIL",
    )

    print(
        "  trajectory shape:",
        "PASS" if trajectory_shape_ok else "FAIL",
    )

    if not trajectory_shape_ok:
        print(
            f"  expected trajectory shape: "
            f"{expected_trajectory_shape}"
        )
        print(f"{hemisphere.upper()}: FAIL")
        return False

    points = trajectory.reshape(
        n_trajectories,
        points_num,
        3,
    )

    white = pair[:, 0:3]
    pial = pair[:, 3:6]

    white_error = np.max(
        np.abs(
            points[:, 0, :]
            - white
        )
    )

    pial_error = np.max(
        np.abs(
            points[:, -1, :]
            - pial
        )
    )

    depth = np.linspace(
        0.0,
        1.0,
        points_num,
        dtype=float,
    )

    expected_points = (
        white[:, None, :]
        * (1.0 - depth)[None, :, None]
        + pial[:, None, :]
        * depth[None, :, None]
    )

    interpolation_error = np.max(
        np.abs(
            points
            - expected_points
        )
    )

    midpoint_index = (
        points_num - 1
    ) // 2

    midpoint_depth = depth[midpoint_index]

    expected_midpoint = (
        white
        * (1.0 - midpoint_depth)
        + pial
        * midpoint_depth
    )

    midpoint_error = np.max(
        np.abs(
            points[:, midpoint_index, :]
            - expected_midpoint
        )
    )

    white_pass = (
        white_error <= TOLERANCE
    )

    pial_pass = (
        pial_error <= TOLERANCE
    )

    midpoint_pass = (
        midpoint_error <= TOLERANCE
    )

    interpolation_pass = (
        interpolation_error <= TOLERANCE
    )

    print(
        f"  max white endpoint error: "
        f"{white_error}"
    )

    print(
        f"  max pial endpoint error:  "
        f"{pial_error}"
    )

    print(
        f"  max midpoint error:       "
        f"{midpoint_error}"
    )

    print(
        f"  max interpolation error:  "
        f"{interpolation_error}"
    )

    print(
        "  white endpoint:",
        "PASS" if white_pass else "FAIL",
    )

    print(
        "  pial endpoint:",
        "PASS" if pial_pass else "FAIL",
    )

    print(
        "  midpoint:",
        "PASS" if midpoint_pass else "FAIL",
    )

    print(
        "  linear interpolation:",
        "PASS" if interpolation_pass else "FAIL",
    )

    hemisphere_pass = (
        pair_shape_ok
        and trajectory_shape_ok
        and white_pass
        and pial_pass
        and midpoint_pass
        and interpolation_pass
    )

    print(
        f"{hemisphere.upper()}:",
        "PASS" if hemisphere_pass else "FAIL",
    )

    return hemisphere_pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check cortical trajectory outputs."
        )
    )

    parser.add_argument(
        "--trajectory-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing pair and trajectory MAT files."
        ),
    )

    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject identifier.",
    )

    parser.add_argument(
        "--points",
        type=int,
        default=POINTS_NUM,
        help=(
            "Number of points per trajectory. "
            "Default: 21."
        ),
    )

    args = parser.parse_args()

    trajectory_dir = (
        args.trajectory_dir
        .expanduser()
        .resolve()
    )

    if not trajectory_dir.is_dir():
        print(
            f"Trajectory directory does not exist: "
            f"{trajectory_dir}"
        )
        sys.exit(1)

    if args.points < 2:
        print(
            "Number of points per trajectory "
            "must be at least 2."
        )
        sys.exit(1)

    print("Cortical trajectory QC")
    print(f"Subject: {args.subject_id}")
    print(
        f"Trajectory directory: "
        f"{trajectory_dir}"
    )
    print(
        f"Expected points per trajectory: "
        f"{args.points}"
    )

    lh_pass = check_hemisphere(
        trajectory_dir=trajectory_dir,
        subject_id=args.subject_id,
        hemisphere="lh",
        points_num=args.points,
    )

    rh_pass = check_hemisphere(
        trajectory_dir=trajectory_dir,
        subject_id=args.subject_id,
        hemisphere="rh",
        points_num=args.points,
    )

    overall_pass = (
        lh_pass
        and rh_pass
    )

    print()
    print("Overall result:")
    print(
        "Cortical trajectory QC:",
        "PASS" if overall_pass else "FAIL",
    )

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()