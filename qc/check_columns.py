#!/usr/bin/env python

"""
Check cortical column outputs generated from FreeSurfer surfaces.

This script verifies that the generated pair and column files are
internally consistent for both hemispheres.

For each hemisphere, it checks:
- pair data have shape (N, 6)
- column points have shape (N * points_num, 3)
- column points can be reshaped to (N, points_num, 3)
- the first column point matches the pial endpoint
- the last column point matches the white endpoint
- the midpoint matches the expected linear interpolation
- all column points follow the expected straight-line interpolation

These checks confirm that each cortical column was constructed correctly
from the corresponding white and pial surface vertices.

Usage:
    python qc/check_columns.py \
        --column-dir <column_output_dir> \
        --subject-id <subject_id>

Optional:
    --points <number_of_points>

Default:
    21 points per column
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
    column_dir,
    subject_id,
    hemisphere,
    points_num,
):
    print()
    print(f"Checking {hemisphere.upper()} hemisphere")

    pair_file = (
        column_dir
        / f"{subject_id}_pair_{hemisphere}.mat"
    )

    column_file = (
        column_dir
        / f"{subject_id}_column_{hemisphere}.mat"
    )

    files_ok = True

    for path in [pair_file, column_file]:
        if not check_file(path):
            files_ok = False

    if not files_ok:
        print(f"{hemisphere.upper()}: FAIL")
        return False

    pair_variable = f"{hemisphere}_pair"
    column_variable = (
        f"{hemisphere}_column_points"
    )

    try:
        pair_data = loadmat(pair_file)
        column_data = loadmat(column_file)

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

    if column_variable not in column_data:
        print(
            f"  [FAIL] Variable '{column_variable}' "
            f"not found in {column_file.name}"
        )
        print(f"{hemisphere.upper()}: FAIL")
        return False

    pair = np.asarray(
        pair_data[pair_variable],
        dtype=float,
    )

    column = np.asarray(
        column_data[column_variable],
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

    n_columns = pair.shape[0]

    expected_column_shape = (
        n_columns * points_num,
        3,
    )

    column_shape_ok = (
        column.shape
        == expected_column_shape
    )

    print(f"  pair shape:              {pair.shape}")
    print(f"  column shape:        {column.shape}")
    print(f"  number of columns:  {n_columns}")
    print(f"  points per column:   {points_num}")

    print(
        "  pair shape:",
        "PASS" if pair_shape_ok else "FAIL",
    )

    print(
        "  column shape:",
        "PASS" if column_shape_ok else "FAIL",
    )

    if not column_shape_ok:
        print(
            f"  expected column shape: "
            f"{expected_column_shape}"
        )
        print(f"{hemisphere.upper()}: FAIL")
        return False

    points = column.reshape(
        n_columns,
        points_num,
        3,
    )

    white = pair[:, 0:3]
    pial = pair[:, 3:6]

    pial_error = np.max(
        np.abs(
            points[:, 0, :]
            - pial
        )
    )

    white_error = np.max(
        np.abs(
            points[:, -1, :]
            - white
        )
    )

    depth = np.linspace(
        0.0,
        1.0,
        points_num,
        dtype=float,
    )

    expected_points = (
        pial[:, None, :]
        * (1.0 - depth)[None, :, None]
        + white[:, None, :]
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
        pial
        * (1.0 - midpoint_depth)
        + white
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
        and column_shape_ok
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
            "Check cortical column outputs."
        )
    )

    parser.add_argument(
        "--column-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing pair and column MAT files."
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
            "Number of points per column. "
            "Default: 21."
        ),
    )

    args = parser.parse_args()

    column_dir = (
        args.column_dir
        .expanduser()
        .resolve()
    )

    if not column_dir.is_dir():
        print(
            f"Column directory does not exist: "
            f"{column_dir}"
        )
        sys.exit(1)

    if args.points < 2:
        print(
            "Number of points per column "
            "must be at least 2."
        )
        sys.exit(1)

    print("Cortical column QC")
    print(f"Subject: {args.subject_id}")
    print(
        f"Column directory: "
        f"{column_dir}"
    )
    print(
        f"Expected points per column: "
        f"{args.points}"
    )

    lh_pass = check_hemisphere(
        column_dir=column_dir,
        subject_id=args.subject_id,
        hemisphere="lh",
        points_num=args.points,
    )

    rh_pass = check_hemisphere(
        column_dir=column_dir,
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
        "Cortical column QC:",
        "PASS" if overall_pass else "FAIL",
    )

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
