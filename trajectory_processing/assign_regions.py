#!/usr/bin/env python
'''
Purpose:
Assign cortical trajectories to Desikan-Killiany regions using
FreeSurfer aparc annotations.

Usage:
python trajectory_processing/assign_regions.py \
   --subject-id <ID> \
   --subjects-dir <FreeSurfer subjects dir> \
   --trajectory-dir <trajectory dir> \
   --output-dir <output dir>

 Inputs:
 FreeSurfer lh/rh.aparc.annot files and DWI-space trajectory MAT files.

 Outputs:
 Region-wise trajectory coordinate MAT and CSV files.
'''
import argparse
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.io import loadmat, savemat


DEPTH_SAMPLES = 21


def load_annotation(subjects_dir, subject_id, hemisphere):
    annot_path = (
        Path(subjects_dir)
        / subject_id
        / "label"
        / f"{hemisphere}.aparc.annot"
    )

    if not annot_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annot_path}"
        )

    labels, ctab, names = fsio.read_annot(
        str(annot_path)
    )

    region_names = [
        name.decode("utf-8")
        if isinstance(name, bytes)
        else str(name)
        for name in names
    ]

    print()
    print(f"Hemisphere: {hemisphere}")
    print(f"Annotation: {annot_path}")
    print(f"Vertices: {labels.shape[0]}")
    print(f"Regions in annotation: {len(region_names)}")

    return labels, ctab, region_names


def load_dwi_trajectories(
    trajectory_dir,
    subject_id,
    hemisphere,
):
    trajectory_path = (
        Path(trajectory_dir)
        / f"{subject_id}_trajectory_{hemisphere}_dwi.mat"
    )

    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Trajectory file not found: {trajectory_path}"
        )

    data = loadmat(
        str(trajectory_path)
    )

    variable_name = (
        f"{hemisphere}_trajectory_dwi"
    )

    if variable_name not in data:
        raise KeyError(
            f"{variable_name} not found in {trajectory_path}"
        )

    trajectory_dwi = np.asarray(
        data[variable_name],
        dtype=float,
    )

    if trajectory_dwi.ndim != 2:
        raise ValueError(
            "Trajectory array must be 2-D. "
            f"Got {trajectory_dwi.shape}."
        )

    if trajectory_dwi.shape[0] not in [3, 4]:
        raise ValueError(
            "Trajectory array must have 3 or 4 rows. "
            f"Got {trajectory_dwi.shape}."
        )

    print(
        f"DWI trajectory shape: {trajectory_dwi.shape}"
    )

    return trajectory_dwi


def get_region_point_indices(
    vertex_indices,
    depth_samples=DEPTH_SAMPLES,
):
    vertex_indices = np.asarray(
        vertex_indices,
        dtype=int,
    )

    if vertex_indices.size == 0:
        return np.array(
            [],
            dtype=int,
        )

    depth_indices = np.arange(
        depth_samples,
        dtype=int,
    )

    point_indices = (
        vertex_indices[:, None] * depth_samples
        + depth_indices[None, :]
    )

    return point_indices.ravel()


def save_region(
    subject_id,
    hemisphere,
    region_name,
    vertex_indices,
    trajectory_dwi,
    output_dir,
):
    point_indices = get_region_point_indices(
        vertex_indices
    )

    valid_indices = point_indices[
        point_indices < trajectory_dwi.shape[1]
    ]

    if valid_indices.size == 0:
        return None

    region_trajectory = trajectory_dwi[
        :,
        valid_indices,
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mat_path = (
        output_dir
        / f"{hemisphere}_{region_name}.mat"
    )

    csv_path = (
        output_dir
        / f"{hemisphere}_{region_name}.csv"
    )

    variable_name = (
        f"{hemisphere}_trajectory_dwi"
    )

    savemat(
        str(mat_path),
        {
            variable_name: region_trajectory,
        },
    )

    np.savetxt(
        str(csv_path),
        region_trajectory,
        delimiter=",",
    )

    print(
        f"{hemisphere}_{region_name}: "
        f"{vertex_indices.size} trajectories, "
        f"{valid_indices.size} points"
    )

    return region_trajectory


def process_hemisphere(
    subject_id,
    hemisphere,
    subjects_dir,
    trajectory_dir,
    output_dir,
):
    labels, ctab, region_names = load_annotation(
        subjects_dir,
        subject_id,
        hemisphere,
    )

    trajectory_dwi = load_dwi_trajectories(
        trajectory_dir,
        subject_id,
        hemisphere,
    )

    expected_points = (
        labels.shape[0] * DEPTH_SAMPLES
    )

    print(
        f"Expected trajectory points: {expected_points}"
    )

    print(
        f"Actual trajectory points: "
        f"{trajectory_dwi.shape[1]}"
    )

    if trajectory_dwi.shape[1] != expected_points:
        raise ValueError(
            "Trajectory point count does not match "
            "FreeSurfer vertex count × 21."
        )

    region_count = 0

    # Follow the original Brain_column_builder logic:
    # MATLAB-style regions i = 2..36, excluding i = 5.
    max_i = min(
        36,
        len(region_names),
        ctab.shape[0],
    )

    for i in range(
        2,
        max_i + 1,
    ):
        if i == 5:
            continue

        region_index = i - 1
        region_name = region_names[
            region_index
        ]

        vertex_indices = np.where(
            labels == region_index
        )[0]

        if vertex_indices.size == 0:
            continue

        save_region(
            subject_id=subject_id,
            hemisphere=hemisphere,
            region_name=region_name,
            vertex_indices=vertex_indices,
            trajectory_dwi=trajectory_dwi,
            output_dir=output_dir,
        )

        region_count += 1

    print()
    print(
        f"{hemisphere.upper()} regions written: "
        f"{region_count}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assign DWI-space cortical trajectories "
            "to Desikan-Killiany regions."
        )
    )

    parser.add_argument(
        "--subject-id",
        required=True,
    )

    parser.add_argument(
        "--subjects-dir",
        required=True,
        help=(
            "FreeSurfer SUBJECTS_DIR containing "
            "<subject_id>/label/lh.aparc.annot."
        ),
    )

    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help=(
            "Directory containing transformed "
            "trajectory MAT files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    print(f"Subject: {args.subject_id}")

    for hemisphere in [
        "lh",
        "rh",
    ]:
        process_hemisphere(
            subject_id=args.subject_id,
            hemisphere=hemisphere,
            subjects_dir=args.subjects_dir,
            trajectory_dir=args.trajectory_dir,
            output_dir=args.output_dir,
        )

    print()
    print("Region assignment complete.")


if __name__ == "__main__":
    main()