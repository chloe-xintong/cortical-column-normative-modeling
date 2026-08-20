#!/usr/bin/env python

# Purpose:
# Compute cortical thickness from white-pial vertex pairs
# and summarize thickness within FreeSurfer aparc regions.
#
# Thickness is defined as the Euclidean distance between
# the white and pial endpoints of each trajectory.
#
# This reproduces the behavior of the original
# Brain_column_builder/get_thickness.py.
#
# Usage:
# python trajectory_processing/compute_thickness.py \
#   --subject-id <ID> \
#   --subjects-dir <FreeSurfer subjects directory> \
#   --trajectory-dir <directory containing pair files> \
#   --output-dir <thickness output directory>

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat


def load_pair(pair_path, hemisphere):
    pair_path = Path(pair_path)

    if not pair_path.is_file():
        raise FileNotFoundError(
            f"Pair file not found: {pair_path}"
        )

    data = loadmat(
        str(pair_path)
    )

    variable_name = (
        f"{hemisphere}_pair"
    )

    if variable_name not in data:
        raise KeyError(
            f"{variable_name} not found "
            f"in {pair_path}"
        )

    pair = np.asarray(
        data[variable_name],
        dtype=float,
    )

    if (
        pair.ndim != 2
        or pair.shape[1] != 6
    ):
        raise ValueError(
            f"Expected pair shape (N, 6), "
            f"got {pair.shape}"
        )

    return pair


def compute_thickness(pair):
    white = pair[:, 0:3]
    pial = pair[:, 3:6]

    difference = white - pial

    thickness = np.sqrt(
        np.sum(
            difference ** 2,
            axis=1,
        )
    )

    return thickness.reshape(
        -1,
        1,
    )


def load_annotation(annotation_path):
    annotation_path = Path(
        annotation_path
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: "
            f"{annotation_path}"
        )

    labels, color_table, names = (
        nib.freesurfer.io.read_annot(
            str(annotation_path)
        )
    )

    names = [
        name.decode("utf-8")
        if isinstance(name, bytes)
        else name
        for name in names
    ]

    return (
        labels,
        color_table,
        names,
    )


def save_region_thickness(
    subject_id,
    hemisphere,
    region_name,
    thickness_values,
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    base_name = (
        f"{subject_id}_"
        f"{hemisphere}_"
        f"{region_name}_"
        f"thickness"
    )

    mat_path = (
        output_dir
        / f"{base_name}.mat"
    )

    csv_path = (
        output_dir
        / f"{base_name}.csv"
    )

    variable_name = (
        f"thickness_"
        f"{hemisphere}_region"
    )

    savemat(
        str(mat_path),
        {
            variable_name:
            thickness_values
        },
    )

    np.savetxt(
        str(csv_path),
        thickness_values,
        delimiter=",",
    )


def process_hemisphere(
    subject_id,
    hemisphere,
    pair,
    annotation_path,
    output_dir,
):
    thickness = compute_thickness(
        pair
    )

    labels, color_table, names = (
        load_annotation(
            annotation_path
        )
    )

    if labels.shape[0] != thickness.shape[0]:
        raise ValueError(
            f"{hemisphere}: annotation vertices "
            f"({labels.shape[0]}) do not match "
            f"pair vertices "
            f"({thickness.shape[0]})."
        )

    print()
    print(
        f"{hemisphere.upper()} thickness:"
    )
    print(
        f"  vertices: "
        f"{thickness.shape[0]}"
    )

    region_summary = []

    # Reproduce MATLAB loop:
    # for i = 2:36
    #     if i == 5
    #         continue
    max_i = min(
        36,
        len(names),
        len(color_table),
    )

    for i in range(
        2,
        max_i + 1,
    ):
        if i == 5:
            continue

        region_index = i - 1
        region_name = names[
            region_index
        ]

        # read_annot returns annotation
        # labels as region indices.
        vertex_indices = np.where(
            labels == region_index
        )[0]

        if vertex_indices.size == 0:
            continue

        region_thickness = thickness[
            vertex_indices,
            :
        ]

        save_region_thickness(
            subject_id=subject_id,
            hemisphere=hemisphere,
            region_name=region_name,
            thickness_values=region_thickness,
            output_dir=output_dir,
        )

        n_vertices = int(
            region_thickness.shape[0]
        )

        mean_thickness = float(
            region_thickness.mean()
        )

        if n_vertices > 1:
            std_thickness = float(
                region_thickness.std(
                    ddof=1
                )
            )
        else:
            std_thickness = 0.0

        region_summary.append(
            [
                hemisphere,
                region_name,
                n_vertices,
                mean_thickness,
                std_thickness,
            ]
        )

        print(
            f"  {region_name}: "
            f"n={n_vertices}, "
            f"mean={mean_thickness:.4f}"
        )

    return region_summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute cortical thickness "
            "from white-pial trajectory pairs."
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
            "FreeSurfer SUBJECTS_DIR "
            "containing the subject folder."
        ),
    )

    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help=(
            "Directory containing "
            "<subject>_pair_lh.mat and "
            "<subject>_pair_rh.mat."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for regional "
            "thickness outputs."
        ),
    )

    args = parser.parse_args()

    subject_id = args.subject_id

    subjects_dir = Path(
        args.subjects_dir
    )

    trajectory_dir = Path(
        args.trajectory_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lh_pair_path = (
        trajectory_dir
        / f"{subject_id}_pair_lh.mat"
    )

    rh_pair_path = (
        trajectory_dir
        / f"{subject_id}_pair_rh.mat"
    )

    lh_pair = load_pair(
        lh_pair_path,
        "lh",
    )

    rh_pair = load_pair(
        rh_pair_path,
        "rh",
    )

    subject_dir = (
        subjects_dir
        / subject_id
    )

    lh_annotation = (
        subject_dir
        / "label"
        / "lh.aparc.annot"
    )

    rh_annotation = (
        subject_dir
        / "label"
        / "rh.aparc.annot"
    )

    summary = []

    summary.extend(
        process_hemisphere(
            subject_id=subject_id,
            hemisphere="lh",
            pair=lh_pair,
            annotation_path=lh_annotation,
            output_dir=output_dir,
        )
    )

    summary.extend(
        process_hemisphere(
            subject_id=subject_id,
            hemisphere="rh",
            pair=rh_pair,
            annotation_path=rh_annotation,
            output_dir=output_dir,
        )
    )

    summary_path = (
        output_dir
        / (
            f"{subject_id}_"
            f"thickness_region_means.csv"
        )
    )

    with open(
        summary_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "hemi",
                "region",
                "n_vertices",
                "mean_thickness",
                "std_thickness",
            ]
        )

        writer.writerows(
            summary
        )

    print()
    print(
        f"Thickness summary saved: "
        f"{summary_path}"
    )

    print()
    print(
        "Thickness extraction complete."
    )


if __name__ == "__main__":
    main()