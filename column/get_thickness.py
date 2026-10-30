#!/usr/bin/env python3
"""Export geometry and thickness metadata from template-driven columns.

The column builder already interpolates FreeSurfer vertexwise thickness with
the same triangle weights used for the white and pial endpoints. This utility
validates and exports those per-fsaverage-vertex values without reconstructing
native surface pairs or assuming that row numbers are native vertex IDs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
    "fsaverage_vertex_id",
    "white_coordinates_t1",
    "pial_coordinates_t1",
    "column_length",
    "cortical_thickness",
    "template_label_name",
    "in_analysis_mask",
    "is_zero_length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--columns-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    return parser.parse_args()


def export_hemisphere(
    subject_id: str,
    columns_dir: Path,
    output_dir: Path,
    hemisphere: str,
) -> Path:
    input_path = columns_dir / f"{subject_id}_{hemisphere}_columns.npz"
    if not input_path.is_file():
        raise FileNotFoundError(f"Column file not found: {input_path}")

    with np.load(input_path, allow_pickle=True) as data:
        missing = REQUIRED_FIELDS - set(data.files)
        if missing:
            raise ValueError(
                f"Column file is missing required fields for {hemisphere}: "
                f"{sorted(missing)}"
            )
        vertex_ids = np.asarray(data["fsaverage_vertex_id"], dtype=np.int64)
        white = np.asarray(data["white_coordinates_t1"], dtype=float)
        pial = np.asarray(data["pial_coordinates_t1"], dtype=float)
        saved_length = np.asarray(data["column_length"], dtype=float)
        thickness = np.asarray(data["cortical_thickness"], dtype=float)
        labels = np.asarray(data["template_label_name"]).astype(str)
        in_mask = np.asarray(data["in_analysis_mask"], dtype=bool)
        zero_length = np.asarray(data["is_zero_length"], dtype=bool)

    recomputed_length = np.linalg.norm(pial - white, axis=1)
    if not np.allclose(saved_length, recomputed_length, equal_nan=True):
        raise ValueError(
            f"Saved column lengths do not match endpoint distances for {hemisphere}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_{hemisphere}_column_geometry.npz"
    np.savez_compressed(
        output_path,
        subject_id=subject_id,
        hemisphere=hemisphere,
        fsaverage_vertex_id=vertex_ids,
        white_coordinates_t1=white,
        pial_coordinates_t1=pial,
        column_length=saved_length,
        cortical_thickness=thickness,
        fsaverage_label=labels,
        in_analysis_mask=in_mask,
        is_zero_length=zero_length,
        source_column_file=str(input_path),
    )

    csv_path = output_dir / f"{subject_id}_{hemisphere}_column_geometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "fsaverage_vertex_id",
                "fsaverage_label",
                "column_length",
                "cortical_thickness",
                "in_analysis_mask",
                "is_zero_length",
            ]
        )
        writer.writerows(
            zip(
                vertex_ids,
                labels,
                saved_length,
                thickness,
                in_mask,
                zero_length,
            )
        )
    return output_path


def main() -> None:
    args = parse_args()
    for hemisphere in args.hemispheres:
        output_path = export_hemisphere(
            args.subject_id,
            args.columns_dir,
            args.output_dir,
            hemisphere,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
