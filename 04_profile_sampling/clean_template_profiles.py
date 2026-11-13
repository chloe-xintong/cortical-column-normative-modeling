#!/usr/bin/env python3
"""Apply a Shared Validity Mask to Column Profiles

Description
-----------
Create cleaned vertex-level profiles by applying one common validity mask
across all requested metrics for each hemisphere.

Methods
-------
1. Load matching metric-profile NPZ files.
2. Confirm consistent vertex ordering and profile dimensions.
3. Combine mapping, bounds, finite-value, analysis, and column-valid masks.
4. Mark a column invalid in every metric if it fails any requested metric.
5. Save cleaned profiles and a cleaning summary.

Inputs
------
- Subject ID, raw profile directory, and requested metric names
- Output directory and optional hemisphere selection

Outputs
-------
- Cleaned per-metric profile NPZ files
- Per-hemisphere shared-mask cleaning JSON summaries

Usage
-----
python 04_profile_sampling/clean_template_profiles.py     --subject-id D0007     --input-dir /path/to/profiles_before_clean     --output-dir /path/to/profiles     --metrics md fa ad rd

Notes
-----
- The same retained vertices are used for every requested metric.
- Raw input profiles are not overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    return parser.parse_args()


def load_profile(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Profile file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def clean_hemisphere(args: argparse.Namespace, hemisphere: str) -> None:
    profiles = {}
    for metric in args.metrics:
        path = args.input_dir / f"{args.subject_id}_{hemisphere}_{metric}_profiles.npz"
        profiles[metric] = load_profile(path)

    reference = profiles[args.metrics[0]]
    vertex_ids = reference["fsaverage_vertex_id"]
    n_columns = len(vertex_ids)
    mapping_or_bounds_bad = ~reference["mapping_valid"].astype(bool)
    mapping_or_bounds_bad |= (
        ~reference["coordinate_in_bounds_mask"].astype(bool).all(axis=1)
    )
    mapping_or_bounds_bad |= ~reference["sampled_finite_mask"].astype(bool).all(axis=1)
    mapping_or_bounds_bad |= ~reference["sampling_valid"].astype(bool)
    mapping_or_bounds_bad |= ~reference["in_analysis_mask"].astype(bool)
    mapping_or_bounds_bad |= reference["is_zero_length"].astype(bool)
    metric_bad = np.zeros(n_columns, dtype=bool)

    for metric, data in profiles.items():
        if not np.array_equal(data["fsaverage_vertex_id"], vertex_ids):
            raise ValueError(f"Target vertex IDs differ for {metric} in {hemisphere}")
        values = np.asarray(data["profiles"], dtype=float)
        if values.shape[0] != n_columns:
            raise ValueError(f"Column count differs for {metric} in {hemisphere}")
        metric_bad |= ~np.isfinite(values).all(axis=1)
        metric_bad |= (values <= 0).any(axis=1)

    bad = mapping_or_bounds_bad | metric_bad

    keep = ~bad
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric, data in profiles.items():
        output = dict(data)
        cleaned_profiles = np.asarray(data["profiles"], dtype=float).copy()
        cleaned_profiles[~keep] = np.nan
        output["profiles"] = cleaned_profiles
        output["fsaverage_vertex_id"] = vertex_ids
        output["target_vertex_id"] = vertex_ids
        output["column_valid"] = keep
        output["cleaned"] = np.asarray(True)
        output["n_columns_before_cleaning"] = np.asarray(n_columns)
        output["n_columns_after_cleaning"] = np.asarray(int(keep.sum()))
        np.savez_compressed(
            args.output_dir / f"{args.subject_id}_{hemisphere}_{metric}_profiles.npz",
            **output,
        )

    summary = {
        "subject_id": args.subject_id,
        "hemisphere": hemisphere,
        "metrics": list(args.metrics),
        "n_columns_before_cleaning": int(n_columns),
        "n_bad_columns": int(bad.sum()),
        "n_columns_after_cleaning": int(keep.sum()),
        "bad_indices": np.flatnonzero(bad).astype(int).tolist(),
        "bad_fsaverage_vertex_ids": vertex_ids[bad].astype(int).tolist(),
        "bad_column_reasons": {
            "mapping_invalid_or_out_of_bounds": int(mapping_or_bounds_bad.sum()),
            "nonfinite_or_nonpositive_metric": int(metric_bad.sum()),
        },
    }
    (args.output_dir / f"{args.subject_id}_{hemisphere}_cleaning.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"{hemisphere}: {n_columns} columns -> {int(keep.sum())} valid columns")


def main() -> None:
    args = parse_args()
    for hemisphere in args.hemispheres:
        clean_hemisphere(args, hemisphere)


if __name__ == "__main__":
    main()
