#!/usr/bin/env python3
"""Collapse each ico4 column's 21-depth MD profile to one mean-MD feature.

Description
-----------
Collapse each ico4 column's 21-depth MD profile to one mean-MD feature.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--feature-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/build_ico4_column_mean_md.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_DEPTH = 21
SOURCE_REPRESENTATION = "ico4_md"
OUTPUT_REPRESENTATION = "ico4_column_mean_md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_dir = args.feature_dir / "matrices"
    metadata_dir = args.feature_dir / "feature_metadata"
    source_matrix_path = matrix_dir / f"X_{SOURCE_REPRESENTATION}.npy"
    source_metadata_path = metadata_dir / f"features_{SOURCE_REPRESENTATION}.csv"
    output_matrix_path = matrix_dir / f"X_{OUTPUT_REPRESENTATION}.npy"
    output_metadata_path = metadata_dir / f"features_{OUTPUT_REPRESENTATION}.csv"
    report_path = args.feature_dir / f"{OUTPUT_REPRESENTATION}_build_report.json"

    source = np.load(source_matrix_path, mmap_mode="r")
    metadata = pd.read_csv(source_metadata_path)
    required = {
        "feature_index", "hemisphere", "ico_level", "ico_node_id",
        "fsaverage_vertex_id", "depth_index",
    }
    if source.ndim != 2 or source.shape[1] != len(metadata):
        raise RuntimeError("ico4 MD matrix shape and metadata length disagree")
    if not required.issubset(metadata.columns):
        raise RuntimeError(f"ico4 metadata lacks columns: {sorted(required - set(metadata.columns))}")
    if not np.array_equal(metadata["feature_index"].to_numpy(), np.arange(len(metadata))):
        raise RuntimeError("ico4 MD feature indices are not contiguous")
    if not metadata["ico_level"].eq(4).all():
        raise RuntimeError("Source metadata contains a non-ico4 feature")

    node_keys = ["hemisphere", "ico_node_id", "fsaverage_vertex_id"]
    group_sizes = metadata.groupby(node_keys, sort=False).size()
    if not group_sizes.eq(N_DEPTH).all():
        raise RuntimeError("Every ico4 node must have exactly 21 depth features")
    expected_depths = np.tile(np.arange(N_DEPTH), len(group_sizes))
    if not np.array_equal(metadata["depth_index"].to_numpy(), expected_depths):
        raise RuntimeError("ico4 depth features are not ordered 0..20 within each node")
    if source.shape[1] % N_DEPTH:
        raise RuntimeError("ico4 MD feature count is not divisible by 21")

    n_subjects = source.shape[0]
    n_nodes = source.shape[1] // N_DEPTH
    reshaped = np.asarray(source).reshape(n_subjects, n_nodes, N_DEPTH)
    finite = np.isfinite(reshaped)
    finite_count = finite.sum(axis=2)
    summed = np.where(finite, reshaped, 0.0).sum(axis=2, dtype=np.float64)
    mean_md = np.full((n_subjects, n_nodes), np.nan, dtype=np.float32)
    valid = finite_count > 0
    mean_md[valid] = (summed[valid] / finite_count[valid]).astype(np.float32)

    output = np.lib.format.open_memmap(
        output_matrix_path, mode="w+", dtype=np.float32, shape=mean_md.shape
    )
    output[:] = mean_md
    output.flush()

    scalar_metadata = metadata.iloc[::N_DEPTH].copy().reset_index(drop=True)
    scalar_metadata["feature_index"] = np.arange(n_nodes, dtype=np.int64)
    scalar_metadata = scalar_metadata.drop(
        columns=[column for column in ("depth_index", "depth_fraction", "depth_direction")
                 if column in scalar_metadata.columns]
    )
    scalar_metadata["aggregation"] = "mean_MD_across_available_21_depth_samples"
    scalar_metadata["n_source_depths"] = N_DEPTH
    scalar_metadata["source_representation"] = SOURCE_REPRESENTATION
    scalar_metadata.to_csv(output_metadata_path, index=False)

    reloaded = np.load(output_matrix_path, mmap_mode="r")
    if reloaded.shape != (n_subjects, n_nodes):
        raise RuntimeError("Saved ico4 column-mean matrix has the wrong shape")
    if not np.allclose(reloaded, mean_md, equal_nan=True, atol=0, rtol=0):
        raise RuntimeError("Saved ico4 column-mean matrix differs from the computed values")
    if len(scalar_metadata) != n_nodes or scalar_metadata[node_keys].duplicated().any():
        raise RuntimeError("Saved ico4 column-mean metadata is not one-to-one with nodes")

    report = {
        "status": "complete",
        "representation": OUTPUT_REPRESENTATION,
        "definition": (
            "For each subject and each retained ico4 column, calculate the arithmetic mean of "
            "the available MD values across its 21 ordered cortical-depth samples. A column with "
            "no finite depth sample remains NaN for fold-local median imputation."
        ),
        "source_matrix": str(source_matrix_path),
        "source_shape": list(source.shape),
        "output_matrix": str(output_matrix_path),
        "output_shape": list(reloaded.shape),
        "n_subjects": n_subjects,
        "n_ico4_columns": n_nodes,
        "n_source_depths_per_column": N_DEPTH,
        "finite_fraction": float(np.isfinite(reloaded).mean()),
        "n_all_missing_subject_columns": int((finite_count == 0).sum()),
        "minimum_available_depths_among_nonmissing_columns": int(finite_count[valid].min()),
        "maximum_available_depths": int(finite_count.max()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
