#!/usr/bin/env python3
"""Compute Figure 4A symmetric relative profile MAE for one subject.

Description
-----------
The aggregation hierarchy is pair -> anchor x 2-mm bin median -> subject
median.  The metric matches the symmetric profile NMAE used in Figure 2:

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-table`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_relative_mae.py --help
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compute_real_spatial_curves import load_profiles, select_subject


CHUNK_SIZE = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    return parser.parse_args()


def hemisphere_anchor_bins(
    profiles: np.ndarray,
    valid_vertices: np.ndarray,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = pairs["anchor_vertex"].to_numpy(np.int64)
    neighbors = pairs["neighbor_vertex"].to_numpy(np.int64)
    parts = []
    for start in range(0, len(pairs), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(pairs))
        anchor_ids = anchors[start:stop]
        neighbor_ids = neighbors[start:stop]
        valid = valid_vertices[anchor_ids] & valid_vertices[neighbor_ids]
        if not valid.any():
            continue
        local = np.flatnonzero(valid)
        left = profiles[anchor_ids[local]]
        right = profiles[neighbor_ids[local]]
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        correlation_denominator = (
            np.linalg.norm(left_centered, axis=1)
            * np.linalg.norm(right_centered, axis=1)
        )
        correlation = np.divide(
            np.sum(left_centered * right_centered, axis=1),
            correlation_denominator,
            out=np.full(len(left), np.nan, dtype=float),
            where=correlation_denominator > 0,
        )
        mae = np.mean(np.abs(left - right), axis=1)
        denominator = np.median(np.abs(left), axis=1) + np.median(
            np.abs(right), axis=1
        )
        relative = np.divide(
            200.0 * mae,
            denominator,
            out=np.full(len(mae), np.nan, dtype=float),
            where=np.isfinite(denominator) & (denominator > 0),
        )
        source = pairs.iloc[start + local]
        part = source[
            [
                "hemisphere", "anchor_vertex", "anchor_dk_label",
                "distance_bin_index", "distance_bin", "pair_type",
            ]
        ].copy()
        part["profile_dissimilarity"] = (1.0 - correlation).astype(np.float32)
        part["profile_mae"] = mae.astype(np.float32)
        part["symmetric_relative_mae_pct"] = relative.astype(np.float32)
        parts.append(part)
    if not parts:
        raise RuntimeError("No valid profile pairs")
    pair_values = pd.concat(parts, ignore_index=True)
    pair_values = pair_values.loc[
        np.isfinite(pair_values["profile_dissimilarity"])
        & np.isfinite(pair_values["profile_mae"])
        & np.isfinite(pair_values["symmetric_relative_mae_pct"])
    ]
    all_anchor = (
        pair_values.groupby(
            ["hemisphere", "anchor_vertex", "distance_bin_index", "distance_bin"],
            as_index=False,
            observed=True,
        )
        .agg(
            median_dissimilarity=("profile_dissimilarity", "median"),
            median_profile_mae=("profile_mae", "median"),
            median_symmetric_relative_mae_pct=(
                "symmetric_relative_mae_pct", "median"
            ),
            n_pairs=("symmetric_relative_mae_pct", "size"),
        )
    )
    within_pairs = pair_values.loc[pair_values["pair_type"].eq("within_region")]
    within_anchor = (
        within_pairs.groupby(
            [
                "hemisphere", "anchor_vertex", "anchor_dk_label",
                "distance_bin_index", "distance_bin",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            median_dissimilarity=("profile_dissimilarity", "median"),
            median_profile_mae=("profile_mae", "median"),
            median_symmetric_relative_mae_pct=(
                "symmetric_relative_mae_pct", "median"
            ),
            n_pairs=("symmetric_relative_mae_pct", "size"),
        )
    )
    return all_anchor, within_anchor


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    started = time.time()

    try:
        pairs = pd.read_parquet(args.pair_table)
        anchor_parts = []
        within_region_parts = []
        for hemisphere in ("lh", "rh"):
            profiles, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            hemi_pairs = pairs.loc[pairs["hemisphere"].eq(hemisphere)].reset_index(
                drop=True
            )
            all_anchor, within_anchor = hemisphere_anchor_bins(
                profiles, valid, hemi_pairs
            )
            anchor_parts.append(all_anchor)
            within_region_parts.append(within_anchor)
        anchor = pd.concat(anchor_parts, ignore_index=True)
        anchor.insert(0, "subject_id", subject_id)
        anchor.to_parquet(
            output_dir / "figure4_relative_mae_anchor_bins.parquet", index=False
        )
        subject = (
            anchor.groupby(["distance_bin_index", "distance_bin"], as_index=False)
            .agg(
                profile_dissimilarity=("median_dissimilarity", "median"),
                profile_mae=("median_profile_mae", "median"),
                symmetric_relative_mae_pct=(
                    "median_symmetric_relative_mae_pct", "median"
                ),
                anchor_count=("anchor_vertex", "size"),
                pair_count=("n_pairs", "sum"),
            )
        )
        subject.insert(0, "subject_id", subject_id)
        subject.to_csv(
            output_dir / "figure4_relative_mae_subject_bins.csv", index=False
        )
        within_region = pd.concat(within_region_parts, ignore_index=True)
        within_region.insert(0, "subject_id", subject_id)
        within_region.to_parquet(
            output_dir / "figure4_relative_mae_within_region_anchor_bins.parquet",
            index=False,
        )
        within_region_subject = (
            within_region.groupby(
                ["anchor_dk_label", "distance_bin_index", "distance_bin"],
                as_index=False,
            )
            .agg(
                profile_dissimilarity=("median_dissimilarity", "median"),
                profile_mae=("median_profile_mae", "median"),
                symmetric_relative_mae_pct=(
                    "median_symmetric_relative_mae_pct", "median"
                ),
                anchor_count=("anchor_vertex", "size"),
                pair_count=("n_pairs", "sum"),
                hemispheres_contributing=("hemisphere", "nunique"),
            )
        )
        within_region_subject.insert(0, "subject_id", subject_id)
        within_region_subject.to_csv(
            output_dir / "figure4_relative_mae_within_region_subject_bins.csv",
            index=False,
        )
        completion.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "complete",
                    "definition": (
                        "200 * profile_MAE / (median_abs_profile_anchor + "
                        "median_abs_profile_neighbor)"
                    ),
                    "aggregation": (
                        "pair -> hemisphere x anchor x distance-bin median -> "
                        "subject median"
                    ),
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        failure.unlink(missing_ok=True)
    except Exception as error:
        failure.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "runtime_seconds": time.time() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        completion.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
