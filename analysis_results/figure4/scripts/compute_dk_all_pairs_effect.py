#!/usr/bin/env python3
"""Compute one subject's all-pair DK-region effect from fixed Figure 4A pairs.

The primary hierarchy is:
pair -> anchor x 0.5-mm sub-bin median -> hemisphere x anchor-DK-region
x sub-bin median -> subject median.  Within- and across-region estimates are
matched at the region/sub-bin level but do not need to come from the same
anchor.
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
from spatial_metrics import pair_metrics


PAIR_TYPES = ("within_region", "across_boundary")
SUBBIN_WIDTH_MM = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    return parser.parse_args()


def anchor_summaries(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics.loc[metrics["pair_valid"]].copy()
    valid["distance_subbin_index"] = np.floor(
        valid["geodesic_distance_mm"] / SUBBIN_WIDTH_MM
    ).astype(np.int16)
    return (
        valid.groupby(
            [
                "hemisphere",
                "anchor_dk_label",
                "anchor_vertex",
                "distance_bin_index",
                "distance_bin",
                "distance_subbin_index",
                "pair_type",
            ],
            observed=True,
        )
        .agg(
            median_dissimilarity=("profile_dissimilarity", "median"),
            median_profile_mae=("profile_mae", "median"),
            median_geodesic_distance_mm=("geodesic_distance_mm", "median"),
            n_pairs=("profile_dissimilarity", "size"),
        )
        .reset_index()
    )


def region_summaries(anchor: pd.DataFrame) -> pd.DataFrame:
    return (
        anchor.groupby(
            [
                "hemisphere",
                "anchor_dk_label",
                "distance_bin_index",
                "distance_bin",
                "distance_subbin_index",
                "pair_type",
            ],
            observed=True,
        )
        .agg(
            median_dissimilarity=("median_dissimilarity", "median"),
            median_profile_mae=("median_profile_mae", "median"),
            median_geodesic_distance_mm=(
                "median_geodesic_distance_mm", "median"
            ),
            n_anchors=("anchor_vertex", "nunique"),
            n_anchor_subbins=("anchor_vertex", "size"),
            n_pairs=("n_pairs", "sum"),
        )
        .reset_index()
    )


def match_region_subbins(region: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "hemisphere",
        "anchor_dk_label",
        "distance_bin_index",
        "distance_bin",
        "distance_subbin_index",
    ]
    value_columns = [
        "median_dissimilarity",
        "median_profile_mae",
        "median_geodesic_distance_mm",
        "n_anchors",
        "n_anchor_subbins",
        "n_pairs",
    ]
    within = region.loc[region["pair_type"].eq(PAIR_TYPES[0]), keys + value_columns]
    across = region.loc[region["pair_type"].eq(PAIR_TYPES[1]), keys + value_columns]
    within = within.rename(columns={name: f"within_{name}" for name in value_columns})
    across = across.rename(columns={name: f"across_{name}" for name in value_columns})
    matched = within.merge(across, on=keys, how="inner", validate="one_to_one")
    matched["across_minus_within"] = (
        matched["across_median_dissimilarity"]
        - matched["within_median_dissimilarity"]
    )
    matched["across_minus_within_mae"] = (
        matched["across_median_profile_mae"]
        - matched["within_median_profile_mae"]
    )
    matched["distance_difference_mm"] = (
        matched["across_median_geodesic_distance_mm"]
        - matched["within_median_geodesic_distance_mm"]
    )
    matched["hemisphere_region"] = (
        matched["hemisphere"].astype(str)
        + ":"
        + matched["anchor_dk_label"].astype(str)
    )
    return matched


def subject_summaries(matched: pd.DataFrame) -> pd.DataFrame:
    return (
        matched.groupby(
            ["distance_bin_index", "distance_bin"], as_index=False
        )
        .agg(
            within_dissimilarity=("within_median_dissimilarity", "median"),
            across_dissimilarity=("across_median_dissimilarity", "median"),
            across_minus_within=("across_minus_within", "median"),
            within_profile_mae=("within_median_profile_mae", "median"),
            across_profile_mae=("across_median_profile_mae", "median"),
            across_minus_within_mae=("across_minus_within_mae", "median"),
            matched_region_subbin_count=("across_minus_within", "size"),
            matched_region_count=("hemisphere_region", "nunique"),
            within_anchor_subbin_count=("within_n_anchor_subbins", "sum"),
            across_anchor_subbin_count=("across_n_anchor_subbins", "sum"),
            within_pair_count=("within_n_pairs", "sum"),
            across_pair_count=("across_n_pairs", "sum"),
            within_distance_median=(
                "within_median_geodesic_distance_mm", "median"
            ),
            across_distance_median=(
                "across_median_geodesic_distance_mm", "median"
            ),
            distance_difference_median_mm=("distance_difference_mm", "median"),
        )
    )


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
        if not bool(row["input_complete"]):
            raise FileNotFoundError(f"Incomplete inputs: {row['missing_inputs']}")
        pairs = pd.read_parquet(args.pair_table)
        if not set(PAIR_TYPES).issubset(set(pairs["pair_type"].unique())):
            raise ValueError("Fixed pair table lacks within/across DK labels")

        anchor_parts = []
        valid_pair_count = 0
        for hemisphere in ("lh", "rh"):
            profiles, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            hemi_pairs = pairs.loc[pairs["hemisphere"].eq(hemisphere)].reset_index(
                drop=True
            )
            metrics = pair_metrics(profiles, valid, hemi_pairs)
            valid_pair_count += int(metrics["pair_valid"].sum())
            anchor_parts.append(anchor_summaries(metrics))

        anchor = pd.concat(anchor_parts, ignore_index=True)
        region = region_summaries(anchor)
        matched = match_region_subbins(region)
        if matched.empty:
            raise RuntimeError("No matched within/across region-subbin cells")
        matched.insert(0, "subject_id", subject_id)
        subject = subject_summaries(matched)
        subject.insert(0, "subject_id", subject_id)

        matched.to_parquet(
            output_dir / "figure4_dk_all_pairs_region_subbins.parquet", index=False
        )
        subject.to_csv(
            output_dir / "figure4_dk_all_pairs_subject_bins.csv", index=False
        )
        completion.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "complete",
                    "pair_table": str(args.pair_table),
                    "subbin_width_mm": SUBBIN_WIDTH_MM,
                    "valid_pair_count": valid_pair_count,
                    "matched_region_subbin_rows": len(matched),
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024.0,
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
