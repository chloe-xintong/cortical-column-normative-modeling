#!/usr/bin/env python3
"""Re-aggregate Figure 4A counts and Figure 4D matched-anchor differences.

This is a post-processing script: it reads cached per-subject anchor summaries
and never reloads profiles or recomputes pairwise profile dissimilarities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ANCHOR_COLUMNS = [
    "subject_id",
    "hemisphere",
    "anchor_vertex",
    "distance_bin_index",
    "distance_bin",
    "pair_type",
    "median_dissimilarity",
    "median_geodesic_distance_mm",
    "n_pairs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize_matched_subject(path: Path) -> pd.DataFrame:
    """Compute one subject's within/across difference on matched anchors."""
    table = pd.read_parquet(path, columns=ANCHOR_COLUMNS)
    boundary = table.loc[
        table["pair_type"].isin(["within_region", "across_boundary"])
    ].copy()
    wide = boundary.pivot(
        index=[
            "subject_id",
            "hemisphere",
            "anchor_vertex",
            "distance_bin_index",
            "distance_bin",
        ],
        columns="pair_type",
        values=["median_dissimilarity", "median_geodesic_distance_mm", "n_pairs"],
    )
    wide.columns = ["_".join(column) for column in wide.columns]
    wide = wide.dropna(
        subset=[
            "median_dissimilarity_within_region",
            "median_dissimilarity_across_boundary",
        ]
    ).reset_index()
    wide["across_minus_within"] = (
        wide["median_dissimilarity_across_boundary"]
        - wide["median_dissimilarity_within_region"]
    )
    return (
        wide.groupby(
            ["subject_id", "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            within_dissimilarity_matched=(
                "median_dissimilarity_within_region", "median"
            ),
            across_dissimilarity_matched=(
                "median_dissimilarity_across_boundary", "median"
            ),
            across_minus_within=("across_minus_within", "median"),
            matched_anchor_count=("anchor_vertex", "size"),
            matched_anchor_count_lh=(
                "hemisphere", lambda values: int((values == "lh").sum())
            ),
            matched_anchor_count_rh=(
                "hemisphere", lambda values: int((values == "rh").sum())
            ),
            within_pair_count=("n_pairs_within_region", "sum"),
            across_pair_count=("n_pairs_across_boundary", "sum"),
            within_distance_median=(
                "median_geodesic_distance_mm_within_region", "median"
            ),
            across_distance_median=(
                "median_geodesic_distance_mm_across_boundary", "median"
            ),
        )
    )


def summarize_matched_group(subject_table: pd.DataFrame) -> pd.DataFrame:
    subject_table = subject_table.copy()
    subject_table["absolute_distance_difference_mm"] = np.abs(
        subject_table["across_distance_median"]
        - subject_table["within_distance_median"]
    )
    summary = (
        subject_table.groupby(["distance_bin_index", "distance_bin"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            delta_group_median=("across_minus_within", "median"),
            delta_q25=("across_minus_within", lambda values: values.quantile(0.25)),
            delta_q75=("across_minus_within", lambda values: values.quantile(0.75)),
            matched_anchor_count_total=("matched_anchor_count", "sum"),
            matched_anchor_count_median_subject=("matched_anchor_count", "median"),
            within_pair_count_total=("within_pair_count", "sum"),
            across_pair_count_total=("across_pair_count", "sum"),
            within_distance_median=("within_distance_median", "median"),
            across_distance_median=("across_distance_median", "median"),
            absolute_distance_difference_mm=(
                "absolute_distance_difference_mm", "median"
            ),
        )
    )
    rng = np.random.default_rng(20261021)
    intervals = []
    for bin_index, table in subject_table.groupby("distance_bin_index"):
        values = table["across_minus_within"].to_numpy()
        boot = np.median(
            rng.choice(values, size=(10_000, values.size), replace=True), axis=1
        )
        intervals.append(
            {
                "distance_bin_index": bin_index,
                "delta_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                "delta_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
            }
        )
    return summary.merge(pd.DataFrame(intervals), on="distance_bin_index", how="left")


def summarize_a_with_counts(anchor_paths: list[Path]) -> pd.DataFrame:
    """Summarize A from anchors, retaining hemisphere in anchor identity."""
    subject_tables = []
    for path in anchor_paths:
        table = pd.read_parquet(path, columns=ANCHOR_COLUMNS)
        all_anchor = table.loc[table["pair_type"].eq("all")].copy()
        all_anchor["hemisphere_anchor"] = (
            all_anchor["hemisphere"].astype(str)
            + ":"
            + all_anchor["anchor_vertex"].astype(str)
        )
        subject_tables.append(
            all_anchor.groupby(
                [
                    "subject_id",
                    "distance_bin_index",
                    "distance_bin",
                    "pair_type",
                ],
                as_index=False,
            )
            .agg(
                median_dissimilarity=("median_dissimilarity", "median"),
                n_anchors=("hemisphere_anchor", "nunique"),
                n_pairs=("n_pairs", "sum"),
            )
        )
    subject_bins = pd.concat(subject_tables, ignore_index=True)
    return (
        subject_bins.groupby(
            ["distance_bin_index", "distance_bin", "pair_type"], as_index=False
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            median=("median_dissimilarity", "median"),
            q25=("median_dissimilarity", lambda values: values.quantile(0.25)),
            q75=("median_dissimilarity", lambda values: values.quantile(0.75)),
            anchor_count_total=("n_anchors", "sum"),
            anchor_count_median_subject=("n_anchors", "median"),
            pair_count_total=("n_pairs", "sum"),
            pair_count_median_subject=("n_pairs", "median"),
        )
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.results_root.glob("subjects/*/figure4_real_anchor_bins.parquet"))
    if not paths:
        raise FileNotFoundError(f"No anchor summaries under {args.results_root}")

    matched_subject = pd.concat(
        [summarize_matched_subject(path) for path in paths], ignore_index=True
    )
    matched_group = summarize_matched_group(matched_subject)
    matched_subject.to_csv(
        args.output_dir / "figure4_within_across_matched_subject_bins.csv",
        index=False,
    )
    matched_group.to_csv(
        args.output_dir / "figure4_within_across_matched_group_summary.csv",
        index=False,
    )

    summarize_a_with_counts(paths).to_csv(
        args.output_dir / "figure4_real_group_summary.csv", index=False
    )

    old_path = args.results_root / "cohort/figure4_within_across_subject_bins.csv"
    if old_path.is_file():
        old = pd.read_csv(old_path)
        old_summary = (
            old.groupby(["distance_bin_index", "distance_bin"], as_index=False)
            .agg(old_delta_median=("across_minus_within_dissimilarity", "median"))
        )
        matched_group.merge(
            old_summary, on=["distance_bin_index", "distance_bin"], how="left"
        ).to_csv(
            args.output_dir / "figure4_within_across_old_vs_matched.csv",
            index=False,
        )
    print(f"Saved matched-anchor Figure 4D summaries for {len(paths)} subjects")


if __name__ == "__main__":
    main()
