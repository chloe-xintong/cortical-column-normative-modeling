#!/usr/bin/env python3
"""Summarize whole-cortex Figure 4A curves using only within-DK pairs.

Hierarchy: within-region pair -> hemisphere x anchor x 2-mm-bin median ->
subject median across bilateral cortical anchors -> group median/IQR.
Pair and anchor counts describe support and never weight the metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_BOOTSTRAP = 10_000
RANDOM_SEED = 20261022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def bootstrap_median_ci(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    medians = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, [.025, .975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = pd.read_csv(args.manifest, dtype={"subject_id": str})[
        "subject_id"
    ].drop_duplicates().tolist()
    tables = []
    missing = []
    filename = "figure4_relative_mae_within_region_anchor_bins.parquet"
    for subject_id in subject_ids:
        path = args.results_root / "subjects" / subject_id / filename
        if path.is_file():
            tables.append(pd.read_parquet(path))
        else:
            missing.append(subject_id)
    if not tables:
        raise RuntimeError(f"No {filename} files found")
    anchor = pd.concat(tables, ignore_index=True)
    if "subject_id" not in anchor:
        raise ValueError("Anchor-level tables lack subject_id")
    anchor["hemisphere_anchor"] = (
        anchor["hemisphere"].astype(str) + ":"
        + anchor["anchor_vertex"].astype(str)
    )
    subject = (
        anchor.groupby(
            ["subject_id", "distance_bin_index", "distance_bin"], as_index=False
        )
        .agg(
            profile_dissimilarity=("median_dissimilarity", "median"),
            profile_mae=("median_profile_mae", "median"),
            symmetric_relative_mae_pct=(
                "median_symmetric_relative_mae_pct", "median"
            ),
            anchor_count=("hemisphere_anchor", "nunique"),
            pair_count=("n_pairs", "sum"),
            region_count=("anchor_dk_label", "nunique"),
            hemispheres_contributing=("hemisphere", "nunique"),
        )
        .sort_values(["subject_id", "distance_bin_index"])
    )
    subject.to_csv(
        args.output_dir / "figure4_within_region_whole_cortex_subject_bins.csv",
        index=False,
    )

    pairs = pd.read_parquet(
        args.pair_table,
        columns=[
            "hemisphere", "anchor_vertex", "neighbor_vertex", "anchor_dk_label",
            "neighbor_dk_label", "distance_bin_index", "distance_bin", "pair_type",
        ],
    )
    fixed = pairs.loc[
        pairs["pair_type"].eq("within_region")
        & pairs["anchor_dk_label"].eq(pairs["neighbor_dk_label"])
    ].copy()
    fixed["hemisphere_anchor"] = (
        fixed["hemisphere"].astype(str) + ":"
        + fixed["anchor_vertex"].astype(str)
    )
    fixed_counts = (
        fixed.groupby(["distance_bin_index", "distance_bin"], as_index=False)
        .agg(
            fixed_pair_count=("neighbor_vertex", "size"),
            fixed_anchor_count=("hemisphere_anchor", "nunique"),
            fixed_region_count=("anchor_dk_label", "nunique"),
        )
    )

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for (bin_index, distance_bin), frame in subject.groupby(
        ["distance_bin_index", "distance_bin"], sort=True
    ):
        row = {
            "distance_bin_index": int(bin_index),
            "distance_bin": distance_bin,
            "distance_bin_center_mm": 2 * int(bin_index) + 1,
            "n_subjects": int(frame["subject_id"].nunique()),
            "anchor_count_total": int(frame["anchor_count"].sum()),
            "anchor_count_median_subject": float(frame["anchor_count"].median()),
            "valid_pair_count_total": int(frame["pair_count"].sum()),
            "valid_pair_count_median_subject": float(frame["pair_count"].median()),
            "region_count_median_subject": float(frame["region_count"].median()),
        }
        for metric in (
            "profile_dissimilarity", "profile_mae", "symmetric_relative_mae_pct"
        ):
            values = frame[metric].dropna().to_numpy(float)
            low, high = bootstrap_median_ci(values, rng)
            row.update({
                f"{metric}_median": float(np.median(values)),
                f"{metric}_q25": float(np.quantile(values, .25)),
                f"{metric}_q75": float(np.quantile(values, .75)),
                f"{metric}_bootstrap_ci95_low": low,
                f"{metric}_bootstrap_ci95_high": high,
            })
        rows.append(row)
    group = pd.DataFrame(rows).merge(
        fixed_counts,
        on=["distance_bin_index", "distance_bin"],
        how="left",
        validate="one_to_one",
    )
    group.to_csv(
        args.output_dir / "figure4_within_region_whole_cortex_group_summary.csv",
        index=False,
    )

    sys.path.insert(0, str(args.repo_root))
    from qc.publication_style import (  # pylint: disable=import-outside-toplevel
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style,
        style_axes,
    )

    set_publication_style()
    x = group["distance_bin_center_mm"].to_numpy(float)
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_IN, 3.15),
        gridspec_kw={"wspace": .36},
    )
    specifications = [
        (
            "profile_dissimilarity", "Within-region profile dissimilarity (1 − r)",
            PALETTE["blue"],
        ),
        (
            "symmetric_relative_mae_pct", "Within-region symmetric relative MAE (%)",
            PALETTE["orange"],
        ),
    ]
    for ax, (metric, ylabel, color) in zip(axes, specifications):
        ax.plot(
            x, group[f"{metric}_median"], color=color,
            marker="o", markersize=3,
        )
        ax.fill_between(
            x, group[f"{metric}_q25"], group[f"{metric}_q75"],
            color=color, alpha=.2, linewidth=0,
        )
        ax.set(
            xlabel="White-surface geodesic distance (mm)",
            ylabel=ylabel,
            xlim=(0, 30), xticks=np.arange(0, 31, 5),
        )
        style_axes(ax, grid=True)
    fig.subplots_adjust(left=.10, right=.985, bottom=.20, top=.98, wspace=.36)
    save_publication_figure(
        fig, args.figure_dir / "Figure4A_within_region_whole_cortex"
    )
    plt.close(fig)

    status = {
        "status": "complete" if not missing else "incomplete",
        "subjects_expected": len(subject_ids),
        "subjects_completed": int(subject["subject_id"].nunique()),
        "subjects_missing": missing,
        "fixed_within_region_pair_count": int(fixed_counts["fixed_pair_count"].sum()),
        "aggregation": (
            "within-region pair -> hemisphere x anchor x 2-mm-bin median -> "
            "subject median across bilateral cortex -> group median/IQR"
        ),
        "pair_count_weighting": False,
        "bootstrap_iterations": N_BOOTSTRAP,
    }
    (args.output_dir / "figure4_within_region_whole_cortex_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
