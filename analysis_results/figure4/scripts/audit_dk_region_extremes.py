#!/usr/bin/env python3
"""Audit extreme Figure 4D DK-region effects and pair composition.

Description
-----------
Audit extreme Figure 4D DK-region effects and pair composition.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--region-subbin-table`` command-line input
- ``--bilateral-subject-table`` command-line input
- ``--pair-table`` command-line input
- ``--template-root`` command-line input
- ``--output-dir`` command-line input
- ``--figure-dir`` command-line input
- ``--repo-root`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/audit_dk_region_extremes.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


EXTREME_REGIONS = (
    "caudalanteriorcingulate",
    "inferiortemporal",
    "temporalpole",
    "fusiform",
)
MIN_PAIRS = 20
MIN_LOCAL_BINS = 3
N_BOOTSTRAP = 10_000
RANDOM_SEED = 20261021


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-subbin-table", type=Path, required=True)
    parser.add_argument("--bilateral-subject-table", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def dk_key(template_root: Path) -> pd.DataFrame:
    rows = []
    for hemisphere in ("lh", "rh"):
        _, _, names = fsio.read_annot(
            str(template_root / "label" / f"{hemisphere}.aparc.annot")
        )
        for label, raw_name in enumerate(names):
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            rows.append(
                {"hemisphere": hemisphere, "dk_label": label, "dk_region": name}
            )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    boot = np.median(values[indices], axis=1)
    return tuple(np.quantile(boot, [.025, .975]))


def local_summary(table: pd.DataFrame, scope: str, rng: np.random.Generator) -> dict:
    local = (
        table.loc[table["distance_bin_index"].lt(5) & table["eligible"]]
        .groupby("subject_id", as_index=False)
        .agg(local_effect_0_10=("delta_d", "mean"), valid_bins=("delta_d", "size"))
    )
    local = local.loc[local["valid_bins"].ge(MIN_LOCAL_BINS)]
    values = local["local_effect_0_10"].to_numpy(float)
    low, high = bootstrap_ci(values, rng)
    return {
        "scope": scope,
        "n_subjects": len(values),
        "local_effect_median": np.median(values),
        "local_effect_q25": np.quantile(values, .25),
        "local_effect_q75": np.quantile(values, .75),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "positive_subject_fraction": np.mean(values > 0),
    }, local


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo_root))
    from qc.publication_style import (  # pylint: disable=import-outside-toplevel
        FIGURE_WIDTH_IN,
        PALETTE,
        save_publication_figure,
        set_publication_style,
        style_axes,
    )

    set_publication_style()
    rng = np.random.default_rng(RANDOM_SEED)
    key = dk_key(args.template_root)
    name_by_label = key.drop_duplicates("dk_label").set_index("dk_label")["dk_region"]
    label_by_name = name_by_label.reset_index().set_index("dk_region")["dk_label"]
    labels = [int(label_by_name.loc[name]) for name in EXTREME_REGIONS]

    subbin = pd.read_parquet(args.region_subbin_table)
    hemi_subject = (
        subbin.groupby(
            ["subject_id", "hemisphere", "anchor_dk_label",
             "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            within_dissimilarity=("within_median_dissimilarity", "median"),
            across_dissimilarity=("across_median_dissimilarity", "median"),
            delta_d=("across_minus_within", "median"),
            within_profile_mae=("within_median_profile_mae", "median"),
            across_profile_mae=("across_median_profile_mae", "median"),
            within_pair_count=("within_n_pairs", "sum"),
            across_pair_count=("across_n_pairs", "sum"),
            distance_difference_median_mm=("distance_difference_mm", "median"),
            absolute_distance_error_mm=("distance_difference_mm", lambda x: np.median(np.abs(x))),
            matched_subbin_count=("distance_subbin_index", "nunique"),
        )
    )
    hemi_subject["eligible"] = (
        hemi_subject["within_pair_count"].ge(MIN_PAIRS)
        & hemi_subject["across_pair_count"].ge(MIN_PAIRS)
    )
    hemi_subject["dk_region"] = hemi_subject["anchor_dk_label"].map(name_by_label)
    hemi_subject.to_parquet(
        args.output_dir / "figure4D_extreme_region_hemisphere_subject_bins.parquet",
        index=False,
    )

    bilateral = pd.read_parquet(args.bilateral_subject_table).rename(
        columns={"meets_pair_threshold": "eligible"}
    )
    group_rows = []
    local_rows = []
    subject_local_parts = []
    for region in EXTREME_REGIONS:
        region_hemi = hemi_subject.loc[hemi_subject["dk_region"].eq(region)]
        region_bilateral = bilateral.loc[bilateral["dk_region"].eq(region)]
        for scope in ("bilateral", "lh", "rh"):
            frame = region_bilateral if scope == "bilateral" else region_hemi.loc[
                region_hemi["hemisphere"].eq(scope)
            ]
            eligible = frame.loc[frame["eligible"]]
            for (bin_index, distance_bin), bin_frame in eligible.groupby(
                ["distance_bin_index", "distance_bin"]
            ):
                group_rows.append(
                    {
                        "dk_region": region,
                        "scope": scope,
                        "distance_bin_index": bin_index,
                        "distance_bin": distance_bin,
                        "n_subjects": bin_frame["subject_id"].nunique(),
                        "within_median": bin_frame["within_dissimilarity"].median()
                        if "within_dissimilarity" in bin_frame else np.nan,
                        "within_q25": bin_frame["within_dissimilarity"].quantile(.25)
                        if "within_dissimilarity" in bin_frame else np.nan,
                        "within_q75": bin_frame["within_dissimilarity"].quantile(.75)
                        if "within_dissimilarity" in bin_frame else np.nan,
                        "across_median": bin_frame["across_dissimilarity"].median()
                        if "across_dissimilarity" in bin_frame else np.nan,
                        "across_q25": bin_frame["across_dissimilarity"].quantile(.25)
                        if "across_dissimilarity" in bin_frame else np.nan,
                        "across_q75": bin_frame["across_dissimilarity"].quantile(.75)
                        if "across_dissimilarity" in bin_frame else np.nan,
                        "delta_median": bin_frame["delta_d"].median(),
                        "delta_q25": bin_frame["delta_d"].quantile(.25),
                        "delta_q75": bin_frame["delta_d"].quantile(.75),
                        "within_pair_count_total": bin_frame["within_pair_count"].sum(),
                        "across_pair_count_total": bin_frame["across_pair_count"].sum(),
                        "distance_error_median_mm": bin_frame["absolute_distance_error_mm"].median(),
                    }
                )
            summary, subject_local = local_summary(frame, scope, rng)
            summary["dk_region"] = region
            local_rows.append(summary)
            subject_local.insert(1, "dk_region", region)
            subject_local.insert(2, "scope", scope)
            subject_local_parts.append(subject_local)

    group = pd.DataFrame(group_rows)
    local = pd.DataFrame(local_rows)
    subject_local = pd.concat(subject_local_parts, ignore_index=True)
    group.to_csv(args.output_dir / "figure4D_extreme_region_curve_audit.csv", index=False)
    local.to_csv(args.output_dir / "figure4D_extreme_region_local_effects.csv", index=False)
    subject_local.to_csv(
        args.output_dir / "figure4D_extreme_region_subject_local_effects.csv", index=False
    )

    # Across-region comparator composition comes from the fixed geometry table.
    pairs = pd.read_parquet(args.pair_table)
    across_pairs = pairs.loc[
        pairs["pair_type"].eq("across_boundary")
        & pairs["anchor_dk_label"].isin(labels)
    ].copy()
    across_pairs["anchor_region"] = across_pairs["anchor_dk_label"].map(name_by_label)
    across_pairs["neighbor_region"] = across_pairs["neighbor_dk_label"].map(name_by_label)
    composition = (
        across_pairs.groupby(
            ["anchor_region", "hemisphere", "distance_bin_index", "distance_bin",
             "neighbor_dk_label", "neighbor_region"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "pair_count"})
    )
    totals = composition.groupby(
        ["anchor_region", "hemisphere", "distance_bin_index"], as_index=False
    )["pair_count"].sum().rename(columns={"pair_count": "pair_count_bin_total"})
    composition = composition.merge(
        totals, on=["anchor_region", "hemisphere", "distance_bin_index"], how="left"
    )
    composition["pair_fraction"] = (
        composition["pair_count"] / composition["pair_count_bin_total"]
    )
    composition["rank_within_bin"] = composition.groupby(
        ["anchor_region", "hemisphere", "distance_bin_index"]
    )["pair_count"].rank(method="first", ascending=False).astype(int)
    composition.to_csv(
        args.output_dir / "figure4D_extreme_region_across_neighbor_composition.csv",
        index=False,
    )
    across_pairs["distance_range"] = np.where(
        across_pairs["distance_bin_index"].lt(5), "0-10 mm", "10-30 mm"
    )
    composition_range = (
        across_pairs.groupby(
            ["anchor_region", "hemisphere", "distance_range", "neighbor_dk_label", "neighbor_region"],
            as_index=False,
        ).size().rename(columns={"size": "pair_count"})
    )
    range_totals = composition_range.groupby(
        ["anchor_region", "hemisphere", "distance_range"], as_index=False
    )["pair_count"].sum().rename(columns={"pair_count": "pair_count_range_total"})
    composition_range = composition_range.merge(
        range_totals, on=["anchor_region", "hemisphere", "distance_range"]
    )
    composition_range["pair_fraction"] = (
        composition_range["pair_count"] / composition_range["pair_count_range_total"]
    )
    composition_range["rank_within_range"] = composition_range.groupby(
        ["anchor_region", "hemisphere", "distance_range"]
    )["pair_count"].rank(method="first", ascending=False).astype(int)
    composition_range.to_csv(
        args.output_dir / "figure4D_extreme_region_across_neighbor_composition_ranges.csv",
        index=False,
    )

    # Separate within/across curves: columns show bilateral, left, and right.
    fig, axes = plt.subplots(
        len(EXTREME_REGIONS), 3, figsize=(FIGURE_WIDTH_IN, 7.4),
        sharex=True, sharey=False,
        gridspec_kw={"hspace": .38, "wspace": .32},
    )
    scopes = ("bilateral", "lh", "rh")
    for row, region in enumerate(EXTREME_REGIONS):
        for column, scope in enumerate(scopes):
            ax = axes[row, column]
            frame = group.loc[
                group["dk_region"].eq(region) & group["scope"].eq(scope)
            ].sort_values("distance_bin_index")
            x = frame["distance_bin_index"].to_numpy() * 2 + 1
            ax.plot(x, frame["within_median"], color=PALETTE["blue"], label="Within")
            ax.fill_between(x, frame["within_q25"], frame["within_q75"],
                            color=PALETTE["blue"], alpha=.18, linewidth=0)
            ax.plot(x, frame["across_median"], color=PALETTE["orange"], label="Across")
            ax.fill_between(x, frame["across_q25"], frame["across_q75"],
                            color=PALETTE["orange"], alpha=.18, linewidth=0)
            ax.axvline(10, color=PALETTE["reference"], linewidth=.7, linestyle="--")
            style_axes(ax, grid=True)
            if row == 0:
                ax.set_title({"bilateral": "Bilateral", "lh": "Left", "rh": "Right"}[scope])
            if column == 0:
                ax.set_ylabel(f"{region}\nDissimilarity (1 − r)")
            if row == len(EXTREME_REGIONS) - 1:
                ax.set_xlabel("Distance (mm)")
            if row == 0 and column == 0:
                ax.legend(frameon=False, loc="best")
    fig.subplots_adjust(left=.16, right=.99, bottom=.08, top=.97)
    save_publication_figure(fig, args.figure_dir / "Figure4D_extreme_regions_within_across")
    plt.close(fig)

    # Subject distributions show whether each bilateral extreme is broadly supported.
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, 4.8),
                             gridspec_kw={"hspace": .42, "wspace": .28})
    for ax, region in zip(axes.flat, EXTREME_REGIONS):
        values = subject_local.loc[
            subject_local["dk_region"].eq(region) & subject_local["scope"].eq("bilateral"),
            "local_effect_0_10",
        ].to_numpy()
        jitter = rng.uniform(-.10, .10, len(values))
        ax.scatter(jitter, values, s=5, alpha=.28, color=PALETTE["blue"], edgecolors="none")
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        ax.plot([-.18, .18], [median, median], color=PALETTE["reference"], linewidth=1.5)
        ax.vlines(0, q25, q75, color=PALETTE["reference"], linewidth=2.5)
        ax.axhline(0, color=PALETTE["gray"], linewidth=.8, linestyle="--")
        ax.set_title(region)
        ax.set_xlim(-.25, .25)
        ax.set_xticks([])
        ax.set_ylabel("0–10-mm mean Δ(1 − r)")
        style_axes(ax, grid=True)
    fig.subplots_adjust(left=.12, right=.98, bottom=.08, top=.95)
    save_publication_figure(
        fig, args.figure_dir / "Figure4D_extreme_regions_subject_distributions"
    )
    plt.close(fig)

    status = {
        "status": "complete",
        "metric_confirmed": "profile_dissimilarity = 1 - Pearson profile correlation",
        "mae_is_separate": True,
        "extreme_regions": list(EXTREME_REGIONS),
        "minimum_pairs_per_class_subject_region_bin": MIN_PAIRS,
        "minimum_local_bins": MIN_LOCAL_BINS,
        "curve_rows": len(group),
        "composition_rows": len(composition),
        "subject_local_rows": len(subject_local),
    }
    (args.output_dir / "figure4D_extreme_region_audit_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
