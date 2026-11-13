#!/usr/bin/env python3
"""Summarize bilateral within-DK-region Figure 4A2 results.

Description
-----------
The analysis uses only fixed Figure 4A pairs whose anchor and neighbor share
the same DK label.  Pair metrics are reduced to anchor x distance-bin medians,
then homologous hemispheres are combined inside each subject, and subjects are
summarized with the group median and IQR.  Pair counts never weight a metric.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--results-root`` command-line input
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
python analysis_results/figure4/scripts/summarize_figure4a2_within_regions.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


MIN_SUBJECT_FRACTION = 0.80
MIN_VALID_PAIRS_PER_SUBJECT_CELL = 20
N_DISTANCE_BINS = 15
DISSIMILARITY_VMAX = 0.35

REGION_DISPLAY_NAMES = {
    "bankssts": "banks STS",
    "caudalanteriorcingulate": "caudal anterior cingulate",
    "caudalmiddlefrontal": "caudal middle frontal",
    "frontalpole": "frontal pole",
    "inferiorparietal": "inferior parietal",
    "inferiortemporal": "inferior temporal",
    "isthmuscingulate": "isthmus cingulate",
    "lateraloccipital": "lateral occipital",
    "lateralorbitofrontal": "lateral orbitofrontal",
    "medialorbitofrontal": "medial orbitofrontal",
    "middletemporal": "middle temporal",
    "parahippocampal": "parahippocampal",
    "parsopercularis": "pars opercularis",
    "parsorbitalis": "pars orbitalis",
    "parstriangularis": "pars triangularis",
    "posteriorcingulate": "posterior cingulate",
    "rostralanteriorcingulate": "rostral anterior cingulate",
    "rostralmiddlefrontal": "rostral middle frontal",
    "superiorfrontal": "superior frontal",
    "superiorparietal": "superior parietal",
    "superiortemporal": "superior temporal",
    "temporalpole": "temporal pole",
    "transversetemporal": "transverse temporal",
}


def display_region(region: str, multiline: bool = False) -> str:
    label = REGION_DISPLAY_NAMES.get(region, region)
    return textwrap.fill(label, width=18) if multiline else label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def dk_region_key(template_root: Path) -> pd.DataFrame:
    rows = []
    for hemisphere in ("lh", "rh"):
        _, _, names = fsio.read_annot(
            str(template_root / "label" / f"{hemisphere}.aparc.annot")
        )
        for label, raw_name in enumerate(names):
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            if name.lower().replace("-", "_") in {
                "unknown", "corpuscallosum", "medial_wall", "medialwall"
            }:
                continue
            rows.append(
                {"hemisphere": hemisphere, "anchor_dk_label": label,
                 "dk_region": name}
            )
    key = pd.DataFrame(rows)
    bilateral = key.groupby(["anchor_dk_label", "dk_region"]).size()
    if len(key) != 68 or len(bilateral) != 34 or not bilateral.eq(2).all():
        raise ValueError("Expected 34 homologous bilateral DK regions")
    return key[["anchor_dk_label", "dk_region"]].drop_duplicates()


def subject_tables(results_root: Path, subject_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    tables = []
    missing = []
    filename = "figure4_relative_mae_within_region_subject_bins.csv"
    for subject_id in subject_ids:
        path = results_root / "subjects" / subject_id / filename
        if path.is_file():
            tables.append(pd.read_csv(path, dtype={"subject_id": str}))
        else:
            missing.append(subject_id)
    if not tables:
        raise RuntimeError(f"No {filename} files found")
    return pd.concat(tables, ignore_index=True), missing


def fixed_pair_counts(pair_table: Path, key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(
        pair_table,
        columns=[
            "hemisphere", "anchor_vertex", "neighbor_vertex", "anchor_dk_label",
            "neighbor_dk_label", "distance_bin_index", "distance_bin", "pair_type",
        ],
    )
    within = pairs.loc[
        pairs["pair_type"].eq("within_region")
        & pairs["anchor_dk_label"].eq(pairs["neighbor_dk_label"])
    ].copy()
    within["hemisphere_anchor"] = (
        within["hemisphere"].astype(str) + ":"
        + within["anchor_vertex"].astype(str)
    )
    low = np.minimum(within["anchor_vertex"], within["neighbor_vertex"])
    high = np.maximum(within["anchor_vertex"], within["neighbor_vertex"])
    within["undirected_pair"] = (
        within["hemisphere"].astype(str) + ":" + low.astype(str) + ":" + high.astype(str)
    )
    counts = (
        within.groupby(
            ["anchor_dk_label", "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            fixed_directed_pair_count=("neighbor_vertex", "size"),
            fixed_unique_undirected_pair_count=("undirected_pair", "nunique"),
            fixed_anchor_count=("hemisphere_anchor", "nunique"),
            fixed_pair_count_lh=("hemisphere", lambda x: int(x.eq("lh").sum())),
            fixed_pair_count_rh=("hemisphere", lambda x: int(x.eq("rh").sum())),
        )
        .merge(key, on="anchor_dk_label", how="left", validate="many_to_one")
    )
    inventory = (
        counts.groupby(["anchor_dk_label", "dk_region"], as_index=False)
        .agg(
            fixed_directed_pairs_0_30=("fixed_directed_pair_count", "sum"),
            fixed_unique_undirected_pairs_0_30=(
                "fixed_unique_undirected_pair_count", "sum"
            ),
            fixed_directed_pairs_0_10=(
                "fixed_directed_pair_count",
                lambda x: int(x[counts.loc[x.index, "distance_bin_index"].lt(5)].sum()),
            ),
            distance_bins_with_fixed_pairs=("distance_bin_index", "nunique"),
            maximum_supported_bin_index=("distance_bin_index", "max"),
        )
    )
    inventory["maximum_supported_distance_mm"] = (
        2 * (inventory["maximum_supported_bin_index"] + 1)
    )
    return counts, inventory


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo_root))
    from qc.publication_style import (  # pylint: disable=import-outside-toplevel
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style,
    )

    set_publication_style()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    subject_ids = manifest["subject_id"].drop_duplicates().tolist()
    expected_subjects = len(subject_ids)
    minimum_subjects = int(np.ceil(MIN_SUBJECT_FRACTION * expected_subjects))
    key = dk_region_key(args.template_root)
    subject, missing = subject_tables(args.results_root, subject_ids)
    required = {
        "profile_dissimilarity", "profile_mae", "symmetric_relative_mae_pct",
        "pair_count", "anchor_count", "hemispheres_contributing",
    }
    absent = required.difference(subject.columns)
    if absent:
        raise ValueError(f"Regional subject tables lack columns: {sorted(absent)}")
    subject = subject.merge(
        key, on="anchor_dk_label", how="left", validate="many_to_one"
    )
    if subject["dk_region"].isna().any():
        raise ValueError("Unmapped DK labels in subject tables")
    subject["meets_pair_threshold"] = subject["pair_count"].ge(
        MIN_VALID_PAIRS_PER_SUBJECT_CELL
    )
    subject.to_parquet(
        args.output_dir / "figure4A2_within_region_subject_bins.parquet", index=False
    )

    eligible = subject.loc[subject["meets_pair_threshold"]]
    group = (
        eligible.groupby(
            ["anchor_dk_label", "dk_region", "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            profile_dissimilarity_median=("profile_dissimilarity", "median"),
            profile_dissimilarity_q25=("profile_dissimilarity", lambda x: x.quantile(.25)),
            profile_dissimilarity_q75=("profile_dissimilarity", lambda x: x.quantile(.75)),
            profile_mae_median=("profile_mae", "median"),
            profile_mae_q25=("profile_mae", lambda x: x.quantile(.25)),
            profile_mae_q75=("profile_mae", lambda x: x.quantile(.75)),
            symmetric_relative_mae_pct_median=("symmetric_relative_mae_pct", "median"),
            symmetric_relative_mae_pct_q25=(
                "symmetric_relative_mae_pct", lambda x: x.quantile(.25)
            ),
            symmetric_relative_mae_pct_q75=(
                "symmetric_relative_mae_pct", lambda x: x.quantile(.75)
            ),
            valid_pair_count_total=("pair_count", "sum"),
            valid_pair_count_median_subject=("pair_count", "median"),
            anchor_count_total=("anchor_count", "sum"),
            anchor_count_median_subject=("anchor_count", "median"),
        )
    )
    bins = pd.DataFrame({
        "distance_bin_index": np.arange(N_DISTANCE_BINS),
        "distance_bin": [f"{start}-{start + 2}" for start in range(0, 30, 2)],
    })
    skeleton = key.merge(bins, how="cross")
    group = skeleton.merge(
        group,
        on=["anchor_dk_label", "dk_region", "distance_bin_index", "distance_bin"],
        how="left",
    )
    count_columns = [
        "n_subjects", "valid_pair_count_total", "anchor_count_total",
    ]
    for column in count_columns:
        group[column] = group[column].fillna(0).astype(np.int64)
    group["formal_coverage_pass"] = group["n_subjects"].ge(minimum_subjects)

    fixed_counts, inventory = fixed_pair_counts(args.pair_table, key)
    group = group.merge(
        fixed_counts.drop(columns="dk_region"),
        on=["anchor_dk_label", "distance_bin_index", "distance_bin"],
        how="left",
        validate="one_to_one",
    )
    fixed_count_columns = [
        "fixed_directed_pair_count", "fixed_unique_undirected_pair_count",
        "fixed_anchor_count", "fixed_pair_count_lh", "fixed_pair_count_rh",
    ]
    for column in fixed_count_columns:
        group[column] = group[column].fillna(0).astype(np.int64)
    group.to_csv(
        args.output_dir / "figure4A2_within_region_group_summary.csv", index=False
    )
    group[[
        "anchor_dk_label", "dk_region", "distance_bin_index", "distance_bin",
        "fixed_directed_pair_count", "fixed_unique_undirected_pair_count",
        "fixed_anchor_count", "fixed_pair_count_lh", "fixed_pair_count_rh",
        "n_subjects", "valid_pair_count_total", "valid_pair_count_median_subject",
        "formal_coverage_pass",
    ]].to_csv(
        args.output_dir / "figure4A2_within_region_pair_counts.csv", index=False
    )

    per_subject_total = (
        subject.groupby(["subject_id", "anchor_dk_label"], as_index=False)
        .agg(valid_profile_pairs_0_30=("pair_count", "sum"))
        .groupby("anchor_dk_label", as_index=False)
        .agg(
            valid_profile_pairs_0_30_total=("valid_profile_pairs_0_30", "sum"),
            valid_profile_pairs_0_30_median_subject=("valid_profile_pairs_0_30", "median"),
        )
    )
    inventory = inventory.merge(per_subject_total, on="anchor_dk_label", how="left")
    coverage = (
        group.groupby(["anchor_dk_label", "dk_region"], as_index=False)
        .agg(
            formal_distance_bins=("formal_coverage_pass", "sum"),
            minimum_subjects_across_bins=("n_subjects", "min"),
        )
    )
    inventory = inventory.merge(
        coverage, on=["anchor_dk_label", "dk_region"], how="left"
    ).sort_values("fixed_directed_pairs_0_30", ascending=False)
    inventory.to_csv(
        args.output_dir / "figure4A2_within_region_inventory.csv", index=False
    )

    row_order = key.sort_values("dk_region")["dk_region"].tolist()
    column_order = bins["distance_bin"].tolist()
    metric_specs = [
        ("profile_dissimilarity_median", "Within-region profile\ndissimilarity (1 − r)", "project_blue"),
        ("symmetric_relative_mae_pct_median", "Within-region symmetric\nrelative MAE (%)", "project_blue"),
        ("fixed_directed_pair_count", "Fixed directed within-region\npairs, log10(n)", "project_blue"),
    ]
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 7.2))
    grid = fig.add_gridspec(
        2, 3, height_ratios=[.045, 1], hspace=.13, wspace=.16
    )
    for column, (metric, colorbar_label, cmap_name) in enumerate(metric_specs):
        colorbar_ax = fig.add_subplot(grid[0, column])
        ax = fig.add_subplot(grid[1, column])
        display_metric = metric
        if metric == "fixed_directed_pair_count":
            display_metric = "fixed_pair_log10"
            group[display_metric] = np.log10(
                group["fixed_directed_pair_count"].replace(0, np.nan)
            )
        matrix = group.pivot(
            index="dk_region", columns="distance_bin", values=display_metric
        ).reindex(index=row_order, columns=column_order)
        if metric != "fixed_directed_pair_count":
            coverage_matrix = group.pivot(
                index="dk_region", columns="distance_bin", values="formal_coverage_pass"
            ).reindex(index=row_order, columns=column_order)
            matrix = matrix.where(coverage_matrix)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(PALETTE["masked"])
        image_kwargs = {}
        if metric == "profile_dissimilarity_median":
            image_kwargs = {"vmin": 0.0, "vmax": DISSIMILARITY_VMAX}
        image = ax.imshow(
            np.ma.masked_invalid(matrix.to_numpy(float)), cmap=cmap,
            aspect="auto", interpolation="nearest", **image_kwargs,
        )
        ax.set_xticks(
            np.arange(len(column_order)), labels=column_order, rotation=55, ha="right"
        )
        ax.set_yticks(np.arange(len(row_order)))
        if column == 0:
            ax.set_yticklabels([display_region(region) for region in row_order])
            ax.set_ylabel("Bilateral DK region")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.axvline(4.5, color="#FFFFFF", linewidth=.8, linestyle="--", alpha=.9)
        ax.tick_params(width=.8, length=2, pad=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        colorbar = fig.colorbar(
            image, cax=colorbar_ax, orientation="horizontal",
            extend="max" if metric == "profile_dissimilarity_median" else "neither",
        )
        if metric == "profile_dissimilarity_median":
            ticks = [0.0, 0.1, 0.2, 0.3, DISSIMILARITY_VMAX]
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(["0", "0.1", "0.2", "0.3", "≥0.35"])
        colorbar.ax.xaxis.set_ticks_position("top")
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.set_label(colorbar_label, labelpad=2)
        colorbar.ax.tick_params(length=2, pad=1)
    fig.supxlabel("White-surface geodesic distance (mm)", y=.018)
    fig.subplots_adjust(left=.245, right=.985, bottom=.105, top=.91)
    save_publication_figure(
        fig, args.figure_dir / "Figure4A2_bilateral_within_region_by_distance"
    )
    plt.close(fig)

    # Compact companion heatmap for every region clipped in the main 0--0.35 scale.
    high_regions = (
        group.loc[
            group["formal_coverage_pass"]
            & group["profile_dissimilarity_median"].gt(DISSIMILARITY_VMAX),
            "dk_region",
        ]
        .drop_duplicates()
        .tolist()
    )
    high_order = (
        group.loc[group["dk_region"].isin(high_regions)]
        .groupby("dk_region")["profile_dissimilarity_median"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    high_matrix = group.pivot(
        index="dk_region", columns="distance_bin", values="profile_dissimilarity_median"
    ).reindex(index=high_order, columns=column_order)
    high_coverage = group.pivot(
        index="dk_region", columns="distance_bin", values="formal_coverage_pass"
    ).reindex(index=high_order, columns=column_order)
    high_matrix = high_matrix.where(high_coverage)
    high_cmap = plt.get_cmap("project_blue").copy()
    high_cmap.set_bad(PALETTE["masked"])
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 2.25))
    high_grid = fig.add_gridspec(1, 2, width_ratios=[1, .025], wspace=.08)
    ax = fig.add_subplot(high_grid[0, 0])
    colorbar_ax = fig.add_subplot(high_grid[0, 1])
    high_image = ax.imshow(
        np.ma.masked_invalid(high_matrix.to_numpy(float)), cmap=high_cmap,
        vmin=0.0, vmax=0.90, aspect="auto", interpolation="nearest",
    )
    ax.set_xticks(
        np.arange(len(column_order)), labels=column_order, rotation=45, ha="right"
    )
    ax.set_yticks(
        np.arange(len(high_order)),
        labels=[display_region(region) for region in high_order],
    )
    ax.set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel="Region exceeding 0.35",
    )
    ax.axvline(4.5, color="#FFFFFF", linewidth=.8, linestyle="--", alpha=.9)
    ax.tick_params(width=.8, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    colorbar = fig.colorbar(high_image, cax=colorbar_ax, orientation="vertical")
    colorbar.set_ticks([0.0, 0.2, 0.35, 0.6, 0.9])
    colorbar.set_ticklabels(["0", "0.2", "main limit 0.35", "0.6", "0.9"])
    colorbar.set_label("Full-range profile\ndissimilarity (1 − r)")
    fig.subplots_adjust(left=.25, right=.92, bottom=.30, top=.96)
    save_publication_figure(
        fig, args.figure_dir / "Figure4A2_high_dissimilarity_full_range"
    )
    plt.close(fig)

    # Prespecified, data-driven representative selection: regions are divided
    # into tertiles by their 0--10-mm mean group-median dissimilarity.  The
    # closest regions to the 20th, 50th, and 80th percentiles within each
    # tertile are selected so the examples span, rather than duplicate, each
    # heterogeneity class.
    local_score = (
        group.loc[group["formal_coverage_pass"] & group["distance_bin_index"].lt(5)]
        .groupby("dk_region", as_index=False)
        .agg(heterogeneity_0_10=("profile_dissimilarity_median", "mean"))
        .sort_values(["heterogeneity_0_10", "dk_region"])
        .reset_index(drop=True)
    )
    local_score["heterogeneity_class"] = pd.qcut(
        local_score["heterogeneity_0_10"].rank(method="first"),
        3, labels=["Low", "Intermediate", "High"],
    )
    full_coverage_regions = set(
        group.groupby("dk_region")["formal_coverage_pass"].sum()
        .loc[lambda x: x.eq(N_DISTANCE_BINS)].index
    )
    representative_parts = []
    for class_name in ("Low", "Intermediate", "High"):
        frame = local_score.loc[
            local_score["heterogeneity_class"].astype(str).eq(class_name)
            & local_score["dk_region"].isin(full_coverage_regions)
        ].copy()
        selected_rows = []
        used_regions = set()
        for quantile in (.20, .50, .80):
            target = frame["heterogeneity_0_10"].quantile(quantile)
            candidates = frame.loc[~frame["dk_region"].isin(used_regions)].copy()
            candidates["distance_from_target"] = (
                candidates["heterogeneity_0_10"] - target
            ).abs()
            selected = candidates.sort_values(
                ["distance_from_target", "dk_region"]
            ).iloc[0].copy()
            selected["within_class_quantile_target"] = quantile
            selected_rows.append(selected)
            used_regions.add(selected["dk_region"])
        representative_parts.append(pd.DataFrame(selected_rows))
    representatives = pd.concat(representative_parts, ignore_index=True)
    representatives.to_csv(
        args.output_dir / "figure4A2_representative_regions.csv", index=False
    )

    class_colors = {
        "Low": PALETTE["blue"],
        "Intermediate": PALETTE["gray"],
        "High": PALETTE["orange"],
    }
    fig, axes = plt.subplots(
        3, 3, figsize=(FIGURE_WIDTH_IN, 5.7), sharex=True, sharey=True
    )
    for row_index, class_name in enumerate(("Low", "Intermediate", "High")):
        chosen = representatives.loc[
            representatives["heterogeneity_class"].astype(str).eq(class_name)
        ].sort_values("heterogeneity_0_10")
        for column_index, (_, selected) in enumerate(chosen.iterrows()):
            ax = axes[row_index, column_index]
            frame = group.loc[group["dk_region"].eq(selected["dk_region"])].sort_values(
                "distance_bin_index"
            )
            valid = frame["formal_coverage_pass"]
            x = frame.loc[valid, "distance_bin_index"].to_numpy(float) * 2 + 1
            color = class_colors[class_name]
            ax.plot(
                x, frame.loc[valid, "profile_dissimilarity_median"],
                color=color, marker="o", markersize=2.5,
            )
            ax.fill_between(
                x,
                frame.loc[valid, "profile_dissimilarity_q25"],
                frame.loc[valid, "profile_dissimilarity_q75"],
                color=color, alpha=.2, linewidth=0,
            )
            ax.text(
                .03, .94, display_region(selected["dk_region"], multiline=True),
                transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
            )
            ax.set(xlim=(0, 30), ylim=(0, .95), xticks=np.arange(0, 31, 10))
            ax.grid(color=PALETTE["masked"], linewidth=.5)
            ax.tick_params(width=.8, length=2, pad=1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[row_index, 0].set_ylabel(class_name)
    for ax in axes[-1, :]:
        ax.set_xlabel("Distance (mm)")
    fig.supylabel("Within-region profile dissimilarity (1 − r)", x=.018)
    fig.subplots_adjust(left=.13, right=.985, bottom=.10, top=.98, hspace=.16, wspace=.10)
    save_publication_figure(
        fig, args.figure_dir / "Figure4A2_representative_within_region_curves"
    )
    plt.close(fig)

    # All 34 bilateral regions as small multiples; common axes preserve direct
    # comparison and avoid an unreadable 34-line overlay.
    fig, axes = plt.subplots(
        6, 6, figsize=(FIGURE_WIDTH_IN, 8.0), sharex=True, sharey=True
    )
    for ax, region in zip(axes.flat, row_order):
        frame = group.loc[group["dk_region"].eq(region)].sort_values(
            "distance_bin_index"
        )
        valid = frame["formal_coverage_pass"]
        x = frame.loc[valid, "distance_bin_index"].to_numpy(float) * 2 + 1
        ax.plot(
            x, frame.loc[valid, "profile_dissimilarity_median"],
            color=PALETTE["blue"], marker="o", markersize=1.8, linewidth=1.0,
        )
        ax.fill_between(
            x,
            frame.loc[valid, "profile_dissimilarity_q25"],
            frame.loc[valid, "profile_dissimilarity_q75"],
            color=PALETTE["blue"], alpha=.2, linewidth=0,
        )
        ax.text(
            .03, .94, display_region(region, multiline=True), transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5,
        )
        ax.set(xlim=(0, 30), ylim=(0, .95), xticks=[0, 10, 20, 30])
        ax.grid(color=PALETTE["masked"], linewidth=.4)
        ax.tick_params(width=.7, length=1.5, pad=1, labelsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(row_order):]:
        ax.set_visible(False)
    fig.supxlabel("White-surface geodesic distance (mm)", y=.018)
    fig.supylabel("Within-region profile dissimilarity (1 − r)", x=.018)
    fig.subplots_adjust(left=.09, right=.99, bottom=.07, top=.99, hspace=.18, wspace=.12)
    save_publication_figure(
        fig, args.figure_dir / "Figure4A2_all_within_region_small_multiples"
    )
    plt.close(fig)

    status = {
        "status": "complete" if not missing else "incomplete",
        "subjects_expected": expected_subjects,
        "subjects_completed": int(subject["subject_id"].nunique()),
        "subjects_missing": missing,
        "bilateral_regions": int(key["dk_region"].nunique()),
        "distance_bins": N_DISTANCE_BINS,
        "minimum_subject_count": minimum_subjects,
        "minimum_valid_pairs_per_subject_region_bin": MIN_VALID_PAIRS_PER_SUBJECT_CELL,
        "formal_cells": int(group["formal_coverage_pass"].sum()),
        "total_cells": int(len(group)),
        "fixed_directed_pairs_total": int(group["fixed_directed_pair_count"].sum()),
        "main_dissimilarity_vmax": DISSIMILARITY_VMAX,
        "cells_clipped_in_main_dissimilarity_heatmap": int(
            (
                group["formal_coverage_pass"]
                & group["profile_dissimilarity_median"].gt(DISSIMILARITY_VMAX)
            ).sum()
        ),
        "regions_with_clipped_cells": high_order,
        "aggregation": (
            "within-region pair -> hemisphere x anchor x 2-mm-bin median -> "
            "bilateral subject x DK-region median -> group median/IQR"
        ),
        "pair_count_note": (
            "fixed_directed_pair_count is the number of anchor-neighbor rows in "
            "the reusable fsaverage Figure 4A pair table; valid_pair_count_total "
            "sums retained profile-pair observations across subjects"
        ),
    }
    (args.output_dir / "figure4A2_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
