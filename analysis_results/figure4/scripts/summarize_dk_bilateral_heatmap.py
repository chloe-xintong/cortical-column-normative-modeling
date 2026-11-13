#!/usr/bin/env python3
"""Build the bilateral 34-region Figure 4D heatmap and laterality audit.

Description
-----------
The cached hemisphere x DK-region x 0.5-mm sub-bin summaries are combined
within subject before any group statistic is calculated.  Each hemisphere is
therefore weighted equally when both are available; a single eligible side can
contribute when its homologue is unavailable.  Classification is based only on
the prespecified 0--10-mm subject-level local effect.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--region-subbin-table`` command-line input
- ``--template-root`` command-line input
- ``--output-dir`` command-line input
- ``--figure-dir`` command-line input
- ``--repo-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/summarize_dk_bilateral_heatmap.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


MIN_SUBJECT_FRACTION = 0.80
MIN_PAIRS_PER_CLASS_PER_SUBJECT = 20
MIN_LOCAL_BINS = 3
N_DISTANCE_BINS = 15
N_BOOTSTRAP = 10_000
RANDOM_SEED = 20261021


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-subbin-table", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, default=325)
    return parser.parse_args()


def dk_region_key(template_root: Path) -> pd.DataFrame:
    hemispheres = []
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
            hemispheres.append(
                {"hemisphere": hemisphere, "anchor_dk_label": label, "dk_region": name}
            )
    key = pd.DataFrame(hemispheres)
    counts = key.groupby(["anchor_dk_label", "dk_region"]).size()
    if len(key) != 68 or not counts.eq(2).all():
        raise ValueError("Expected 34 homologous bilateral DK regions")
    return key[["anchor_dk_label", "dk_region"]].drop_duplicates().sort_values(
        "anchor_dk_label"
    )


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    medians = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo_root))
    from qc.publication_style import (  # pylint: disable=import-outside-toplevel
        FIGURE_WIDTH_IN,
        PALETTE, CMAP_SEQUENTIAL, CMAP_DIVERGING,
        save_publication_figure,
        set_publication_style,
    )

    set_publication_style()
    rng = np.random.default_rng(RANDOM_SEED)
    region_key = dk_region_key(args.template_root)
    minimum_subjects = int(np.ceil(MIN_SUBJECT_FRACTION * args.expected_subjects))
    subbin = pd.read_parquet(args.region_subbin_table)

    # Laterality audit is based on hemisphere-specific subject-level local effects.
    hemi_bin = (
        subbin.groupby(
            ["subject_id", "hemisphere", "anchor_dk_label", "distance_bin_index"],
            as_index=False,
        )
        .agg(
            delta_d=("across_minus_within", "median"),
            within_pair_count=("within_n_pairs", "sum"),
            across_pair_count=("across_n_pairs", "sum"),
        )
    )
    hemi_bin["eligible"] = (
        hemi_bin["within_pair_count"].ge(MIN_PAIRS_PER_CLASS_PER_SUBJECT)
        & hemi_bin["across_pair_count"].ge(MIN_PAIRS_PER_CLASS_PER_SUBJECT)
    )
    hemi_local = (
        hemi_bin.loc[hemi_bin["eligible"] & hemi_bin["distance_bin_index"].lt(5)]
        .groupby(["subject_id", "hemisphere", "anchor_dk_label"], as_index=False)
        .agg(local_effect_0_10=("delta_d", "mean"), valid_local_bins=("delta_d", "size"))
    )
    hemi_local = hemi_local.loc[hemi_local["valid_local_bins"].ge(MIN_LOCAL_BINS)]
    hemi_group = (
        hemi_local.groupby(["hemisphere", "anchor_dk_label"], as_index=False)
        .agg(
            group_median_local_effect=("local_effect_0_10", "median"),
            n_subjects=("subject_id", "nunique"),
        )
    )
    paired_group = hemi_group.pivot(
        index="anchor_dk_label", columns="hemisphere", values="group_median_local_effect"
    ).dropna()
    paired_counts = hemi_group.pivot(
        index="anchor_dk_label", columns="hemisphere", values="n_subjects"
    ).dropna()
    paired_group = paired_group.loc[
        paired_counts.index[(paired_counts[["lh", "rh"]] >= minimum_subjects).all(axis=1)]
    ]
    group_pearson = paired_group["lh"].corr(paired_group["rh"], method="pearson")
    group_spearman = paired_group["lh"].corr(paired_group["rh"], method="spearman")
    group_sign_agreement = np.mean(
        np.sign(paired_group["lh"]) == np.sign(paired_group["rh"])
    )

    subject_wide = hemi_local.pivot_table(
        index=["subject_id", "anchor_dk_label"], columns="hemisphere",
        values="local_effect_0_10",
    ).dropna().reset_index()
    subject_agreement_rows = []
    for subject_id, frame in subject_wide.groupby("subject_id"):
        if len(frame) < 20:
            continue
        subject_agreement_rows.append(
            {
                "subject_id": subject_id,
                "n_homologous_regions": len(frame),
                "pearson_r": frame["lh"].corr(frame["rh"], method="pearson"),
                "spearman_rho": frame["lh"].corr(frame["rh"], method="spearman"),
            }
        )
    subject_agreement = pd.DataFrame(subject_agreement_rows)
    subject_agreement.to_csv(
        args.output_dir / "figure4D_hemisphere_consistency_subjects.csv", index=False
    )

    # Combine hemispheres inside each subject and matched 0.5-mm sub-bin.
    # Medians give equal side weighting and avoid pair-count dominance.
    bilateral_subbin = (
        subbin.groupby(
            ["subject_id", "anchor_dk_label", "distance_bin_index", "distance_bin",
             "distance_subbin_index"],
            as_index=False,
        )
        .agg(
            within_dissimilarity=("within_median_dissimilarity", "median"),
            across_dissimilarity=("across_median_dissimilarity", "median"),
            within_profile_mae=("within_median_profile_mae", "median"),
            across_profile_mae=("across_median_profile_mae", "median"),
            within_distance_mm=("within_median_geodesic_distance_mm", "median"),
            across_distance_mm=("across_median_geodesic_distance_mm", "median"),
            within_pair_count=("within_n_pairs", "sum"),
            across_pair_count=("across_n_pairs", "sum"),
            hemispheres_contributing=("hemisphere", "nunique"),
        )
    )
    bilateral_subbin["delta_d"] = (
        bilateral_subbin["across_dissimilarity"]
        - bilateral_subbin["within_dissimilarity"]
    )
    bilateral_subbin["delta_mae"] = (
        bilateral_subbin["across_profile_mae"] - bilateral_subbin["within_profile_mae"]
    )
    bilateral_subbin["distance_difference_mm"] = (
        bilateral_subbin["across_distance_mm"] - bilateral_subbin["within_distance_mm"]
    )

    subject_region = (
        bilateral_subbin.groupby(
            ["subject_id", "anchor_dk_label", "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            within_dissimilarity=("within_dissimilarity", "median"),
            across_dissimilarity=("across_dissimilarity", "median"),
            within_profile_mae=("within_profile_mae", "median"),
            across_profile_mae=("across_profile_mae", "median"),
            delta_d=("delta_d", "median"),
            delta_mae=("delta_mae", "median"),
            matched_subbin_count=("distance_subbin_index", "nunique"),
            within_pair_count=("within_pair_count", "sum"),
            across_pair_count=("across_pair_count", "sum"),
            absolute_distance_error_mm=("distance_difference_mm", lambda x: np.median(np.abs(x))),
            hemispheres_contributing=("hemispheres_contributing", "max"),
        )
    )
    subject_region["meets_pair_threshold"] = (
        subject_region["within_pair_count"].ge(MIN_PAIRS_PER_CLASS_PER_SUBJECT)
        & subject_region["across_pair_count"].ge(MIN_PAIRS_PER_CLASS_PER_SUBJECT)
    )
    subject_region = subject_region.merge(region_key, on="anchor_dk_label", how="left")
    subject_region.to_parquet(
        args.output_dir / "figure4D_bilateral_region_distance_subject_effects.parquet",
        index=False,
    )

    eligible = subject_region.loc[subject_region["meets_pair_threshold"]]
    group_effect = (
        eligible.groupby(
            ["anchor_dk_label", "dk_region", "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            eligible_subject_count=("subject_id", "nunique"),
            delta_d_median=("delta_d", "median"),
            delta_d_q25=("delta_d", lambda x: x.quantile(.25)),
            delta_d_q75=("delta_d", lambda x: x.quantile(.75)),
            within_pair_count_total=("within_pair_count", "sum"),
            across_pair_count_total=("across_pair_count", "sum"),
            distance_error_median_mm=("absolute_distance_error_mm", "median"),
        )
    )
    bins = pd.DataFrame(
        {"distance_bin_index": np.arange(N_DISTANCE_BINS),
         "distance_bin": [f"{x}-{x + 2}" for x in range(0, 30, 2)]}
    )
    skeleton = region_key.merge(bins, how="cross")
    group_effect = skeleton.merge(
        group_effect,
        on=["anchor_dk_label", "dk_region", "distance_bin_index", "distance_bin"],
        how="left",
    )
    for column in ("eligible_subject_count", "within_pair_count_total", "across_pair_count_total"):
        group_effect[column] = group_effect[column].fillna(0).astype(int)
    group_effect["formal_coverage_pass"] = group_effect["eligible_subject_count"].ge(
        minimum_subjects
    )
    group_effect["delta_d_display"] = group_effect["delta_d_median"].where(
        group_effect["formal_coverage_pass"]
    )
    group_effect.to_csv(
        args.output_dir / "figure4D_bilateral_region_distance_effect_coverage.csv",
        index=False,
    )

    local_subject = (
        eligible.loc[eligible["distance_bin_index"].lt(5)]
        .groupby(["subject_id", "anchor_dk_label", "dk_region"], as_index=False)
        .agg(local_effect_0_10=("delta_d", "mean"), valid_local_bins=("delta_d", "size"))
    )
    local_subject = local_subject.loc[local_subject["valid_local_bins"].ge(MIN_LOCAL_BINS)]
    local_effect_rows = []
    for (label, region), frame in local_subject.groupby(["anchor_dk_label", "dk_region"]):
        values = frame["local_effect_0_10"].to_numpy(float)
        low, high = bootstrap_median_ci(values, rng)
        local_effect_rows.append(
            {
                "anchor_dk_label": label,
                "dk_region": region,
                "n_subjects": len(values),
                "local_effect_median": np.median(values),
                "local_effect_q25": np.quantile(values, .25),
                "local_effect_q75": np.quantile(values, .75),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "formal_coverage_pass": len(values) >= minimum_subjects,
            }
        )
    local_effect = pd.DataFrame(local_effect_rows).sort_values(
        "local_effect_median", ascending=False
    ).reset_index(drop=True)
    local_effect["display_order"] = np.arange(len(local_effect))
    local_effect.to_csv(
        args.output_dir / "figure4D_bilateral_region_local_effect_summary.csv",
        index=False,
    )

    row_order = local_effect["dk_region"].tolist()
    column_order = bins["distance_bin"].tolist()
    effect_matrix = group_effect.pivot(
        index="dk_region", columns="distance_bin", values="delta_d_display"
    ).reindex(index=row_order, columns=column_order)
    values = effect_matrix.to_numpy(float)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(np.quantile(finite, .98)) if finite.size else .01
    (args.output_dir / 'heatmap_color_caption.txt').write_text('Signed difference: delta D = D_across - D_within. Red indicates greater across-region dissimilarity; blue indicates greater within-region dissimilarity; neutral white indicates zero. Symmetric color limits use the 98th percentile of absolute finite effects across the complete displayed matrix; only colors are clipped. Missing or insufficient-coverage cells are gray. Coverage uses a separate light-to-dark blue sequential scale from 0 to 100 percent.\n')
    cmap = CMAP_DIVERGING.copy()
    cmap.set_bad(PALETTE["masked"])

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 6.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, .34, .035], wspace=.10)
    ax = fig.add_subplot(grid[0, 0])
    effect_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    colorbar_ax = fig.add_subplot(grid[0, 2])

    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto", interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(column_order)), labels=column_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_order)), labels=row_order)
    ax.set_xlabel("White-surface geodesic distance (mm)")
    ax.set_ylabel("Bilateral DK anchor region")
    ax.axvline(4.5, color=PALETTE["reference"], linewidth=1.0, linestyle="--")
    ax.tick_params(width=.8, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    y = np.arange(len(local_effect))
    point_colors = np.where(
        local_effect["local_effect_median"].ge(0), PALETTE["orange"], PALETTE["blue"]
    )
    effect_ax.axvline(0, color=PALETTE["reference"], linewidth=.8)
    effect_ax.axvline(-.01, color=PALETTE["gray"], linewidth=.7, linestyle=":")
    effect_ax.axvline(.01, color=PALETTE["gray"], linewidth=.7, linestyle=":")
    effect_ax.hlines(
        y, local_effect["bootstrap_ci95_low"], local_effect["bootstrap_ci95_high"],
        color=PALETTE["gray"], linewidth=.8,
    )
    effect_ax.scatter(
        local_effect["local_effect_median"], y, s=9, color=point_colors, zorder=3,
    )
    effect_ax.set_xlabel("0–10-mm mean Δ(1 − r)")
    effect_ax.tick_params(axis="y", left=False, labelleft=False)
    effect_ax.grid(axis="x", color=PALETTE["masked"], linewidth=.5)
    for side in ("top", "right", "left"):
        effect_ax.spines[side].set_visible(False)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Region-level Δ profile dissimilarity, Δ(1 − r)")
    fig.subplots_adjust(left=.245, right=.92, bottom=.10, top=.98)
    save_publication_figure(
        fig, args.figure_dir / "Figure4D_bilateral_region_distance_effect_heatmap"
    )
    plt.close(fig)

    consistency = {
        "group_level_homologous_regions": int(len(paired_group)),
        "group_level_pearson_r": float(group_pearson),
        "group_level_spearman_rho": float(group_spearman),
        "group_level_same_sign_fraction": float(group_sign_agreement),
        "subject_level_n": int(len(subject_agreement)),
        "subject_level_spearman_median": float(subject_agreement["spearman_rho"].median()),
        "subject_level_spearman_q25": float(subject_agreement["spearman_rho"].quantile(.25)),
        "subject_level_spearman_q75": float(subject_agreement["spearman_rho"].quantile(.75)),
        "subject_level_positive_spearman_fraction": float(
            subject_agreement["spearman_rho"].gt(0).mean()
        ),
    }
    status = {
        "status": "complete",
        "bilateral_regions": len(region_key),
        "distance_bins": N_DISTANCE_BINS,
        "expected_subjects": args.expected_subjects,
        "minimum_subject_count": minimum_subjects,
        "minimum_pairs_per_class_per_subject_region_bin": MIN_PAIRS_PER_CLASS_PER_SUBJECT,
        "minimum_0_10_bins_for_summary": MIN_LOCAL_BINS,
        "effect_ci_bootstrap_iterations": N_BOOTSTRAP,
        "region_ordering": "descending subject-level median 0-10-mm mean delta",
        "local_effect_median_min": float(local_effect["local_effect_median"].min()),
        "local_effect_median_max": float(local_effect["local_effect_median"].max()),
        "formal_cells": int(group_effect["formal_coverage_pass"].sum()),
        "total_cells": int(len(group_effect)),
        "effect_color_limit": limit,
        "hemisphere_consistency": consistency,
        "bilateral_aggregation": (
            "subject x region x 0.5-mm sub-bin median across available hemispheres; "
            "then subject x region x 2-mm-bin median"
        ),
    }
    (args.output_dir / "figure4D_bilateral_region_heatmap_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
