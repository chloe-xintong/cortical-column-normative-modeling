#!/usr/bin/env python3
"""Create the Figure 4D DK-region x distance effect and coverage heatmaps."""

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
N_DISTANCE_BINS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-subbin-table", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, default=325)
    return parser.parse_args()


def dk_regions(template_root: Path) -> pd.DataFrame:
    rows = []
    for hemisphere in ("lh", "rh"):
        _, _, names = fsio.read_annot(
            str(template_root / "label" / f"{hemisphere}.aparc.annot")
        )
        normalized = [
            name.decode() if isinstance(name, bytes) else str(name) for name in names
        ]
        for label, name in enumerate(normalized):
            if name.lower().replace("-", "_") in {
                "unknown",
                "corpuscallosum",
                "medial_wall",
                "medialwall",
            }:
                continue
            rows.append(
                {
                    "hemisphere": hemisphere,
                    "anchor_dk_label": label,
                    "dk_region": name,
                    "region_display": f"{'L' if hemisphere == 'lh' else 'R'} {name}",
                }
            )
    return pd.DataFrame(rows)


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
    )

    set_publication_style()
    region_key = dk_regions(args.template_root)
    if len(region_key) != 68:
        raise ValueError(f"Expected 68 cortical DK regions, found {len(region_key)}")

    subbin = pd.read_parquet(args.region_subbin_table)
    subject_region = (
        subbin.groupby(
            [
                "subject_id",
                "hemisphere",
                "anchor_dk_label",
                "distance_bin_index",
                "distance_bin",
            ],
            as_index=False,
        )
        .agg(
            delta_d=("across_minus_within", "median"),
            delta_mae=("across_minus_within_mae", "median"),
            matched_subbin_count=("distance_subbin_index", "nunique"),
            within_pair_count=("within_n_pairs", "sum"),
            across_pair_count=("across_n_pairs", "sum"),
            within_anchor_subbin_count=("within_n_anchor_subbins", "sum"),
            across_anchor_subbin_count=("across_n_anchor_subbins", "sum"),
            absolute_distance_error_mm=(
                "distance_difference_mm", lambda x: np.median(np.abs(x))
            ),
        )
    )
    subject_region["meets_pair_threshold"] = (
        subject_region["within_pair_count"].ge(MIN_PAIRS_PER_CLASS_PER_SUBJECT)
        & subject_region["across_pair_count"].ge(MIN_PAIRS_PER_CLASS_PER_SUBJECT)
    )

    eligible = subject_region.loc[subject_region["meets_pair_threshold"]].copy()
    effect = (
        eligible.groupby(
            ["hemisphere", "anchor_dk_label", "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            eligible_subject_count=("subject_id", "nunique"),
            delta_d_median=("delta_d", "median"),
            delta_d_q25=("delta_d", lambda x: x.quantile(.25)),
            delta_d_q75=("delta_d", lambda x: x.quantile(.75)),
            delta_mae_median=("delta_mae", "median"),
            delta_mae_q25=("delta_mae", lambda x: x.quantile(.25)),
            delta_mae_q75=("delta_mae", lambda x: x.quantile(.75)),
        )
    )
    coverage = (
        subject_region.groupby(
            ["hemisphere", "anchor_dk_label", "distance_bin_index", "distance_bin"],
            as_index=False,
        )
        .agg(
            any_subject_count=("subject_id", "nunique"),
            within_pair_count_total=("within_pair_count", "sum"),
            across_pair_count_total=("across_pair_count", "sum"),
            within_pair_count_median_subject=("within_pair_count", "median"),
            across_pair_count_median_subject=("across_pair_count", "median"),
            matched_subbin_count_median_subject=("matched_subbin_count", "median"),
            distance_error_median_mm=("absolute_distance_error_mm", "median"),
        )
    )

    bins = pd.DataFrame(
        {
            "distance_bin_index": np.arange(N_DISTANCE_BINS),
            "distance_bin": [f"{lower}-{lower + 2}" for lower in range(0, 30, 2)],
        }
    )
    skeleton = region_key.merge(bins, how="cross")
    keys = ["hemisphere", "anchor_dk_label", "distance_bin_index", "distance_bin"]
    result = skeleton.merge(coverage, on=keys, how="left").merge(
        effect, on=keys, how="left"
    )
    count_columns = [
        "any_subject_count",
        "eligible_subject_count",
        "within_pair_count_total",
        "across_pair_count_total",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    minimum_subjects = int(np.ceil(MIN_SUBJECT_FRACTION * args.expected_subjects))
    result["formal_coverage_pass"] = result["eligible_subject_count"].ge(
        minimum_subjects
    )
    result["delta_d_display"] = result["delta_d_median"].where(
        result["formal_coverage_pass"]
    )

    pairs = pd.read_parquet(
        args.pair_table,
        columns=[
            "hemisphere",
            "anchor_dk_label",
            "pair_type",
            "geodesic_distance_mm",
        ],
    )
    within_pairs = pairs.loc[pairs["pair_type"].eq("within_region")]
    support = (
        within_pairs.groupby(["hemisphere", "anchor_dk_label"], as_index=False)
        .agg(
            fixed_within_pair_count=("geodesic_distance_mm", "size"),
            max_supported_within_distance_mm=("geodesic_distance_mm", "max"),
        )
    )
    near_error = (
        subject_region.loc[subject_region["distance_bin_index"].lt(5)]
        .groupby(["hemisphere", "anchor_dk_label"], as_index=False)
        .agg(
            distance_error_0_10_median_mm=("absolute_distance_error_mm", "median")
        )
    )
    region_summary = (
        result.groupby(
            ["hemisphere", "anchor_dk_label", "dk_region", "region_display"],
            as_index=False,
        )
        .agg(
            valid_distance_bins=("formal_coverage_pass", "sum"),
            minimum_eligible_subjects_across_bins=("eligible_subject_count", "min"),
            maximum_eligible_subjects_across_bins=("eligible_subject_count", "max"),
            within_pair_count_total=("within_pair_count_total", "sum"),
            across_pair_count_total=("across_pair_count_total", "sum"),
        )
        .merge(near_error, on=["hemisphere", "anchor_dk_label"], how="left")
        .merge(support, on=["hemisphere", "anchor_dk_label"], how="left")
    )

    subject_region.to_parquet(
        args.output_dir / "figure4D_region_distance_subject_coverage.parquet",
        index=False,
    )
    result.to_csv(
        args.output_dir / "figure4D_region_distance_effect_coverage.csv", index=False
    )
    region_summary.to_csv(
        args.output_dir / "figure4D_region_coverage_summary.csv", index=False
    )

    row_order = region_key["region_display"].tolist()
    column_order = bins["distance_bin"].tolist()
    effect_matrix = (
        result.pivot(index="region_display", columns="distance_bin", values="delta_d_display")
        .reindex(index=row_order, columns=column_order)
    )
    coverage_matrix = (
        result.pivot(
            index="region_display", columns="distance_bin", values="eligible_subject_count"
        )
        .reindex(index=row_order, columns=column_order)
    )

    effect_values = effect_matrix.to_numpy(dtype=float)
    finite = np.abs(effect_values[np.isfinite(effect_values)])
    limit = float(np.quantile(finite, .98)) if finite.size else .01
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dk_effect", [PALETTE["blue"], "#FFFFFF", PALETTE["orange"]]
    )
    cmap.set_bad(PALETTE["masked"])

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 10.0))
    image = ax.imshow(
        np.ma.masked_invalid(effect_values),
        cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(column_order)), labels=column_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_order)), labels=row_order)
    ax.set_xlabel("White-surface geodesic distance (mm)")
    ax.set_ylabel("DK anchor region")
    ax.axhline(33.5, color=PALETTE["reference"], linewidth=.8)
    ax.tick_params(width=.8, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=.028, pad=.025)
    colorbar.set_label("Region-level Δ profile dissimilarity, Δ(1 − r)")
    fig.text(
        .12,
        .995,
        (
            f"Displayed cells: ≥{minimum_subjects}/{args.expected_subjects} subjects; "
            f"each subject has ≥{MIN_PAIRS_PER_CLASS_PER_SUBJECT} pairs per class. "
            "Gray: insufficient coverage."
        ),
        ha="left",
        va="top",
        fontsize=7,
    )
    fig.subplots_adjust(left=.27, right=.91, bottom=.08, top=.97)
    save_publication_figure(
        fig, args.figure_dir / "Figure4D_region_distance_effect_heatmap"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 10.0))
    coverage_image = ax.imshow(
        coverage_matrix.to_numpy(dtype=float) / args.expected_subjects * 100,
        cmap="viridis",
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(column_order)), labels=column_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_order)), labels=row_order)
    ax.set_xlabel("White-surface geodesic distance (mm)")
    ax.set_ylabel("DK anchor region")
    ax.axhline(33.5, color="white", linewidth=.8)
    ax.tick_params(width=.8, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    colorbar = fig.colorbar(coverage_image, ax=ax, fraction=.028, pad=.025)
    colorbar.set_label("Subjects meeting pair threshold (%)")
    fig.subplots_adjust(left=.27, right=.91, bottom=.08, top=.98)
    save_publication_figure(
        fig, args.figure_dir / "Figure4D_region_distance_coverage_heatmap"
    )
    plt.close(fig)

    status = {
        "status": "complete",
        "regions": len(region_key),
        "distance_bins": N_DISTANCE_BINS,
        "expected_subjects": args.expected_subjects,
        "minimum_subject_fraction": MIN_SUBJECT_FRACTION,
        "minimum_subject_count": minimum_subjects,
        "minimum_pairs_per_class_per_subject_region_bin": (
            MIN_PAIRS_PER_CLASS_PER_SUBJECT
        ),
        "formal_cells": int(result["formal_coverage_pass"].sum()),
        "total_cells": int(len(result)),
        "regions_with_all_bins": int(
            (region_summary["valid_distance_bins"] == N_DISTANCE_BINS).sum()
        ),
        "regions_with_no_formal_bins": int(
            (region_summary["valid_distance_bins"] == 0).sum()
        ),
        "max_within_distance_is_capped_by_pair_table_mm": 30,
        "effect_color_limit": limit,
    }
    (args.output_dir / "figure4D_region_heatmap_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
