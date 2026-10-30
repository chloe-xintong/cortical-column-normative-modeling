#!/usr/bin/env python3
"""Summarize a whole-profile permutation pilot and compare it with PSF-0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    "real_dissimilarity",
    "null_dissimilarity",
    "null_minus_real",
    "real_local_excess_similarity",
    "null_local_excess_similarity",
    "real_minus_null_local_excess",
)


def group_summary(table: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for keys, part in table.groupby(
        ["distance_bin_index", "distance_bin"], observed=True
    ):
        bin_index, bin_label = keys
        row = {
            "method": method,
            "distance_bin_index": int(bin_index),
            "distance_bin": bin_label,
            "n_subjects": int(part.subject_id.nunique()),
        }
        for metric in METRICS:
            values = part[metric]
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q25"] = float(values.quantile(0.25))
            row[f"{metric}_q75"] = float(values.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def auc_table(table: pd.DataFrame, method: str) -> pd.DataFrame:
    near = table.loc[table.distance_bin_index.between(0, 4)].copy()
    counts = near.groupby("subject_id").size()
    if not counts.eq(5).all():
        raise ValueError(f"{method}: every subject must have five 0--10-mm bins")
    output = (
        near.groupby("subject_id", observed=True)
        .real_minus_null_local_excess.agg(
            mean_residual_local_excess_0_10="mean",
            sum_residual="sum",
        )
        .reset_index()
    )
    output["auc_residual_local_excess_0_10_mm"] = 2 * output.pop("sum_residual")
    output.insert(1, "method", method)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--number-of-subjects", type=int, default=5)
    parser.add_argument("--permutation-root", type=Path, required=True)
    parser.add_argument("--acquisition-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected = pd.read_csv(
        args.manifest, dtype={"subject_id": str}
    ).head(args.number_of_subjects).subject_id.tolist()
    curve_rows = []
    qc_rows = []
    completion_rows = []
    for subject_id in expected:
        subject_dir = args.permutation_root / "subjects" / subject_id
        completion_path = subject_dir / "completion.json"
        if not completion_path.exists():
            raise FileNotFoundError(f"Missing completion file: {completion_path}")
        completion_rows.append(json.loads(completion_path.read_text()))
        curve_rows.append(pd.read_csv(
            subject_dir / "figure4_whole_profile_subject_bins.csv",
            dtype={"subject_id": str},
        ))
        qc_rows.append(pd.read_csv(
            subject_dir / "figure4_whole_profile_permutation_qc.csv",
            dtype={"subject_id": str},
        ))
    realization = pd.concat(curve_rows, ignore_index=True)
    qc = pd.concat(qc_rows, ignore_index=True)
    if not qc.permutation_is_bijection.all():
        raise ValueError("At least one whole-profile permutation is not bijective")
    count_mismatch = (
        (realization.anchor_count != realization.real_anchor_count)
        | (realization.pair_count != realization.real_pair_count)
    )
    if count_mismatch.any():
        raise ValueError("Real and permutation pair counts differ")
    permutation_subject = (
        realization.groupby(
            ["subject_id", "distance_bin_index", "distance_bin"], observed=True
        )
        .agg({
            "real_dissimilarity": "first",
            "null_dissimilarity": "median",
            "null_minus_real": "median",
            "real_local_excess_similarity": "first",
            "null_local_excess_similarity": "median",
            "real_minus_null_local_excess": "median",
            "anchor_count": "first",
            "pair_count": "first",
        })
        .reset_index()
    )

    acquisition = pd.read_csv(
        args.acquisition_summary, dtype={"subject_id": str}
    )
    available = int(acquisition.n_realizations.max())
    acquisition = acquisition.loc[
        acquisition.subject_id.isin(expected)
        & acquisition.psf_fwhm_mm.eq(0)
        & acquisition.n_realizations.eq(available)
    ].copy()
    if set(acquisition.subject_id) != set(expected):
        raise ValueError("Acquisition PSF-0 summary does not contain the same subjects")
    key = ["subject_id", "distance_bin_index"]
    real_check = permutation_subject[key + ["real_dissimilarity"]].merge(
        acquisition[key + ["real_dissimilarity"]],
        on=key, suffixes=("_permutation", "_acquisition"), validate="one_to_one",
    )
    maximum_real_difference = float(np.max(np.abs(
        real_check.real_dissimilarity_permutation
        - real_check.real_dissimilarity_acquisition
    )))
    if maximum_real_difference > 1e-10:
        raise ValueError("Real curves differ between null workflows")

    permutation_group = group_summary(
        permutation_subject, "Whole-profile permutation"
    )
    acquisition_group = group_summary(acquisition, "DWI-grid, PSF 0 mm")
    group = pd.concat([acquisition_group, permutation_group], ignore_index=True)
    permutation_auc = auc_table(
        permutation_subject, "Whole-profile permutation"
    )
    acquisition_auc = auc_table(acquisition, "DWI-grid, PSF 0 mm")
    auc = pd.concat([acquisition_auc, permutation_auc], ignore_index=True)
    auc_group = (
        auc.groupby("method", observed=True)
        .auc_residual_local_excess_0_10_mm.agg(
            n_subjects="size", median="median",
            q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
            positive_fraction=lambda x: float(np.mean(x > 0)),
        ).reset_index()
    )
    paired_auc = acquisition_auc[[
        "subject_id", "auc_residual_local_excess_0_10_mm"
    ]].merge(
        permutation_auc[["subject_id", "auc_residual_local_excess_0_10_mm"]],
        on="subject_id", suffixes=("_acquisition", "_permutation"),
        validate="one_to_one",
    )
    paired_auc["permutation_minus_acquisition"] = (
        paired_auc.auc_residual_local_excess_0_10_mm_permutation
        - paired_auc.auc_residual_local_excess_0_10_mm_acquisition
    )
    paired_bins = permutation_subject[[
        "subject_id", "distance_bin_index", "distance_bin",
        "real_local_excess_similarity", "null_local_excess_similarity",
        "real_minus_null_local_excess",
    ]].merge(
        acquisition[[
            "subject_id", "distance_bin_index", "distance_bin",
            "real_local_excess_similarity", "null_local_excess_similarity",
            "real_minus_null_local_excess",
        ]],
        on=["subject_id", "distance_bin_index", "distance_bin"],
        suffixes=("_permutation", "_acquisition"), validate="one_to_one",
    )
    paired_bins["permutation_minus_acquisition_residual"] = (
        paired_bins.real_minus_null_local_excess_permutation
        - paired_bins.real_minus_null_local_excess_acquisition
    )
    paired_bins["acquisition_fraction_of_real_local_excess"] = (
        paired_bins.null_local_excess_similarity_acquisition
        / paired_bins.real_local_excess_similarity_acquisition
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    realization.to_csv(
        args.output_dir / "figure4B_whole_profile_realization_bins.csv", index=False
    )
    permutation_subject.to_csv(
        args.output_dir / "figure4B_whole_profile_subject_summary.csv", index=False
    )
    group.to_csv(
        args.output_dir / "figure4B_null_comparison_group_summary.csv", index=False
    )
    auc.to_csv(args.output_dir / "figure4B_null_comparison_auc_subject.csv", index=False)
    auc_group.to_csv(
        args.output_dir / "figure4B_null_comparison_auc_group.csv", index=False
    )
    paired_auc.to_csv(
        args.output_dir / "figure4B_null_comparison_auc_paired.csv", index=False
    )
    paired_bins.to_csv(
        args.output_dir / "figure4B_null_comparison_paired_bins.csv", index=False
    )
    qc.to_csv(args.output_dir / "figure4B_whole_profile_permutation_qc.csv", index=False)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )
    set_publication_style()
    colors = {
        "Real": PALETTE["blue"],
        "DWI-grid, PSF 0 mm": PALETTE["orange"],
        "Whole-profile permutation": "#009E73",
    }
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.55, .85], wspace=.42)
    absolute_ax = fig.add_subplot(grid[0, 0])
    residual_ax = fig.add_subplot(grid[0, 1])
    auc_ax = fig.add_subplot(grid[0, 2])
    real = acquisition_group.sort_values("distance_bin_index")
    x = real.distance_bin_index.to_numpy(dtype=float) * 2 + 1
    absolute_ax.plot(
        x, real.real_dissimilarity_median, color=colors["Real"], label="Real"
    )
    absolute_ax.fill_between(
        x, real.real_dissimilarity_q25, real.real_dissimilarity_q75,
        color=colors["Real"], alpha=.2, linewidth=0,
    )
    for method, table in group.groupby("method", observed=True):
        table = table.sort_values("distance_bin_index")
        absolute_ax.plot(
            x, table.null_dissimilarity_median, color=colors[method],
            linestyle="--" if method.startswith("DWI") else ":", label=method,
        )
        residual_ax.plot(
            x, table.real_minus_null_local_excess_median,
            color=colors[method],
            linestyle="--" if method.startswith("DWI") else ":", label=method,
        )
        residual_ax.fill_between(
            x, table.real_minus_null_local_excess_q25,
            table.real_minus_null_local_excess_q75,
            color=colors[method], alpha=.16, linewidth=0,
        )
    absolute_ax.set(
        xlabel="Geodesic distance (mm)",
        ylabel="Profile dissimilarity (1 − r)",
    )
    absolute_ax.legend(frameon=False, loc="best")
    style_axes(absolute_ax)
    residual_ax.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    residual_ax.set(
        xlabel="Geodesic distance (mm)",
        ylabel="Real − null local excess similarity (Δr)",
    )
    residual_ax.legend(frameon=False, loc="best")
    style_axes(residual_ax)

    methods = ["DWI-grid, PSF 0 mm", "Whole-profile permutation"]
    rng = np.random.default_rng(20261021)
    for position, method in enumerate(methods):
        values = auc.loc[
            auc.method.eq(method), "auc_residual_local_excess_0_10_mm"
        ].to_numpy(dtype=float)
        auc_ax.scatter(
            position + rng.uniform(-.1, .1, len(values)), values,
            s=10, color=colors[method], alpha=.65, linewidth=0,
        )
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        auc_ax.vlines(position, q25, q75, color=PALETTE["reference"], linewidth=2.2)
        auc_ax.hlines(
            median, position - .18, position + .18,
            color=PALETTE["reference"], linewidth=1.2,
        )
    auc_ax.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    auc_ax.set(
        ylabel="Residual local-excess AUC\n(Δr × mm; 0–10 mm)",
        xticks=[0, 1], xticklabels=["DWI grid\nPSF 0", "Whole-profile\npermutation"],
    )
    style_axes(auc_ax)
    save_publication_figure(
        fig, args.output_dir / "Figure4B_null_comparison_pilot5"
    )
    plt.close(fig)

    nearest = paired_bins.loc[paired_bins.distance_bin_index.eq(0)]
    report = {
        "status": "preliminary_pilot_complete",
        "subjects": expected,
        "number_of_subjects": len(expected),
        "whole_profile_realizations": int(realization.replicate.nunique()),
        "acquisition_psf0_realizations": available,
        "permutation_scope": "within_hemisphere_valid_vertices",
        "all_permutations_bijective": bool(qc.permutation_is_bijection.all()),
        "count_mismatch_rows": int(count_mismatch.sum()),
        "maximum_real_curve_difference_between_workflows": maximum_real_difference,
        "median_paired_auc_difference_permutation_minus_acquisition": float(
            paired_auc.permutation_minus_acquisition.median()
        ),
        "median_0_2_mm_residual_difference_permutation_minus_acquisition": float(
            nearest.permutation_minus_acquisition_residual.median()
        ),
        "median_0_2_mm_acquisition_fraction_of_real_local_excess": float(
            nearest.acquisition_fraction_of_real_local_excess.median()
        ),
    }
    (args.output_dir / "figure4B_null_comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(auc_group.to_string(index=False))


if __name__ == "__main__":
    main()
