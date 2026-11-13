#!/usr/bin/env python3
"""Summarize real/null column correlations with subject mean profiles.

Description
-----------
Summarize real/null column correlations with subject mean profiles.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--condition`` command-line input
- ``--manifest`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/summarize_common_laminar_shape.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_condition(value: str) -> tuple[float, Path]:
    fwhm, root = value.split("=", 1)
    return float(fwhm), Path(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = pd.read_csv(args.manifest, dtype={"subject_id": str}).subject_id
    summaries = []
    histograms = []
    for condition in args.condition:
        fwhm, root = parse_condition(condition)
        for subject_id in expected:
            subject_dir = root / "subjects" / subject_id
            summary_path = subject_dir / "figure4_profile_mean_correlation_summary.csv"
            histogram_path = subject_dir / "figure4_profile_mean_correlation_histogram.csv"
            if not summary_path.exists() or not histogram_path.exists():
                continue
            summary = pd.read_csv(summary_path, keep_default_na=False)
            histogram = pd.read_csv(histogram_path, keep_default_na=False)
            summaries.append(summary.assign(psf_fwhm_mm=fwhm))
            histograms.append(histogram.assign(psf_fwhm_mm=fwhm))
    if not summaries:
        raise RuntimeError("No common-laminar-shape results found")
    summary = pd.concat(summaries, ignore_index=True)
    histogram = pd.concat(histograms, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_summary = (
        summary.loc[
            summary.data_type.eq("real") & summary.psf_fwhm_mm.eq(0)
        ]
        .assign(condition="Real")
    )
    null_summary = (
        summary.loc[summary.data_type.eq("null")]
        .groupby(["subject_id", "psf_fwhm_mm", "depth_set"], observed=True)
        .median_correlation.median()
        .reset_index()
    )
    null_summary["condition"] = null_summary.psf_fwhm_mm.map(
        lambda value: f"FWHM {value:g} mm"
    )
    subject_summary = pd.concat([
        real_summary[["subject_id", "depth_set", "condition", "median_correlation"]],
        null_summary[["subject_id", "depth_set", "condition", "median_correlation"]],
    ], ignore_index=True)
    subject_summary.to_csv(
        args.output_dir / "figure4_common_laminar_shape_subject_summary.csv",
        index=False,
    )

    group_rows = []
    rng = np.random.default_rng(20261020)
    for keys, table in subject_summary.groupby(
        ["condition", "depth_set"], observed=True
    ):
        condition, depth_set = keys
        values = table.median_correlation.to_numpy(dtype=float)
        indices = rng.integers(0, len(values), size=(10_000, len(values)))
        bootstrap = np.median(values[indices], axis=1)
        group_rows.append({
            "condition": condition,
            "depth_set": depth_set,
            "n_subjects": table.subject_id.nunique(),
            "median": np.median(values),
            "q25": np.quantile(values, .25),
            "q75": np.quantile(values, .75),
            "bootstrap_ci_low": np.quantile(bootstrap, .025),
            "bootstrap_ci_high": np.quantile(bootstrap, .975),
        })
    pd.DataFrame(group_rows).to_csv(
        args.output_dir / "figure4_common_laminar_shape_group_summary.csv",
        index=False,
    )

    real_histogram = (
        histogram.loc[
            histogram.data_type.eq("real") & histogram.psf_fwhm_mm.eq(0)
        ]
        .assign(condition="Real")
    )
    null_histogram = (
        histogram.loc[histogram.data_type.eq("null")]
        .groupby([
            "subject_id", "psf_fwhm_mm", "depth_set",
            "correlation_bin_index", "correlation_bin_center",
        ], observed=True).fraction.mean().reset_index()
    )
    null_histogram["condition"] = null_histogram.psf_fwhm_mm.map(
        lambda value: f"FWHM {value:g} mm"
    )
    subject_histogram = pd.concat([
        real_histogram[[
            "subject_id", "depth_set", "condition",
            "correlation_bin_index", "correlation_bin_center", "fraction",
        ]],
        null_histogram[[
            "subject_id", "depth_set", "condition",
            "correlation_bin_index", "correlation_bin_center", "fraction",
        ]],
    ], ignore_index=True)
    density = (
        subject_histogram.groupby([
            "condition", "depth_set", "correlation_bin_index",
            "correlation_bin_center",
        ], observed=True).fraction.agg(
            median="median",
            q25=lambda values: values.quantile(.25),
            q75=lambda values: values.quantile(.75),
        ).reset_index()
    )
    density.to_csv(
        args.output_dir / "figure4_common_laminar_shape_density.csv", index=False
    )

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )
    set_publication_style()
    conditions = ["Real", "FWHM 0 mm", "FWHM 1 mm", "FWHM 2 mm", "FWHM 3 mm"]
    null_colors = plt.get_cmap("cividis")(np.linspace(.15, .85, 4))
    colors = {"Real": PALETTE["blue"]}
    colors.update(dict(zip(conditions[1:], null_colors)))
    line_styles = dict(zip(conditions, ["-", "-", "--", "-.", ":"]))
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.25))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=.34)
    density_ax = fig.add_subplot(grid[0, 0])
    summary_ax = fig.add_subplot(grid[0, 1])
    for condition in conditions:
        table = density.loc[
            density.condition.eq(condition)
            & density.depth_set.eq("all_21_depths")
        ].sort_values("correlation_bin_index")
        x = table.correlation_bin_center.to_numpy(dtype=float)
        density_ax.plot(
            x, table["median"], color=colors[condition],
            linestyle=line_styles[condition], linewidth=1.3, label=condition,
        )
        density_ax.fill_between(
            x, table.q25, table.q75,
            color=colors[condition], alpha=.1, linewidth=0,
        )
    density_ax.set(
        xlabel="Column correlation with subject mean profile (r)",
        ylabel="Fraction of valid columns per 0.02 bin",
        xlim=(-1, 1),
    )
    density_ax.legend(frameon=False, ncol=2, loc="upper left")
    style_axes(density_ax)

    jitter_rng = np.random.default_rng(20261020)
    for position, condition in enumerate(conditions):
        for depth_set, offset, marker in (
            ("all_21_depths", -.08, "o"),
            ("inner_19_depths", .08, "s"),
        ):
            values = subject_summary.loc[
                subject_summary.condition.eq(condition)
                & subject_summary.depth_set.eq(depth_set),
                "median_correlation",
            ].to_numpy(dtype=float)
            jitter = jitter_rng.uniform(-.045, .045, size=len(values))
            summary_ax.scatter(
                position + offset + jitter, values, s=7,
                marker=marker, facecolor=colors[condition],
                edgecolor="none", alpha=.45,
            )
            q25, median, q75 = np.quantile(values, [.25, .5, .75])
            summary_ax.vlines(
                position + offset, q25, q75,
                color=PALETTE["reference"], linewidth=1.6,
            )
            summary_ax.hlines(
                median, position + offset - .07, position + offset + .07,
                color=PALETTE["reference"], linewidth=1.0,
            )
    summary_ax.set(
        ylabel="Subject median column-to-mean correlation",
        xticks=np.arange(len(conditions)),
        xticklabels=["Real", "0", "1", "2", "3"],
        xlabel="PSF FWHM (mm; real at left)",
    )
    summary_ax.scatter([], [], marker="o", color=PALETTE["gray"], label="21 depths")
    summary_ax.scatter([], [], marker="s", color=PALETTE["gray"], label="Inner 19")
    summary_ax.legend(frameon=False, loc="lower left")
    style_axes(summary_ax)
    save_publication_figure(
        fig, args.output_dir / "Figure4_common_laminar_shape_supplement"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
