#!/usr/bin/env python3
"""Summarize formal ico5 within-DK K=15 Figure 4C correlation lengths.

Description
-----------
Summarize formal ico5 within-DK K=15 Figure 4C correlation lengths.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--results-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/summarize_ico5_formal_c.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

from fit_correlation_lengths import icc_two_way_single


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = pd.read_csv(args.manifest, dtype={"subject_id": str}).subject_id.tolist()
    fits, split_bins, missing = [], [], []
    for subject_id in expected:
        root = args.results_root / "subjects" / subject_id
        fit_path = root / "figure4C_correlation_length_fits.csv"
        split_path = root / "figure4C_fixed_split_subject_bins.csv"
        if fit_path.is_file() and split_path.is_file():
            fits.append(pd.read_csv(fit_path))
            split_bins.append(pd.read_csv(split_path))
        else:
            missing.append(subject_id)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} Figure 4C subjects: {missing[:10]}")
    fits = pd.concat(fits, ignore_index=True)
    split_bins = pd.concat(split_bins, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits.to_csv(args.output_dir / "figure4C_correlation_length_fits.csv", index=False)
    split_bins.to_csv(args.output_dir / "figure4C_fixed_split_subject_bins.csv", index=False)

    fit_qc = (
        fits.groupby("fit_scope", as_index=False)
        .agg(
            n_fits=("subject_id", "size"),
            n_success=("fit_success", "sum"),
            n_acceptable=("fit_acceptable", "sum"),
            n_boundary=("hit_parameter_boundary", "sum"),
            median_r2=("fit_r2", "median"),
            median_residual_rmse=("residual_rmse", "median"),
        )
    )
    fit_qc.to_csv(args.output_dir / "figure4C_fit_qc_summary.csv", index=False)
    full = fits.loc[fits.fit_scope.eq("full") & fits.fit_acceptable].copy()
    full_summary = pd.DataFrame([{
        "n_subjects": full.subject_id.nunique(),
        "lambda_median_mm": full.lambda_mm.median(),
        "lambda_q25_mm": full.lambda_mm.quantile(.25),
        "lambda_q75_mm": full.lambda_mm.quantile(.75),
        "d_infinity_median": full.d_infinity.median(),
        "amplitude_median": full.amplitude.median(),
        "r2_median": full.fit_r2.median(),
    }])
    full_summary.to_csv(args.output_dir / "figure4C_full_fit_summary.csv", index=False)

    halves = fits.loc[
        fits.fit_scope.isin(["fixed_half_1", "fixed_half_2"])
        & fits.fit_acceptable
    ].pivot(index="subject_id", columns="fit_scope", values="lambda_mm")
    paired = halves.dropna(subset=["fixed_half_1", "fixed_half_2"])
    difference = paired.fixed_half_1 - paired.fixed_half_2
    reliability = pd.DataFrame([{
        "n_subjects_total": len(expected),
        "n_paired_acceptable": len(paired),
        "pearson_r": stats.pearsonr(paired.fixed_half_1, paired.fixed_half_2).statistic,
        "spearman_rho": stats.spearmanr(paired.fixed_half_1, paired.fixed_half_2).statistic,
        "icc_two_way_single": icc_two_way_single(
            paired.fixed_half_1.to_numpy(), paired.fixed_half_2.to_numpy()
        ),
        "median_absolute_difference_mm": difference.abs().median(),
        "n_failed_or_unacceptable_subjects": len(expected) - len(paired),
    }])
    reliability.to_csv(args.output_dir / "figure4C_fixed_split_reliability.csv", index=False)

    plot_cached(args.output_dir)
    report = {
        "status": "full_cohort_complete",
        "n_subjects": len(expected),
        "anchor_definition": "official_fsaverage5_fixed",
        "candidate_definition": "within_DK_region", "k": 15,
        "pair_seed": 0, "split_seed": 20261021,
        "split_stratification": "hemisphere_x_DK_region",
    }
    (args.output_dir / "figure4C_status.json").write_text(json.dumps(report, indent=2) + "\n")


def plot_cached(output_dir):
    output_dir = Path(output_dir)
    fits = pd.read_csv(output_dir / "figure4C_correlation_length_fits.csv")
    full = fits.loc[fits.fit_scope.eq("full") & fits.fit_acceptable].copy()
    paired = fits.loc[fits.fit_scope.isin(["fixed_half_1", "fixed_half_2"]) & fits.fit_acceptable].pivot(
        index="subject_id", columns="fit_scope", values="lambda_mm").dropna()
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, VIOLIN_STYLE, save_publication_figure,
        set_publication_style, style_axes,
    )
    set_publication_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[.8, 1.4], wspace=.36)
    ax = fig.add_subplot(grid[0, 0])
    values = full.lambda_mm.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    bw_method = lambda kde: kde.scotts_factor() * VIOLIN_STYLE["bw_adjust"]
    violin = ax.violinplot(
        [values], positions=[0], widths=VIOLIN_STYLE["width"],
        showextrema=False, showmedians=False, bw_method=bw_method,
    )
    violin_edge = "#233E68"
    for body in violin["bodies"]:
        body.set_gid("subject_violin")
        body.set_alpha(None)
        body.set_facecolor(mcolors.to_rgba(PALETTE["blue"], alpha=.42))
        body.set_edgecolor(mcolors.to_rgba(violin_edge, alpha=1))
        body.set_linewidths([1.1])
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    ax.vlines(0, q25, q75, color=violin_edge, linewidth=.8, zorder=3)
    ax.hlines([q25, q75], -.055, .055, color=violin_edge,
              linewidth=.9, zorder=4)
    ax.hlines(median, -.085, .085, color=violin_edge,
              linewidth=1.3, zorder=4)
    ax.set(xlim=(-.5, .5), xticks=[0], xticklabels=["Full-data fit"],
           ylabel="Correlation length, λ (mm)")
    style_axes(ax)

    ax = fig.add_subplot(grid[0, 1])
    ax.scatter(paired.fixed_half_1, paired.fixed_half_2, s=9,
               color=PALETTE["blue"], alpha=.55, linewidth=0)
    low = float(min(paired.min().min(), 0))
    high = float(paired.max().max())
    padding = .04 * (high - low)
    ax.plot([low, high], [low, high], color=PALETTE["reference"], linewidth=.9, linestyle="--")
    ax.set(xlim=(low-padding, high+padding), ylim=(low-padding, high+padding),
           xlabel="Half 1 λ (mm)", ylabel="Half 2 λ (mm)")
    style_axes(ax)
    save_publication_figure(fig, output_dir / "Figure4C_ico5_K15_correlation_length")
    plt.close(fig)


if __name__ == "__main__":
    main()
