#!/usr/bin/env python3
"""Summarize formal ico5 within-DK K=15 Figure 4C correlation lengths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style,
        style_axes,
    )
    set_publication_style()
    rng = np.random.default_rng(20261021)
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[.8, 1.4], wspace=.36)
    ax = fig.add_subplot(grid[0, 0])
    jitter = rng.uniform(-.10, .10, len(full))
    ax.scatter(jitter, full.lambda_mm, s=8, color=PALETTE["blue"], alpha=.45, linewidth=0)
    q25, median, q75 = full.lambda_mm.quantile([.25, .5, .75])
    ax.vlines(0, q25, q75, color=PALETTE["reference"], linewidth=2.4)
    ax.hlines(median, -.18, .18, color=PALETTE["reference"], linewidth=1.3)
    ax.set(xlim=(-.32, .32), xticks=[], xlabel="Full-data fit", ylabel="Correlation length, λ (mm)")
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
    row = reliability.iloc[0]
    ax.text(.03, .97,
            f"Pearson r = {row.pearson_r:.2f}\nSpearman ρ = {row.spearman_rho:.2f}\nICC = {row.icc_two_way_single:.2f}\nMAE = {row.median_absolute_difference_mm:.2f} mm",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    style_axes(ax)
    save_publication_figure(fig, args.output_dir / "Figure4C_ico5_K15_correlation_length")
    plt.close(fig)
    report = {
        "status": "full_cohort_complete",
        "n_subjects": len(expected),
        "anchor_definition": "official_fsaverage5_fixed",
        "candidate_definition": "within_DK_region", "k": 15,
        "pair_seed": 0, "split_seed": 20261021,
        "split_stratification": "hemisphere_x_DK_region",
    }
    (args.output_dir / "figure4C_status.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
