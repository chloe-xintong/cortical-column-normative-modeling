#!/usr/bin/env python3
"""Render the independent Figure 6 regional pilot figure (Panels A-E).

Description
-----------
Render the independent Figure 6 regional pilot figure (Panels A-E).

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/plot_pilot.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

COLORS = {"M0": "#4C78A8", "M1": "#72B7B2", "M2": "#F58518", "M3": "#E45756"}
MODEL_LABELS = {"M0": "Linear", "M1": "Spline", "M2": "Linear + thickness", "M3": "Spline + thickness"}
DEPTH_COLORS = {0.2: "#3B82A0", 0.5: "#E28E2C", 0.8: "#8E63A9"}


def panel_label(axes, label: str, title: str) -> None:
    first = np.ravel(np.asarray(axes, dtype=object))[0]
    first.text(-0.16, 1.14, label, transform=first.transAxes, fontsize=16, fontweight="bold", va="top")
    first.text(-0.08, 1.14, title, transform=first.transAxes, fontsize=13, fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    cohort = pd.read_csv(root / "cohort_audit.csv").query("included == True")
    folds = pd.read_csv(root / "fold_balance.csv")
    calibration = pd.read_csv(root / "calibration_by_model.csv").set_index("model")
    by_age = pd.read_csv(root / "calibration_by_age_group.csv")
    predictions = pd.read_parquet(root / "oof_predictions_long.parquet")
    effects = pd.read_csv(root / "fold_age_effects.csv")
    centiles = pd.read_csv(root / "selected_region_centile_data.csv")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 120})
    fig = plt.figure(figsize=(18, 24), constrained_layout=True)
    subfigures = fig.subfigures(5, 1, height_ratios=[1.0, 1.0, 1.05, 0.95, 1.7])

    axes_a = subfigures[0].subplots(1, 4)
    panel_label(axes_a, "A", "Cohort support")
    axes_a[0].hist(cohort.age, bins=np.arange(40, 81, 2), color="#5B8E7D", alpha=.85, edgecolor="white")
    axes_a[0].plot(cohort.age, np.full(len(cohort), -0.45), "|", color="#254F45", alpha=.45)
    axes_a[0].set(xlabel="Age (years)", ylabel="Subjects", title="Age distribution")
    counts = cohort.groupby(["age_bin", "sex_code"], observed=True).size().unstack(fill_value=0)
    x = np.arange(len(counts)); bottom = np.zeros(len(counts))
    for code, color in ((1, "#4C78A8"), (2, "#F58518")):
        axes_a[1].bar(x, counts[code], bottom=bottom, label=f"sex_code_{code}", color=color)
        bottom += counts[code].to_numpy()
    axes_a[1].set_xticks(x, counts.index, rotation=25)
    axes_a[1].set(ylabel="Subjects", title="Age bin by sex code")
    axes_a[1].legend(frameon=False)
    fold_totals = folds.groupby("outer_fold").n.sum()
    axes_a[2].bar(fold_totals.index, fold_totals.values, color="#9C755F")
    axes_a[2].set(xlabel="Outer fold", ylabel="Held-out subjects", title="Fold sizes")
    fold_age = folds.groupby(["outer_fold", "age_bin"], observed=True).n.sum().unstack(fill_value=0)
    fold_age.plot(kind="bar", stacked=True, ax=axes_a[3], colormap="viridis")
    axes_a[3].set(xlabel="Outer fold", ylabel="Held-out subjects", title="Fold-wise age support")
    axes_a[3].legend(title="Age bin", frameon=False, fontsize=8)

    axes_b = subfigures[1].subplots(1, 4)
    panel_label(axes_b, "B", "OOF model calibration comparison")
    plot_metrics = [("mean_z", "Mean z", 0.0), ("sd_z", "SD z", 1.0),
                    ("coverage_error", "Mean absolute coverage error", 0.0), ("mean_nlpd", "Mean NLPD", None)]
    models = list(MODEL_LABELS)
    for ax, (metric, title, reference) in zip(axes_b, plot_metrics):
        values, lows, highs = [], [], []
        for model in models:
            row = calibration.loc[model]
            if metric == "coverage_error":
                value = np.mean([row.coverage_50_absolute_error, row.coverage_80_absolute_error, row.coverage_90_absolute_error])
                low = np.mean([row.coverage_50_absolute_error_ci_low, row.coverage_80_absolute_error_ci_low,
                               row.coverage_90_absolute_error_ci_low])
                high = np.mean([row.coverage_50_absolute_error_ci_high, row.coverage_80_absolute_error_ci_high,
                                row.coverage_90_absolute_error_ci_high])
            else:
                value = row[metric]; low = row[f"{metric}_ci_low"]; high = row[f"{metric}_ci_high"]
            values.append(value); lows.append(value - low); highs.append(high - value)
        for i, model in enumerate(models):
            ax.errorbar(i, values[i], yerr=[[lows[i]], [highs[i]]], fmt="o", color=COLORS[model], capsize=3)
        ax.set_xticks(range(4), models)
        ax.set_title(title)
        if reference is not None:
            ax.axhline(reference, color="0.45", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=.2)

    selected = predictions[predictions.model.eq("M2")].copy()
    axes_c = subfigures[2].subplots(1, 4)
    panel_label(axes_c, "C", "Selected-model OOF z-score diagnostics")
    z = selected.z.to_numpy()
    axes_c[0].hist(z, bins=np.linspace(-4, 4, 65), density=True, color=COLORS["M2"], alpha=.7)
    grid = np.linspace(-4, 4, 300)
    axes_c[0].plot(grid, stats.norm.pdf(grid), color="black", linewidth=1.5, label="N(0,1)")
    axes_c[0].set(xlabel="OOF z", ylabel="Density", title="Z-score distribution")
    axes_c[0].legend(frameon=False)
    probabilities = np.linspace(.005, .995, 199)
    axes_c[1].scatter(stats.norm.ppf(probabilities), np.quantile(z, probabilities), s=10, color=COLORS["M2"])
    axes_c[1].plot([-3, 3], [-3, 3], color="0.4", linestyle="--")
    axes_c[1].set(xlabel="Theoretical normal quantile", ylabel="Observed z quantile", title="Q–Q plot")
    subject_z = selected.groupby(["subject_id", "age"], observed=True).z.mean().reset_index()
    axes_c[2].scatter(subject_z.age, subject_z.z, s=14, alpha=.65, color=COLORS["M2"])
    smooth = np.polyfit(subject_z.age, subject_z.z, 1)
    axes_c[2].plot([40, 79], np.polyval(smooth, [40, 79]), color="black")
    axes_c[2].axhline(0, color="0.5", linestyle="--")
    axes_c[2].set(xlabel="Age (years)", ylabel="Subject-mean OOF z", title="Z versus age")
    age_selected = by_age[by_age.model.eq("M2")]
    x = np.arange(len(age_selected))
    for metric, expected, marker in (("coverage_50", .5, "o"), ("coverage_80", .8, "s"), ("coverage_90", .9, "^")):
        axes_c[3].plot(x, age_selected[metric], marker=marker, label=metric.replace("coverage_", "") + "% interval")
        axes_c[3].axhline(expected, color="0.75", linestyle=":", linewidth=.8)
    axes_c[3].set_xticks(x, age_selected.age_bin, rotation=25)
    axes_c[3].set(ylabel="Observed coverage", title="Coverage by age group")
    axes_c[3].legend(frameon=False, fontsize=8)

    axes_d = subfigures[3].subplots(1, 3)
    panel_label(axes_d, "D", "Thickness sensitivity")
    metrics = ["coverage_50_absolute_error", "coverage_80_absolute_error", "coverage_90_absolute_error"]
    x = np.arange(3)
    axes_d[0].bar(x - .18, [calibration.loc["M0", m] for m in metrics], .36, color=COLORS["M0"], label="M0")
    axes_d[0].bar(x + .18, [calibration.loc["M2", m] for m in metrics], .36, color=COLORS["M2"], label="M2")
    axes_d[0].set_xticks(x, ["50%", "80%", "90%"])
    axes_d[0].set(xlabel="Predictive interval", ylabel="Absolute coverage error", title="Calibration with/without thickness")
    axes_d[0].legend(frameon=False)
    effect_pair = effects[effects.model.isin(["M0", "M2"])].groupby(
        ["model", "region_index", "depth_index"], observed=True
    ).predicted_change_age_40_to_78.mean().unstack("model").dropna()
    axes_d[1].scatter(effect_pair.M0, effect_pair.M2, s=8, alpha=.35, color="#6F4E7C")
    limits = [float(effect_pair.to_numpy().min()), float(effect_pair.to_numpy().max())]
    axes_d[1].plot(limits, limits, color="0.4", linestyle="--")
    axes_d[1].set(xlabel="M0 age effect", ylabel="M2 age effect", title="Age-effect agreement")
    z_pair = predictions[predictions.model.isin(["M0", "M2"])].pivot_table(
        index=["subject_id", "region_index", "depth_index"], columns="model", values="z"
    ).dropna()
    sample = z_pair.sample(min(12000, len(z_pair)), random_state=20261106)
    axes_d[2].scatter(sample.M0, sample.M2, s=4, alpha=.15, color="#B279A2")
    axes_d[2].plot([-4, 4], [-4, 4], color="0.4", linestyle="--")
    axes_d[2].set(xlim=(-4, 4), ylim=(-4, 4), xlabel="M0 OOF z", ylabel="M2 OOF z",
                  title="Individual z-score agreement")

    axes_e = subfigures[4].subplots(2, 3, sharex=True)
    panel_label(axes_e, "E", "Figure 3-selected regional centile curves")
    region_order = ["lateraloccipital", "caudalmiddlefrontal", "frontalpole"]
    for row_i, hemi in enumerate(["lh", "rh"]):
        for col_i, region_name in enumerate(region_order):
            ax = axes_e[row_i, col_i]
            subset = centiles[(centiles.hemisphere.eq(hemi)) & (centiles.dk_label_name.eq(region_name))]
            for depth in [0.2, 0.5, 0.8]:
                observed = subset[(subset.row_type.eq("observed")) & np.isclose(subset.depth, depth)]
                curve = subset[(subset.row_type.eq("curve")) & np.isclose(subset.depth, depth)].sort_values("age")
                ax.scatter(observed.age, observed.observed_md, s=4, alpha=.07, color=DEPTH_COLORS[depth])
                ax.plot(curve.age, curve.pred_median, color=DEPTH_COLORS[depth], linewidth=1.5,
                        label=f"depth {depth:.2f}")
                ax.fill_between(curve.age, curve.pred_p10, curve.pred_p90, color=DEPTH_COLORS[depth], alpha=.10)
            ax.set_title(f"{hemi} {region_name.replace('middlefrontal', ' middle frontal').replace('lateraloccipital', 'lateral occipital')}")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("MD")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 6 regional normative-model pilot", fontsize=18, fontweight="bold")
    fig.savefig(root / "figure6.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "figure6.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
