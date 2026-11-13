#!/usr/bin/env python3
"""Build the integrated Figure 6 overview figure and numerical summary.

Description
-----------
Build the integrated Figure 6 overview figure and numerical summary.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/build_figure6_overall_summary.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_row(rows, section, metric, estimate, ci_lower=np.nan, ci_upper=np.nan,
            unit="", interpretation=""):
    rows.append({
        "section": section,
        "metric": metric,
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "unit": unit,
        "interpretation": interpretation,
    })


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Unified Figure 6 result root.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    plot_input = root / "plotting_inputs" / "model_selection_calibration_186"
    ico4 = root / "ico4_warped_m2_186"
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    regional_selection = pd.read_csv(plot_input / "figure1_regional_model_selection.csv")
    regional_coverage = pd.read_csv(plot_input / "figure2a_regional_coverage_deviation.csv")
    regional_distribution = pd.read_csv(plot_input / "figure2b_warped_vs_gaussian_mll.csv")
    ico4_cal = pd.read_csv(ico4 / "oof" / "calibration_overall.csv").iloc[0]
    ico4_boot = pd.read_csv(ico4 / "oof" / "subject_bootstrap_calibration_ci.csv")
    ico4_boot = ico4_boot[(ico4_boot.scope == "overall") & (ico4_boot.level == "all")]
    boot = ico4_boot.set_index("metric")
    with (ico4 / "full_models" / "full_cohort_validation.json").open() as handle:
        full = json.load(handle)
    selection = pd.read_csv(ico4 / "full_models" / "representative_node_selection.csv").iloc[0]
    centiles = pd.read_parquet(ico4 / "full_models" / "representative_node_centiles.parquet")

    rows = []
    add_row(rows, "Frozen analysis", "Participants", 186, unit="participants")
    add_row(rows, "Frozen analysis", "Outer-fold test sizes", "38; 38; 37; 36; 37", unit="participants")
    add_row(rows, "Frozen analysis", "Bilateral ico4 nodes", 4687, unit="nodes")
    add_row(rows, "Frozen analysis", "Normalized cortical depths", 21, unit="depths")
    add_row(rows, "Frozen analysis", "Valid OOF predictions", int(ico4_cal.n_predictions), unit="predictions")

    for row in regional_selection.itertuples(index=False):
        add_row(
            rows, "DK-region model selection", f"Delta mean log loss: {row.contrast}",
            row.delta_mll, row.ci_lower, row.ci_upper, "mean log loss",
            "Negative values favor the first model in the contrast.",
        )
    dist = regional_distribution.iloc[0]
    add_row(
        rows, "DK-region distribution selection", "Delta mean log loss: Warped M2 - Gaussian M2",
        dist.delta_mll, dist.ci_lower, dist.ci_upper, "mean log loss",
        "Negative values favor Warped M2.",
    )
    for row in regional_coverage.itertuples(index=False):
        add_row(
            rows, "DK-region calibration", f"{row.display_model}: {int(row.nominal_coverage * 100)}% coverage deviation",
            row.deviation_percentage_points, unit="percentage points",
            interpretation="Zero is ideal.",
        )

    metric_specs = [
        ("mean_z", "Mean z", "z units"),
        ("sd_z", "SD(z)", "z units"),
        ("coverage_50", "50% coverage", "proportion"),
        ("coverage_80", "80% coverage", "proportion"),
        ("coverage_90", "90% coverage", "proportion"),
        ("mean_log_likelihood", "Mean predictive log likelihood", "log density"),
        ("coverage_mae", "Coverage MAE", "proportion"),
        ("fraction_abs_z_gt_2", "Fraction |z| > 2", "proportion"),
        ("fraction_abs_z_gt_3", "Fraction |z| > 3", "proportion"),
        ("fraction_abs_z_gt_4", "Fraction |z| > 4", "proportion"),
    ]
    for key, label, unit in metric_specs:
        r = boot.loc[key]
        add_row(
            rows, "ico4 held-out calibration", label, r.estimate, r.ci_lower, r.ci_upper, unit,
            "95% CI from participant-level bootstrap resampling.",
        )
    add_row(rows, "ico4 held-out calibration", "Skewness", ico4_cal.skewness)
    add_row(rows, "ico4 held-out calibration", "Excess kurtosis", ico4_cal.excess_kurtosis)
    add_row(rows, "ico4 held-out calibration", "Q-Q RMSE", ico4_cal.qq_rmse)
    add_row(rows, "ico4 held-out calibration", "Maximum |z|", ico4_cal.max_abs_z, unit="z units")
    add_row(rows, "Full-cohort model", "Successful node-depth models", full["successful_models"], unit="models")
    add_row(rows, "Full-cohort model", "Failed node-depth models", full["failed_models"], unit="models")
    add_row(rows, "Full-cohort model", "Models with candidate-optimization warnings", full["warning_models"], unit="models")
    add_row(rows, "Full-cohort model", "Displayed age range", "45-76", unit="years")
    add_row(rows, "Representative curve", "Hemisphere", selection.hemisphere)
    add_row(rows, "Representative curve", "DK region", selection.dk_label_name)
    add_row(rows, "Representative curve", "ico4 node ID", selection.ico_node_id)
    add_row(rows, "Representative curve", "fsaverage vertex ID", selection.fsaverage_vertex_id)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "figure6_numerical_summary.csv", index=False)

    md_lines = [
        "# Figure 6 numerical summary", "",
        "## Frozen analysis", "",
        "- 186 CN participants; source-image exclusions were D0271 and D0285.",
        "- Original five held-out fold sizes: 38, 38, 37, 36, and 37.",
        "- 4,687 bilateral ico4 nodes × 21 depths; 18,100,551 valid OOF predictions.", "",
        "## DK-region model selection", "",
        "| Comparison | Delta mean log loss | Participant-bootstrap 95% CI | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for row in regional_selection.itertuples(index=False):
        interpretation = "supports M2" if row.contrast == "M2-M0" and row.ci_upper < 0 else "no stable difference"
        md_lines.append(
            f"| {row.comparison_label} | {row.delta_mll:.6f} | "
            f"[{row.ci_lower:.6f}, {row.ci_upper:.6f}] | {interpretation} |"
        )
    md_lines.extend([
        f"| Warped M2 - Gaussian M2 | {dist.delta_mll:.6f} | "
        f"[{dist.ci_lower:.6f}, {dist.ci_upper:.6f}] | supports Warped M2 |",
        "", "Negative delta mean log loss favors the first model in each comparison.", "",
        "## Final ico4 held-out calibration", "",
        "| Metric | Estimate | Participant-bootstrap 95% CI |",
        "|---|---:|---:|",
    ])
    for key, label, _ in metric_specs[:7]:
        r = boot.loc[key]
        md_lines.append(f"| {label} | {r.estimate:.6f} | [{r.ci_lower:.6f}, {r.ci_upper:.6f}] |")
    md_lines.extend([
        "", f"Skewness = {ico4_cal.skewness:.6f}; excess kurtosis = {ico4_cal.excess_kurtosis:.6f}; "
        f"Q-Q RMSE = {ico4_cal.qq_rmse:.6f}; maximum |z| = {ico4_cal.max_abs_z:.4f}.", "",
        "## Full-cohort model", "",
        f"- Successful models: {full['successful_models']:,}/{full['expected_models']:,}; failures: {full['failed_models']}.",
        f"- Candidate-optimization warnings: {full['warning_models']:,}; all saved final parameters and centiles are finite.",
        "- Conditional 5th, 50th, and 95th percentiles are available over ages 45-76.",
        f"- The overview curve shows {'left' if selection.hemisphere == 'lh' else 'right'} {selection.dk_label_name}, ico4 node "
        f"{int(selection.ico_node_id)}, at depth 0.50, with node-local median thickness fixed over age.", "",
        "Overall calibration does not imply that every node × depth is individually calibrated. "
        "The sex-code source mapping remains unverified, so neutral labels are retained.",
    ])
    (out / "figure6_numerical_summary.md").write_text("\n".join(md_lines) + "\n")

    colors = {"blue": "#3B6FB6", "orange": "#D9822B", "teal": "#2A9D8F", "gray": "#77808A", "dark": "#263238"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 11, "axes.labelsize": 9.5,
        "legend.fontsize": 8.5, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))

    # A: regional model selection.
    ax = axes[0, 0]
    forest = regional_selection.copy()
    y = np.arange(len(forest))[::-1]
    point_colors = [colors["gray"], colors["gray"], colors["teal"]]
    for yi, row, color in zip(y, forest.itertuples(index=False), point_colors):
        ax.errorbar(
            row.delta_mll, yi,
            xerr=[[row.delta_mll - row.ci_lower], [row.ci_upper - row.delta_mll]],
            fmt="o", color=color, ecolor=color, capsize=3, markersize=6, lw=1.5,
        )
    ax.axvline(0, color="0.45", lw=1, ls="--")
    ax.set_yticks(y, ["M1 - M0\nNonlinear age", "M3 - M2\nNonlinear age + thickness", "M2 - M0\nAdd local thickness"])
    ax.set_xlabel("Difference in mean log loss")
    ax.set_title("A  DK-region model selection", loc="left", fontweight="semibold")
    ax.text(0.02, -0.25, "Negative values favor the first model", transform=ax.transAxes, color="0.35")
    clean_axis(ax)

    # B: regional response-distribution selection.
    ax = axes[0, 1]
    offsets = {"Gaussian M2": -0.8, "Warped M2": 0.8}
    model_colors = {"Gaussian M2": colors["gray"], "Warped M2": colors["teal"]}
    for name, group in regional_coverage.groupby("display_model", sort=False):
        x = group.nominal_coverage.to_numpy() * 100 + offsets[name]
        ax.plot(x, group.deviation_percentage_points, marker="o", lw=1.5,
                color=model_colors[name], label=name)
    ax.axhline(0, color="0.45", lw=1, ls="--")
    ax.set_xticks([50, 80, 90], ["50%", "80%", "90%"])
    ax.set_ylim(-1.5, 4.2)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Coverage deviation (percentage points)")
    ax.set_title("B  DK-region distribution calibration", loc="left", fontweight="semibold")
    ax.text(
        0.03, 0.95,
        f"Warped - Gaussian log loss\n{dist.delta_mll:.4f} "
        f"[{dist.ci_lower:.4f}, {dist.ci_upper:.4f}]",
        transform=ax.transAxes, va="top", color=colors["dark"], fontsize=8.7,
    )
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    # C: formal ico4 held-out calibration with participant-bootstrap intervals.
    ax = axes[1, 0]
    nominal = np.array([0.50, 0.80, 0.90])
    keys = ["coverage_50", "coverage_80", "coverage_90"]
    estimates = np.array([boot.loc[k, "estimate"] for k in keys])
    lower = np.array([boot.loc[k, "ci_lower"] for k in keys])
    upper = np.array([boot.loc[k, "ci_upper"] for k in keys])
    deviation = 100 * (estimates - nominal)
    yerr = np.vstack([100 * (estimates - lower), 100 * (upper - estimates)])
    ax.errorbar(nominal * 100, deviation, yerr=yerr, fmt="o", color=colors["blue"],
                ecolor=colors["blue"], capsize=4, lw=1.5, markersize=6)
    ax.axhline(0, color="0.45", lw=1, ls="--")
    ax.set_xticks([50, 80, 90], ["50%", "80%", "90%"])
    ax.set_ylim(-1.5, 4.2)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Coverage deviation (percentage points)")
    ax.set_title("C  Final ico4 held-out calibration", loc="left", fontweight="semibold")
    ax.text(0.03, 0.95, f"Mean z = {ico4_cal.mean_z:.5f}\nSD(z) = {ico4_cal.sd_z:.5f}",
            transform=ax.transAxes, va="top", color=colors["dark"])
    clean_axis(ax)

    # D: one reproducible original-scale conditional reference curve.
    ax = axes[1, 1]
    curve = centiles[centiles.depth_index.eq(10)].copy()
    for sex_code, color in [(1, colors["blue"]), (2, colors["orange"])]:
        group = curve[curve.sex_code.eq(sex_code)].sort_values("age")
        age = group.age.to_numpy()
        p05 = group.p05_md.to_numpy() * 1e3
        p50 = group.p50_md.to_numpy() * 1e3
        p95 = group.p95_md.to_numpy() * 1e3
        ax.fill_between(age, p05, p95, color=color, alpha=0.13, lw=0)
        ax.plot(age, p50, color=color, lw=1.8, label=f"Sex code {sex_code}")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Mean diffusivity (×10⁻³ mm²/s)")
    ax.set_title("D  Conditional reference curves", loc="left", fontweight="semibold")
    ax.text(0.03, 0.96, "Left parahippocampal\nico4 node 2460; depth 0.50",
            transform=ax.transAxes, va="top", fontsize=8.7, color=colors["dark"])
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    fig.suptitle("Figure 6 overview: model selection and final ico4 performance",
                 x=0.07, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.11, hspace=0.43, wspace=0.34)
    fig.text(
        0.07, 0.025,
        "Intervals in A and B are paired participant-bootstrap 95% CIs; intervals in C summarize fixed OOF predictions by resampling participants. "
        "Shading in D is the conditional 5th-95th percentile range, not uncertainty in the median.",
        ha="left", va="bottom", fontsize=8.2, color="0.35",
    )
    outputs = []
    for suffix, kwargs in [("png", {"dpi": 300}), ("pdf", {}), ("svg", {})]:
        path = out / f"figure6_overall_summary.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)

    manifest_paths = outputs + [out / "figure6_numerical_summary.csv", out / "figure6_numerical_summary.md"]
    manifest = pd.DataFrame([
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in manifest_paths
    ])
    manifest.to_csv(out / "summary_manifest.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
