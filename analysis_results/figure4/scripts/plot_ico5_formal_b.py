#!/usr/bin/env python3
"""Create the frozen main and supplementary Figure 4B plots.

Description
-----------
Create the frozen main and supplementary Figure 4B plots.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--cohort-dir`` command-line input

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python analysis_results/figure4/scripts/plot_ico5_formal_b.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    args = parser.parse_args()
    group = pd.read_csv(args.cohort_dir / "figure4B_acquisition_group_summary.csv")
    available = int(group.n_realizations.max())
    table = group.loc[group.n_realizations.eq(available)].sort_values("distance_bin_index")
    auc = pd.read_csv(args.cohort_dir / "figure4B_local_excess_auc_subject.csv")
    x = table.distance_bin_index.to_numpy(dtype=float) * 2 + 1

    repo_root = Path(__file__).resolve().parents[3]
    spacing = pd.read_csv(
        repo_root
        / "analysis_results/figure5/outputs/spatial_resolution/ico_nearest_neighbor_summary.csv"
    )
    spacing = spacing.loc[
        spacing.hemisphere.eq("bilateral_pooled")
        & spacing.level.isin(("ico3", "ico4", "ico5"))
    ].set_index("level")
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style,
        style_axes, DERIVED_STYLE, DERIVED_BAND,
    )
    set_publication_style()

    def draw_real_and_null(axis) -> None:
        axis.plot(
            x, table.real_dissimilarity_median, color=PALETTE["red"],
            linewidth=1.8, linestyle="-", marker="o", markevery=2, label="Real",
        )
        axis.fill_between(
            x, table.real_dissimilarity_q25, table.real_dissimilarity_q75,
            color=PALETTE["red"], alpha=.12, linewidth=0,
        )
        axis.plot(
            x, table.null_dissimilarity_median, color=PALETTE["blue"],
            linewidth=1.6, linestyle="--", marker="s", markevery=2,
            label="Null model",
        )
        axis.fill_between(
            x, table.null_dissimilarity_q25, table.null_dissimilarity_q75,
            color=PALETTE["blue"], alpha=.12, linewidth=0,
        )
        axis.set(
            xlabel="White-surface geodesic distance (mm)",
            ylabel="Profile-shape dissimilarity (1 − r)",
        )
        axis.legend(frameon=False, loc="lower right")
        style_axes(axis)

    def draw_residual(axis) -> None:
        axis.plot(
            x, table.real_minus_null_local_excess_median,
            **DERIVED_STYLE, gid="derived_result_line",
        )
        axis.fill_between(
            x, table.real_minus_null_local_excess_q25,
            table.real_minus_null_local_excess_q75,
            **DERIVED_BAND, gid="derived_result_band",
        )
        axis.axhline(0, color="#777777", linewidth=.8, linestyle="--")
        axis.set(
            xlabel="White-surface geodesic distance (mm)",
            ylabel="Residual local-excess similarity\n$L_{real}(d)-L_{null}(d)$",
        )
        style_axes(axis)

    # Preserve the historical combined export, but use separate panels in the notebook.
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=.34)
    draw_real_and_null(fig.add_subplot(grid[0, 0]))
    draw_residual(fig.add_subplot(grid[0, 1]))
    save_publication_figure(fig, args.cohort_dir / "Figure4B_ico5_K15_formal_main")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.3, 2.8), constrained_layout=True)
    draw_real_and_null(axis)
    save_publication_figure(
        fig, args.cohort_dir / "Figure4B_ico5_K15_real_vs_null"
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.3, 2.8), constrained_layout=True)
    draw_residual(axis)
    save_publication_figure(
        fig, args.cohort_dir / "Figure4B_ico5_K15_residual_local_excess"
    )
    plt.close(fig)

    scale_colors = {
        "ico5": PALETTE["blue"],
        "ico4": PALETTE["orange"],
        "ico3": PALETTE["light_blue"],
    }
    fig, axis = plt.subplots(figsize=(4.3, 2.8), constrained_layout=True)
    draw_residual(axis)
    for level in ("ico5", "ico4", "ico3"):
        distance = float(spacing.loc[level, "median_nn_mm"])
        color = scale_colors[level]
        axis.axvline(
            distance,
            color=color,
            linewidth=1.0,
            alpha=.9,
            zorder=4,
            label=f"{level}: {distance:.2f} mm",
        )
    axis.legend(frameon=False, loc="upper right")
    save_publication_figure(
        fig,
        args.cohort_dir / "Figure4_ico_sampling_distances_on_residual_curve",
    )
    # Keep the earlier filename synchronized, now as a separate plot rather than
    # a modification of the combined Figure 4B canvas.
    save_publication_figure(
        fig,
        args.cohort_dir / "Figure4B_ico5_K15_formal_main_with_sampling_scales",
    )
    plt.close(fig)

    values = auc.auc_residual_local_excess_0_10_mm.to_numpy(dtype=float)
    rng = np.random.default_rng(20261020)
    fig, ax = plt.subplots(figsize=(3.45, 3.25), constrained_layout=True)
    ax.scatter(rng.uniform(-.11, .11, len(values)), values, s=8,
               color=PALETTE["orange"], alpha=.5, linewidth=0)
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    ax.vlines(0, q25, q75, color=PALETTE["reference"], linewidth=2.4)
    ax.hlines(median, -.18, .18, color=PALETTE["reference"], linewidth=1.3)
    ax.set(xlim=(-.3, .3), xticks=[], xlabel="Subjects",
           ylabel="Residual local-excess AUC\n(Δr × mm; 0–10 mm)")
    style_axes(ax)
    save_publication_figure(fig, args.cohort_dir / "Figure4B_ico5_K15_AUC_supplement")
    plt.close(fig)


if __name__ == "__main__":
    main()
