#!/usr/bin/env python3
"""Create the frozen main and supplementary Figure 4B plots."""

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
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style,
        style_axes,
    )
    set_publication_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=.34)
    ax = fig.add_subplot(grid[0, 0])
    ax.axvspan(0, 10, color=PALETTE["masked"], alpha=.28, linewidth=0, zorder=0)
    ax.plot(x, table.real_dissimilarity_median, color=PALETTE["blue"],
            linewidth=1.5, label="Real")
    ax.fill_between(x, table.real_dissimilarity_q25, table.real_dissimilarity_q75,
                    color=PALETTE["blue"], alpha=.20, linewidth=0)
    ax.plot(x, table.null_dissimilarity_median, color=PALETTE["orange"],
            linewidth=1.4, linestyle="--", label="DWI grid, FWHM 0 mm")
    ax.fill_between(x, table.null_dissimilarity_q25, table.null_dissimilarity_q75,
                    color=PALETTE["orange"], alpha=.20, linewidth=0)
    ax.text(5, .98, "0–10 mm", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7)
    ax.set(xlabel="White-surface geodesic distance (mm)",
           ylabel="Profile-shape dissimilarity (1 − r)")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    ax = fig.add_subplot(grid[0, 1])
    ax.axvspan(0, 10, color=PALETTE["masked"], alpha=.28, linewidth=0, zorder=0)
    ax.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    ax.plot(x, table.real_minus_null_local_excess_median,
            color=PALETTE["orange"], linewidth=1.5)
    ax.fill_between(x, table.real_minus_null_local_excess_q25,
                    table.real_minus_null_local_excess_q75,
                    color=PALETTE["orange"], alpha=.20, linewidth=0)
    ax.set(xlabel="White-surface geodesic distance (mm)",
           ylabel="Residual local-excess similarity\n$L_{real}(d)-L_{null}(d)$")
    style_axes(ax)
    save_publication_figure(fig, args.cohort_dir / "Figure4B_ico5_K15_formal_main")
    plt.close(fig)

    values = auc.auc_residual_local_excess_0_10_mm.to_numpy(dtype=float)
    rng = np.random.default_rng(20261020)
    fig, ax = plt.subplots(figsize=(3.45, 3.25), constrained_layout=True)
    ax.scatter(rng.uniform(-.11, .11, len(values)), values, s=8,
               color=PALETTE["blue"], alpha=.5, linewidth=0)
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    ax.vlines(0, q25, q75, color=PALETTE["reference"], linewidth=2.4)
    ax.hlines(median, -.18, .18, color=PALETTE["reference"], linewidth=1.3)
    ax.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    ax.set(xlim=(-.3, .3), xticks=[], xlabel=f"Subjects (n = {auc.subject_id.nunique()})",
           ylabel="Residual local-excess AUC\n(Δr × mm; 0–10 mm)")
    style_axes(ax)
    save_publication_figure(fig, args.cohort_dir / "Figure4B_ico5_K15_AUC_supplement")
    plt.close(fig)


if __name__ == "__main__":
    main()
