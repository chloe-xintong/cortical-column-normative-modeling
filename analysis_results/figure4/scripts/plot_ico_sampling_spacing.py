#!/usr/bin/env python3
"""Plot ico3/ico4/ico5 nearest-neighbor white-surface spacing.

Description
-----------
Plot ico3/ico4/ico5 nearest-neighbor white-surface spacing.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--summary-csv`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/plot_ico_sampling_spacing.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from qc.publication_style import (  # noqa: E402
    PALETTE,
    save_publication_figure,
    set_publication_style,
    style_axes,
)


LEVEL_ORDER = ("ico5", "ico4", "ico3")
LEVEL_COLORS = {
    "ico5": PALETTE["blue"],
    "ico4": PALETTE["orange"],
    "ico3": PALETTE["light_blue"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT
        / "/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure5/outputs/spatial_resolution/ico_nearest_neighbor_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure4/outputs/coverage95_pass_313/real/cohort",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    pooled = summary.loc[
        summary["hemisphere"].eq("bilateral_pooled")
        & summary["level"].isin(LEVEL_ORDER)
    ].copy()
    pooled["level"] = pd.Categorical(
        pooled["level"], categories=LEVEL_ORDER, ordered=True
    )
    pooled = pooled.sort_values("level").reset_index(drop=True)
    if pooled["level"].astype(str).tolist() != list(LEVEL_ORDER):
        raise RuntimeError("Expected one bilateral_pooled row for each of ico5/ico4/ico3")

    set_publication_style()
    fig, axis = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)
    y = np.arange(len(pooled), dtype=float)

    for position, row in zip(y, pooled.itertuples(index=False)):
        color = LEVEL_COLORS[str(row.level)]
        axis.hlines(
            position,
            row.p5_nn_mm,
            row.p95_nn_mm,
            color=color,
            linewidth=1.0,
            alpha=0.72,
            zorder=1,
        )
        axis.hlines(
            position,
            row.q1_nn_mm,
            row.q3_nn_mm,
            color=color,
            linewidth=5.0,
            alpha=0.88,
            zorder=2,
        )
        axis.vlines(
            row.median_nn_mm,
            position - 0.15,
            position + 0.15,
            color=color,
            linewidth=1.5,
            zorder=3,
        )
        axis.annotate(
            f"{row.median_nn_mm:.2f}",
            xy=(row.median_nn_mm, position + 0.15),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#333333",
            fontsize=7.5,
        )

    axis.set_yticks(y, pooled["level"].astype(str))
    axis.set_xlabel("Nearest-neighbor white-surface geodesic distance (mm)")
    axis.set_xlim(0, 12)
    axis.set_xticks(np.arange(0, 13, 2))
    axis.set_ylim(-0.55, len(pooled) - 0.45)
    style_axes(axis)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "Figure4_ico_sampling_spacing"
    save_publication_figure(fig, stem)
    plt.close(fig)

    export = pooled.loc[
        :,
        [
            "level",
            "n_nodes",
            "median_nn_mm",
            "q1_nn_mm",
            "q3_nn_mm",
            "p5_nn_mm",
            "p95_nn_mm",
        ],
    ].copy()
    export.to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
