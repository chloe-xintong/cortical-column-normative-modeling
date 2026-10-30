#!/usr/bin/env python3
"""Plot formal Figure 4A with dissimilarity and symmetric relative MAE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-summary", type=Path, required=True)
    parser.add_argument("--relative-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_summary).sort_values("distance_bin_index")
    relative = pd.read_csv(args.relative_summary).sort_values("distance_bin_index")
    merge_keys = ["distance_bin_index", "distance_bin"]
    if len(real) != 15 or len(relative) != 15:
        raise ValueError("Expected 15 two-millimetre distance bins in both inputs")
    if not real[merge_keys].reset_index(drop=True).equals(
        relative[merge_keys].reset_index(drop=True)
    ):
        raise ValueError("Dissimilarity and relative-MAE distance bins differ")
    if not real["n_subjects"].eq(relative["n_subjects"]).all():
        raise ValueError("Dissimilarity and relative-MAE subject counts differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = real.merge(
        relative[
            merge_keys
            + [
                "median", "q25", "q75", "bootstrap_ci95_low",
                "bootstrap_ci95_high", "anchor_count_total", "pair_count_total",
            ]
        ].rename(
            columns={
                "median": "symmetric_relative_mae_pct_median",
                "q25": "symmetric_relative_mae_pct_q25",
                "q75": "symmetric_relative_mae_pct_q75",
                "bootstrap_ci95_low": "symmetric_relative_mae_pct_ci95_low",
                "bootstrap_ci95_high": "symmetric_relative_mae_pct_ci95_high",
                "anchor_count_total": "relative_mae_anchor_count_total",
                "pair_count_total": "relative_mae_pair_count_total",
            }
        ),
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    combined.to_csv(
        args.output_dir / "figure4A_formal_dissimilarity_and_relative_mae.csv",
        index=False,
    )

    sys.path.insert(0, str(args.repo_root))
    from qc.publication_style import (  # pylint: disable=import-outside-toplevel
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )

    set_publication_style()
    x = combined["distance_bin_index"].to_numpy(dtype=float) * 2 + 1
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_IN, 2.75), sharex=True,
        gridspec_kw={"wspace": 0.30},
    )
    specifications = [
        (
            axes[0], "profile_dissimilarity_median",
            "profile_dissimilarity_q25", "profile_dissimilarity_q75",
            PALETTE["blue"], "Profile-shape dissimilarity (1 − r)",
        ),
        (
            axes[1], "symmetric_relative_mae_pct_median",
            "symmetric_relative_mae_pct_q25", "symmetric_relative_mae_pct_q75",
            PALETTE["orange"], "Symmetric relative profile MAE (%)",
        ),
    ]
    for axis, median, q25, q75, color, ylabel in specifications:
        axis.plot(
            x, combined[median], color=color, linewidth=1.4,
            marker="o", markersize=2.7,
        )
        axis.fill_between(
            x, combined[q25], combined[q75],
            color=color, alpha=0.2, linewidth=0,
        )
        axis.set(
            xlabel="White-surface geodesic distance (mm)",
            ylabel=ylabel,
            xlim=(0, 30), xticks=np.arange(0, 31, 5),
        )
        style_axes(axis, grid=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.98, wspace=0.30)
    output_stem = args.output_dir / "Figure4A_ico5_K15_with_relative_MAE"
    save_publication_figure(fig, output_stem)
    plt.close(fig)
    (args.output_dir / "figure4A_with_relative_mae_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_subjects": int(combined["n_subjects"].min()),
                "anchor_definition": "official_fsaverage5_fixed",
                "candidate_definition": "within_DK_region",
                "k": 15,
                "metrics": [
                    "profile_shape_dissimilarity_1_minus_r",
                    "symmetric_relative_profile_mae_percent",
                ],
                "relative_mae_definition": (
                    "200 * mean_depth(abs(x-y)) / "
                    "(median_depth(abs(x)) + median_depth(abs(y)))"
                ),
                "aggregation": (
                    "pair -> anchor x distance-bin median -> bilateral subject "
                    "median -> group median/IQR"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
