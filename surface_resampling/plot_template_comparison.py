#!/usr/bin/env python3
"""Create group-level tables and figures from template comparison CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.comparison_csv)
    summary = table.agg({"profile_correlation_median": ["median", "quantile"], "profile_mae": ["median", "quantile"], "profile_rmse": ["median", "quantile"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "template_comparison_group_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, column in zip(axes, ["profile_correlation_median", "profile_mae", "profile_rmse"]):
        sns.boxplot(data=table, y=column, ax=ax, color="#D9EAF2", showfliers=False)
        ax.set_title(column)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "template_comparison_group_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
