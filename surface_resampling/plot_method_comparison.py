#!/usr/bin/env python
"""Plot the paired O2-Column versus O3-Coordinate comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_depthwise_agreement(tables: Path, figures: Path) -> None:
    summary = pd.read_csv(tables / "cohort_depth_summary.csv")
    depth = 100.0 * summary["depth"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)

    panels = [
        ("pearson_r", "Pearson correlation", "Agreement"),
        ("nmae_pct", "Normalized MAE (%)", "NMAE"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        median = summary[f"{metric}_median"]
        q25 = summary[f"{metric}_q25"]
        q75 = summary[f"{metric}_q75"]
        ax.plot(depth, median, color="#0072B2", marker="o", linewidth=1.6)
        ax.fill_between(depth, q25, q75, color="#0072B2", alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("Cortical depth (%)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1.02)
    fig.suptitle(
        "O2-Column versus O3-Coordinate across 30 subjects",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, figures / "01_depthwise_agreement_and_nmae.png")


def plot_cross_subject_agreement(tables: Path, figures: Path) -> None:
    cross = pd.read_csv(tables / "cross_subject_depth_summary.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = {"lh": "#0072B2", "rh": "#D55E00"}
    for hemi, group in cross.groupby("hemisphere"):
        group = group.sort_values("depth")
        depth = 100.0 * group["depth"]
        ax.plot(
            depth,
            group["correlation_median"],
            marker="o",
            linewidth=1.5,
            color=colors[hemi],
            label=hemi,
        )
        ax.fill_between(
            depth,
            group["correlation_q25"],
            group["correlation_q75"],
            color=colors[hemi],
            alpha=0.15,
        )
    ax.axhline(0.8, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cortical depth (%)")
    ax.set_ylabel("Cross-subject Pearson correlation")
    ax.set_title("Cross-subject ordering agreement")
    ax.legend(title="Hemisphere", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, figures / "02_cross_subject_depth_agreement.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    tables = args.comparison_root / "tables"
    figures = args.comparison_root / "figures"
    plot_depthwise_agreement(tables, figures)
    plot_cross_subject_agreement(tables, figures)
    print(f"Saved figures to {figures}")


if __name__ == "__main__":
    main()
