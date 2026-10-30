"""Shared publication-quality plotting style for cortical trajectory figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FIGURE_WIDTH_IN = 7.09  # 180 mm, two-column width
RASTER_DPI = 600
PALETTE = {
    "blue": "#3B6FB6",
    "orange": "#E69F00",
    "red": "#B2182B",
    "gray": "#8C8C8C",
    "reference": "#333333",
    "masked": "#D9D9D9",
}


def set_publication_style() -> None:
    """Apply the common typographic and line-art defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "figure.dpi": 120,
            "savefig.dpi": RASTER_DPI,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    """Apply consistent two-dimensional axes styling."""
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.8, length=3, pad=2)
    if grid:
        ax.grid(axis="y", color=PALETTE["masked"], linewidth=0.6, zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    """Place a consistent bold panel letter at the upper-left of an axes."""
    ax.text(
        -0.15,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    """Save vector and 600-dpi raster versions of a figure with one call."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    for suffix in (".png", ".tiff"):
        fig.savefig(stem.with_suffix(suffix), dpi=RASTER_DPI, bbox_inches="tight")
