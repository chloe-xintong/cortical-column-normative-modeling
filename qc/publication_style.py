"""Shared publication-quality plotting style for cortical trajectory figures.

Description
-----------
Shared publication-quality plotting style for cortical trajectory figures.

Methods
-------
1. Provide the reusable computations defined in this module.
2. Return results to the calling analysis code.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
Import this module and call ``heatmap_norm(...)``; it has no command-line entry point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FIGURE_WIDTH_IN = 7.09  # 180 mm, two-column width
RASTER_DPI = 600
PALETTE = {
    "blue": "#4B74B2",
    "light_blue": "#90BEE0",
    "pale_blue": "#E6F1F3",
    "yellow": "#FFDF92",
    "orange": "#FC8C5A",
    "red": "#DB3124",
    "neutral": "#F7F7F7",
    "missing": "#D9D9D9",
    "gray": "#8A8A8A",
    "background": "#F2F2F2",
    "reference": "#333333",
    "masked": "#D9D9D9",
}
# Heatmap colors encode numbers, independently of categorical emphasis colors.
# N=257 includes the exact midpoint (0.5), which must be neutral, not pale blue.
CMAP_SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "project_blue", [PALETTE[k] for k in ("pale_blue", "light_blue", "blue")], N=257)
CMAP_DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "project_diverging", [PALETTE[k] for k in
        ("blue", "light_blue", "pale_blue", "neutral", "yellow", "orange", "red")], N=257)
for cmap in (CMAP_SEQUENTIAL, CMAP_DIVERGING):
    cmap.set_bad(PALETTE["missing"])
    if cmap.name not in mpl.colormaps:
        mpl.colormaps.register(cmap)


def heatmap_norm(kind: str, *, vmin: float = 0, vmax: float = 1):
    """Explicit scales shared by all comparable panels; never infer per panel.

    Nonnegative/similarity scales increase from pale to dark blue. Signed
    coefficients/differences have symmetric limits and an exact zero midpoint.
    Signed correlation matrices always use [-1, 1]. Missing values stay masked.
    """
    import math
    if kind == "correlation":
        return mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    if not math.isfinite(vmax) or vmax <= 0:
        raise ValueError("Heatmap vmax must be finite and positive")
    if kind == "signed":
        return mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    if kind == "nonnegative":
        if not math.isfinite(vmin) or vmin < 0 or vmin >= vmax:
            raise ValueError("Nonnegative heatmap requires 0 <= vmin < vmax")
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    raise ValueError(f"Unknown heatmap kind: {kind}")
# Unordered categories never cycle through red (emphasis) or pale blue (bands).
SERIES_COLORS = tuple(PALETTE[k] for k in ('blue','orange','light_blue','gray'))
DEFAULT_BAND = {"color": PALETTE["light_blue"], "alpha": .12}
DERIVED_STYLE = dict(color=PALETTE["orange"], linewidth=1.9, marker="o",
                     markersize=3, markeredgewidth=0)
DERIVED_BAND = dict(color=PALETTE["orange"], alpha=.16, linewidth=0)
METHOD_COLORS = {"O3": PALETTE["red"], "O2FS": PALETTE["blue"], "O2Column": PALETTE["orange"]}
COMPARISON_COLORS = {"O2Column_vs_O3": PALETTE["orange"], "O2FS_vs_O3": PALETTE["blue"]}
MODEL_COLORS = {
    "regional_mean_md": PALETTE["light_blue"],
    "regional_profile": PALETTE["blue"],
    "regional_thickness": PALETTE["gray"],
    "regional_profile_thickness": PALETTE["orange"],
    "ico3_md": PALETTE["light_blue"],
    "ico4_md": PALETTE["blue"],
    "ico5_md": PALETTE["blue"],
    "fsaverage_full_md": PALETTE["orange"],
    "ico4_column_mean_md": PALETTE["gray"],
    "ico4_md_length": PALETTE["red"],
    "ico4_md_curvature": PALETTE["light_blue"],
    "ico4_md_length_curvature": PALETTE["gray"],
    "o2fs_ico4_md": PALETTE["blue"],
    "o2fs_ico4_md_length": PALETTE["light_blue"],
}
REGION_STYLES = {
    "thin": dict(color=PALETTE["blue"], linestyle="-", marker="o"),
    "intermediate": dict(color=PALETTE["orange"], linestyle="--", marker="s"),
    "thick": dict(color=PALETTE["light_blue"], linestyle=":", marker="^"),
}
PROFILE_STYLES = {
    "raw": dict(color=PALETTE["blue"], linestyle="-", marker="o"),
    "adjusted": dict(color=PALETTE["orange"], linestyle="--", marker="s"),
    "length_adjusted": dict(color=PALETTE["light_blue"], linestyle=":", marker="^"),
}

VIOLIN_STYLE = {
    "fill": PALETTE["light_blue"],
    "edge": PALETTE["blue"],
    "median": PALETTE["orange"],
    "width": .38,
    "alpha": .55,
    "linewidth": 1.1,
    "box_width": .09,
    "box_face": "white",
    "box_linewidth": .9,
    "bw_adjust": .85,
}


def subject_violin(ax: plt.Axes, values, *, position=0, orientation="vertical",
                   color=None, single_group=True, showfliers=True, width=None,
                   fill_color=None, edge_color=None, alpha=None,
                   box_facecolor=None, box_edgecolor=None):
    """Draw the paper's subject-level violin and explicit Tukey boxplot.

    The caller is responsible for supplying independent statistical units. For
    paper figures these are normally one value per subject. Matplotlib clips
    its violin KDE to the observed data range, matching ``cut=0`` semantics.
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        raise ValueError("Violin plots require at least 20 independent observations")
    vertical = orientation == "vertical"
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    fill = (fill_color if fill_color is not None else
            VIOLIN_STYLE["fill"] if single_group else (color or PALETTE["blue"]))
    edge = (edge_color if edge_color is not None else
            VIOLIN_STYLE["edge"] if single_group else (color or PALETTE["reference"]))
    violin_width = (width if width is not None else
                    VIOLIN_STYLE["width"] if single_group else .55)
    violin_alpha = (alpha if alpha is not None else
                    VIOLIN_STYLE["alpha"] if single_group else .40)
    bw_method = lambda kde: kde.scotts_factor() * VIOLIN_STYLE["bw_adjust"]
    violin = ax.violinplot(
        [values], positions=[position], widths=violin_width,
        vert=vertical, showextrema=False, showmedians=False,
        bw_method=bw_method,
    )
    for body in violin["bodies"]:
        body.set_gid("subject_violin")
        # Keep an explicitly requested dark edge fully opaque while allowing
        # the violin interior to remain light.
        if fill_color is not None or edge_color is not None or alpha is not None:
            body.set_facecolor(mpl.colors.to_rgba(fill, violin_alpha))
            body.set_alpha(1)
        else:
            body.set_facecolor(fill)
            body.set_alpha(violin_alpha)
        body.set_edgecolor(edge)
        body.set_linewidth(VIOLIN_STYLE["linewidth"])
    neutral = PALETTE["reference"]
    box_edge = (box_edgecolor if box_edgecolor is not None else
                VIOLIN_STYLE["edge"] if single_group else neutral)
    box_face = (box_facecolor if box_facecolor is not None else
                VIOLIN_STYLE["box_face"])
    median = VIOLIN_STYLE["median"] if single_group else neutral
    flierprops = dict(marker="o", markersize=2.2, markerfacecolor="#555555",
                      markeredgecolor="none", alpha=.55)
    box = ax.boxplot(
        values, positions=[position], widths=VIOLIN_STYLE["box_width"],
        vert=vertical, patch_artist=True, manage_ticks=False,
        whis=1.5, showfliers=showfliers,
        boxprops=dict(facecolor=box_face, edgecolor=box_edge,
                      linewidth=VIOLIN_STYLE["box_linewidth"]),
        whiskerprops=dict(color=box_edge if single_group else "#555555", linewidth=.8),
        capprops=dict(color=box_edge if single_group else "#555555", linewidth=.8),
        medianprops=dict(color=median, linewidth=1.5),
        flierprops=flierprops,
    )
    if single_group:
        # Prevent Matplotlib's one-category autoscaling from making a width=.38
        # body occupy nearly the entire panel.
        if vertical:
            ax.set_xlim(position - .5, position + .5)
        else:
            ax.set_ylim(position - .5, position + .5)
    ax.grid(False)
    return violin, box


def apply_semantic_style() -> None:
    """Set semantic defaults without changing a figure's chosen font sizes."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "text.color": "#333333", "axes.labelcolor": "#333333",
        "axes.titlecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.edgecolor": "#555555", "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=SERIES_COLORS),
        "axes.grid": False, "axes.linewidth": .8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "legend.frameon": False, "legend.facecolor": "white",
        "legend.edgecolor": "#D0D0D0", "legend.framealpha": .90,
        "legend.fancybox": False, "legend.borderpad": .35,
        "legend.labelspacing": .35, "legend.handlelength": 1.8,
        "legend.handletextpad": .5, "lines.markeredgewidth": 0,
    })


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
            "lines.linewidth": 1.8,
            "lines.markersize": 3.0,
            "axes.prop_cycle": mpl.cycler(color=SERIES_COLORS),
            "figure.dpi": 120,
            "savefig.dpi": RASTER_DPI,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    apply_semantic_style()


def style_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    """Apply the no-grid paper style; grid is retained only for call compatibility."""
    ax.set_facecolor("white")
    ax.grid(False, which="both", axis="both")
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.8)
        ax.spines[side].set_color("#555555")
    ax.tick_params(width=.8, length=3, pad=2, direction="out", colors="#333333")


def finalize_figure(fig: plt.Figure) -> None:
    """Enforce paper style on artists, including local notebook overrides.

    No data, intervals, normalization, or axis ranges are changed. Colorbars
    keep their outlines; 3D brain views keep their geometry. Call before saving.
    """
    from matplotlib.collections import PathCollection
    import textwrap
    import re
    for ax in fig.axes:
        if ax.name != "3d" and not hasattr(ax, "_colorbar"):
            style_axes(ax)
        else:
            ax.grid(False)
        ax.xaxis.label.set(fontsize=8.5, color="#333333")
        ax.yaxis.label.set(fontsize=8.5, color="#333333")
        ax.tick_params(labelsize=7.5)
        for text in (ax.title, ax._left_title, ax._right_title):
            text.set_text(re.sub(r"^(?:[A-Da-d][.):]?|S\d+[A-Da-d])\s+", "", text.get_text()))
            text.set(fontsize=9, color="#333333")
            if ax.name != "3d" and "$" not in text.get_text():
                width = max(24, int(fig.get_figwidth() * ax.get_position().width * 72 / 4.5))
                text.set_text("\n".join(textwrap.fill(line, width=width, break_long_words=False)
                                        for line in text.get_text().split("\n")))
        for text in ax.texts:
            text.set(fontsize=min(text.get_fontsize(), 8), color="#333333")
        for line in ax.lines:
            line.set_linewidth(min(line.get_linewidth(), 1.9 if line.get_gid() == "derived_result_line" else 1.8))
            line.set_markersize(min(line.get_markersize(), 3.2))
            line.set_markeredgewidth(min(line.get_markeredgewidth(), .5))
        for collection in ax.collections:
            if ("FillBetween" in type(collection).__name__
                    and collection.get_gid() != "subject_violin"):
                collection.set_alpha(.16 if collection.get_gid() == "derived_result_band" else .12)
                collection.set_linewidth(0)
            elif isinstance(collection, PathCollection):
                collection.set_edgecolor("none")
                collection.set_sizes([min(s, 18) for s in collection.get_sizes()])
        legend = ax.get_legend()
        if legend is not None:
            if legend.get_frame_on():
                frame = legend.get_frame()
                frame.set_facecolor("white")
                frame.set_edgecolor("#D0D0D0")
                frame.set_alpha(.90)
                frame.set_linewidth(.6)
            for text in legend.get_texts():
                text.set(fontsize=7.5, color="#333333")
            for handle in legend.legend_handles:
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(3.2)
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.6)
    for legend in fig.legends:
        if legend.get_frame_on():
            frame = legend.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("#D0D0D0")
            frame.set_alpha(.90)
            frame.set_linewidth(.6)
        for text in legend.get_texts():
            text.set(fontsize=7.5, color="#333333")
    for text in fig.texts:
        size = 9 if text is fig._suptitle else min(text.get_fontsize(), 7.5)
        text.set(fontsize=size, color="#333333")
        width = max(30, int(fig.get_figwidth() * 72 * .95 / (size * .5)))
        text.set_text("\n".join(textwrap.fill(line, width=width, break_long_words=False)
                                for line in text.get_text().split("\n")))


def panel_label(ax: plt.Axes, label: str) -> None:
    """Compatibility entry point: paper figures no longer display panel letters."""
    return None


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    """Save vector and 600-dpi raster versions of a figure with one call."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig)
    for suffix in (".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    for suffix in (".png", ".tiff"):
        fig.savefig(stem.with_suffix(suffix), dpi=RASTER_DPI, bbox_inches="tight")
