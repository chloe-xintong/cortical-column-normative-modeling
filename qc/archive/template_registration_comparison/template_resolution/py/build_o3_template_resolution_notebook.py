#!/usr/bin/env python3
"""Build the summary-only O3 fsaverage versus fsaverage6 notebook.

Description
-----------
Build the summary-only O3 fsaverage versus fsaverage6 notebook.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python qc/archive/template_registration_comparison/template_resolution/py/build_o3_template_resolution_notebook.py --help
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def markdown(source: str) -> dict:
    """Create a markdown notebook cell."""

    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> dict:
    """Create a code notebook cell."""

    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(
        """
        # O3-Coordinate Template Resolution Comparison

        ## Does fsaverage retain non-redundant local MD detail relative to fsaverage6?

        This notebook is summary-only. It reads the existing
        `O3-Coordinate` `md_profiles.npz` files for both `fsaverage` and
        `fsaverage6`. It does not run `mri_vol2surf`, construct new columns, or
        resample an MD volume.

        The fixed correspondence maps each fsaverage vertex to the nearest
        fsaverage6 vertex after normalizing the two `sphere.reg` surfaces to
        unit directions. Spatial groups use the same approximate white-surface
        area budget in both templates; the actual vertex count is reported for
        each template.

        The default analysis uses the existing 30-subject paired manifest
        `manifests/method_comparison_common_30_subjects.csv`.

        Main questions:

        1. Does full-resolution fsaverage contain meaningful within-cell MD
           variation that is absent from fsaverage6?
        2. Does that extra variation affect area-matched spatial-group results?
        """
    ),
    markdown(
        """
        ## Reproducible calculation command

        Run the calculation-heavy step on a compute node:

        ```bash
        python -u qc/archive/template_registration_comparison/template_resolution/py/compare_o3_template_resolution.py \\
            --output-root \
                /mnt/newStor/paros/paros_WORK/xintong/results/o3_template_resolution_comparison
        ```

        The calculation uses only existing O3 profiles and writes tables under
        `tables/`. This notebook reads those tables and produces figures.
        """
    ),
    code(
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns


        RESULTS_ROOT = Path("/mnt/newStor/paros/paros_WORK/xintong/results")
        OUTPUT_ROOT = RESULTS_ROOT / "o3_template_resolution_comparison"
        TABLE_ROOT = OUTPUT_ROOT / "tables"
        FIGURE_ROOT = OUTPUT_ROOT / "figures"
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

        DEPTHS = np.linspace(0.0, 1.0, 21)
        DEPTH_LABELS = [f"{depth:.0%}" for depth in DEPTHS]
        TEMPLATE_ORDER = ["fsaverage", "fsaverage6"]
        TEMPLATE_LABELS = {
            "fsaverage": "fsaverage",
            "fsaverage6": "fsaverage6",
        }
        PALETTE = {
            "fsaverage": "#0072B2",
            "fsaverage6": "#D55E00",
        }

        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": 120,
            }
        )


        def save_figure(figure, name):
            # Save publication-ready PNG and PDF versions.

            figure.savefig(
                FIGURE_ROOT / f"{name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            figure.savefig(
                FIGURE_ROOT / f"{name}.pdf",
                bbox_inches="tight",
            )


        correspondence = pd.read_csv(TABLE_ROOT / "template_correspondence_qc.csv")
        cell = pd.read_csv(TABLE_ROOT / "template_cell_variation.csv")
        adjacent = pd.read_csv(TABLE_ROOT / "adjacent_profile_similarity.csv")
        group = pd.read_csv(TABLE_ROOT / "area_matched_group_metrics.csv")
        cross_subject = pd.read_csv(
            TABLE_ROOT / "area_matched_cross_subject_variability.csv"
        )

        print("Correspondence QC")
        display(correspondence)
        print(f"Subjects: {cell.subject.nunique()}")
        print(f"Hemisphere rows: {cell.hemisphere.nunique()}")
        print(f"Depths: {cell.depth.nunique()}")
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
        midpoint = cell.groupby("depth", as_index=False).agg(
            within_between_sd_ratio=("within_between_sd_ratio", "median"),
            target_vs_cellmean_correlation=(
                "target_vs_cellmean_correlation",
                "median",
            ),
            target_vs_cellmean_mae=("target_vs_cellmean_mae", "median"),
        )

        axes[0].plot(
            midpoint.depth,
            midpoint.within_between_sd_ratio,
            color=PALETTE["fsaverage"],
            marker="o",
            markersize=3,
        )
        axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Cortical depth")
        axes[0].set_ylabel("Within-cell / between-cell SD")
        axes[0].set_title("Full-resolution variation within fsaverage6 cells")
        axes[0].set_xticks(DEPTHS[::4])
        axes[0].set_xticklabels(DEPTH_LABELS[::4])

        axes[1].plot(
            midpoint.depth,
            midpoint.target_vs_cellmean_correlation,
            color=PALETTE["fsaverage6"],
            marker="o",
            markersize=3,
            label="Profile correlation",
        )
        axes[1].set_xlabel("Cortical depth")
        axes[1].set_ylabel("Correlation")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("fsaverage cell mean vs direct fsaverage6 O3")
        axes[1].set_xticks(DEPTHS[::4])
        axes[1].set_xticklabels(DEPTH_LABELS[::4])
        save_figure(fig, "01_cell_variation_and_template_agreement")
        plt.show()
        """
    ),
    code(
        """
        adjacent_plot = adjacent.copy()
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
        sns.boxplot(
            data=adjacent_plot,
            x="relation",
            y="profile_correlation_median",
            color="#D9EAF2",
            showfliers=False,
            ax=axes[0],
        )
        sns.stripplot(
            data=adjacent_plot,
            x="relation",
            y="profile_correlation_median",
            color="#0072B2",
            size=2,
            alpha=0.25,
            ax=axes[0],
        )
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Adjacent 21-point profile correlation")
        axes[0].set_title("Adjacent fsaverage vertices")

        sns.boxplot(
            data=adjacent_plot,
            x="relation",
            y="profile_mae_median",
            color="#F6D7C3",
            showfliers=False,
            ax=axes[1],
        )
        sns.stripplot(
            data=adjacent_plot,
            x="relation",
            y="profile_mae_median",
            color="#D55E00",
            size=2,
            alpha=0.25,
            ax=axes[1],
        )
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Adjacent profile MAE")
        axes[1].set_title("Within-cell versus between-cell edges")
        save_figure(fig, "02_adjacent_profile_similarity")
        plt.show()
        """
    ),
    code(
        """
        depth50 = group[group.depth.eq(0.5)].copy()
        summary = depth50.groupby(
            ["template", "requested_group_size"],
            as_index=False,
        ).agg(
            group_sd_median=("within_template_group_sd", "median"),
            group_sd_q25=(
                "within_template_group_sd",
                lambda values: values.quantile(0.25),
            ),
            group_sd_q75=(
                "within_template_group_sd",
                lambda values: values.quantile(0.75),
            ),
            actual_vertices_median=("median_group_vertex_count", "median"),
        )

        cross50 = cross_subject[cross_subject.depth.eq(0.5)].copy()
        cross_summary = cross50.groupby(
            ["template", "requested_group_size"],
            as_index=False,
        ).agg(cross_subject_sd_median=("cross_subject_sd_median", "median"))
        summary = summary.merge(
            cross_summary,
            on=["template", "requested_group_size"],
            how="left",
        )

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
        for template in TEMPLATE_ORDER:
            part = summary[summary.template.eq(template)].sort_values(
                "requested_group_size"
            )
            axes[0].plot(
                part.requested_group_size,
                part.group_sd_median,
                marker="o",
                color=PALETTE[template],
                label=TEMPLATE_LABELS[template],
            )
            axes[1].plot(
                part.requested_group_size,
                part.cross_subject_sd_median,
                marker="o",
                color=PALETTE[template],
                label=TEMPLATE_LABELS[template],
            )
        for ax in axes:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel("Area-equivalent group size (fsaverage vertices)")
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("Within-template group SD at 50% depth")
        axes[1].set_ylabel("Cross-subject group SD at 50% depth")
        axes[0].set_title("Spatial-group variation")
        axes[1].set_title("Cross-subject variability")
        save_figure(fig, "03_area_matched_group_comparison_depth50")
        plt.show()

        display(
            summary[
                [
                    "template",
                    "requested_group_size",
                    "actual_vertices_median",
                    "group_sd_median",
                    "cross_subject_sd_median",
                ]
            ]
        )
        """
    ),
    code(
        """
        heatmap_data = (
            group.groupby(["template", "requested_group_size", "depth"], as_index=False)
            .within_template_group_sd.median()
            .rename(columns={"within_template_group_sd": "group_sd"})
        )
        fig, axes = plt.subplots(
            1,
            len(TEMPLATE_ORDER),
            figsize=(8.4, 3.8),
            constrained_layout=True,
            sharey=True,
        )
        for ax, template in zip(np.atleast_1d(axes), TEMPLATE_ORDER):
            matrix = heatmap_data[heatmap_data.template.eq(template)].pivot(
                index="requested_group_size",
                columns="depth",
                values="group_sd",
            )
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="viridis",
                cbar=ax is axes[-1] if isinstance(axes, np.ndarray) else True,
                xticklabels=DEPTH_LABELS,
                yticklabels=True,
            )
            ax.set_title(template)
            ax.set_xlabel("Cortical depth")
            ax.set_xticks(np.arange(0.5, len(DEPTHS), 4))
            ax.set_xticklabels(DEPTH_LABELS[::4], rotation=45, ha="right")
            ax.set_ylabel("Area-equivalent group size")
        save_figure(fig, "04_group_variation_depth_heatmap")
        plt.show()
        """
    ),
    markdown(
        """
        ## Interpretation guide

        - A low within-cell / between-cell ratio indicates that the detail
          retained by fsaverage is small relative to variation between
          fsaverage6-scale units.
        - High adjacent profile correlation within cells supports the idea that
          the additional fsaverage vertices are largely redundant at the
          fsaverage6 scale.
        - The group comparison should be interpreted using the reported actual
          vertex counts and physical area summaries, not only the requested
          group-size label.
        - These are O3-Coordinate results only. They do not compare O2-Column,
          O2-FS, or Native → fsaverage → fsaverage6 resampling.
        """
    ),
]


def main() -> None:
    output = Path(
        "qc/archive/template_registration_comparison/template_resolution/notebooks/compare_o3_template_resolution.ipynb"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
