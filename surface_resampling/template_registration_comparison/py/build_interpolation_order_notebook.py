#!/usr/bin/env python
"""Build the summary-only interpolation-order comparison notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def markdown(source: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> dict:
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(
        """
        # Effect of Interpolation Order on Template-Aligned Cortical MD Profiles

        ## Native-column MD value interpolation after volume sampling versus native-coordinate interpolation before volume sampling

        This notebook is summary-only. All profile loading, validity masks,
        depth-wise metrics, cross-subject metrics, fixed spatial groups, and
        difference maps are computed by the Python analysis script:

        `surface_resampling/template_registration_comparison/py/compare_o2_column_o3_coordinate.py`

        The primary comparison uses 30 common ADRC subjects, both hemispheres,
        21 cortical depths from white matter (`0.00`) to pial surface (`1.00`),
        and the `fsaverage` template.

        The notebook addresses individual-profile agreement, preservation of
        cross-subject variation, dependence on spatial group size, local
        spatial structure, and a supplementary O2-FS comparison.

        - **O2-Column:** MD-value interpolation after native-column sampling.
        - **O3-Coordinate:** Native-coordinate interpolation before MD-volume sampling.
        - **O2-FS:** Standard `mri_vol2surf` volume-resampling reference; not a
          primary method in this notebook.
        """
    ),
    markdown(
        """
        ## Load completed Python outputs

        The cells below read CSV summaries and existing map files only. They do
        not load vertex-by-depth arrays and do not repeat profile generation or
        surface mapping.
        """
    ),
    code(
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd


        PROJECT_ROOT = Path(
            "/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
        )
        RESULTS_ROOT = Path("/mnt/newStor/paros/paros_WORK/xintong/results")
        OUTPUT_ROOT = RESULTS_ROOT / "interpolation_order_comparison_fsaverage_python"
        QC_ROOT = OUTPUT_ROOT / "qc"
        METRICS_ROOT = OUTPUT_ROOT / "metrics"
        FIGURE_ROOT = OUTPUT_ROOT / "figures"
        SUMMARY_ROOT = OUTPUT_ROOT / "summary"
        SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

        paired_manifest = pd.read_csv(QC_ROOT / "paired_subject_manifest.csv")
        validation = pd.read_csv(QC_ROOT / "input_validation_summary.csv")
        cohort_depth = pd.read_csv(METRICS_ROOT / "cohort_depth_summary.csv")
        subject_depth = pd.read_csv(
            METRICS_ROOT / "subject_depth_metrics_by_subject.csv"
        )
        profile_metrics = pd.read_csv(METRICS_ROOT / "subject_profile_metrics.csv")
        cross_subject = pd.read_csv(METRICS_ROOT / "cross_subject_metrics.csv")
        variation_ratio = pd.read_csv(
            METRICS_ROOT / "method_difference_to_subject_variation.csv"
        )
        group_subject = pd.read_csv(
            METRICS_ROOT / "spatial_group_subject_metrics.csv"
        )
        group_cross = pd.read_csv(
            METRICS_ROOT / "spatial_group_cross_subject_metrics.csv"
        )
        spatial_structure = pd.read_csv(
            METRICS_ROOT / "spatial_structure_metrics.csv"
        )
        o2fs_summary = pd.read_csv(
            OUTPUT_ROOT / "supplementary/o2fs_vs_o2column_metrics.csv"
        )
        o2fs_regional = pd.read_csv(
            OUTPUT_ROOT / "supplementary/o2fs_vs_o2column_regional_metrics.csv"
        )
        decision_summary = pd.read_csv(
            SUMMARY_ROOT / "interpolation_order_summary.csv"
        )

        print(f"Output root: {OUTPUT_ROOT}")
        print(f"Subjects: {validation.subject.nunique()}")
        print(f"Subject-hemisphere QC rows: {len(validation)}")
        display(paired_manifest.head())
        """
    ),
    markdown(
        """
        ## Input and paired-validity summary

        Only vertices with valid O2-Column and O3-Coordinate profiles, valid
        mapping, finite sampled values, and in-bounds O3 coordinates are used
        in the primary comparison.
        """
    ),
    code(
        """
        validation_summary = pd.DataFrame(
            {
                "subjects": [validation.subject.nunique()],
                "subject_hemisphere_pairs": [len(validation)],
                "mean_common_valid_percent": [
                    validation.common_valid_pct.mean()
                ],
                "min_common_valid_percent": [
                    validation.common_valid_pct.min()
                ],
                "mean_o3_raw_in_bounds_percent": [
                    validation.o3_raw_in_bounds_pct.mean()
                ],
                "mean_o3_sampled_finite_percent": [
                    validation.o3_sampled_finite_pct.mean()
                ],
            }
        ).T.rename(columns={0: "value"})
        display(validation_summary.round(3))
        display(
            paired_manifest[
                [
                    "subject",
                    "hemisphere",
                    "o2_available",
                    "o3_available",
                    "same_source_vertex_ids",
                    "same_barycentric_weights",
                    "included",
                ]
            ].head()
        )
        """
    ),
    markdown(
        """
        ## Figure 1: Depth-Wise Agreement Between MD-Value and Native-Coordinate Interpolation

        The cohort curve is the median of subject-level summaries. The shaded
        region is the subject-level interquartile range. RMSE remains available
        in the saved table but is not used in the primary figures.
        """
    ),
    code(
        """
        def save_figure(fig, name):
            FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURE_ROOT / f"{name}.png", dpi=300, bbox_inches="tight")
            fig.savefig(FIGURE_ROOT / f"{name}.pdf", bbox_inches="tight")
            plt.show()


        x = cohort_depth.depth * 100
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        figure_1_metrics = [
            ("pearson_r", "Pearson correlation", "#0072B2"),
            ("snmae_pct", "Symmetric NMAE (%)", "#D55E00"),
            ("mean_bias_o3_minus_o2", "Signed bias (O3 - O2)", "#009E73"),
        ]
        for ax, (metric, label, color) in zip(axes, figure_1_metrics):
            ax.plot(x, cohort_depth[f"{metric}_median"], color=color)
            ax.fill_between(
                x,
                cohort_depth[f"{metric}_q25"],
                cohort_depth[f"{metric}_q75"],
                color=color,
                alpha=0.18,
            )
            ax.set_xlabel("Cortical depth (%)")
            ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "Depth-Wise Agreement Between MD-Value and Native-Coordinate Interpolation",
            fontsize=10,
        )
        fig.tight_layout()
        save_figure(fig, "depth_wise_agreement")
        """
    ),
    markdown(
        """
        ## Figure 2: Preservation of Cross-Subject MD Variation Across Interpolation Orders

        The ratio is

        `median_subject(abs(F3 - F2)) / SD_subject((F2 + F3) / 2)`.

        It is an interpretation scale for method difference relative to
        between-subject variation, not a significance threshold.
        """
    ),
    code(
        """
        cross_summary = cross_subject.groupby("depth", as_index=False).agg(
            correlation_median=("correlation_median", "median"),
            correlation_q25=("correlation_median", lambda values: values.quantile(0.25)),
            correlation_q75=("correlation_median", lambda values: values.quantile(0.75)),
            pct_below_0_8=("pct_below_0_8", "median"),
            pct_below_0_9=("pct_below_0_9", "median"),
        )
        ratio_summary = variation_ratio.groupby("depth", as_index=False).agg(
            ratio_median=("ratio_median", "median"),
            ratio_q25=("ratio_median", lambda values: values.quantile(0.25)),
            ratio_q75=("ratio_median", lambda values: values.quantile(0.75)),
        )
        x = cross_summary.depth * 100
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        axes[0].plot(x, cross_summary.correlation_median, color="#0072B2")
        axes[0].fill_between(
            x,
            cross_summary.correlation_q25,
            cross_summary.correlation_q75,
            color="#0072B2",
            alpha=0.18,
        )
        axes[0].set_ylabel("Cross-subject correlation")
        axes[1].plot(x, ratio_summary.ratio_median, color="#D55E00")
        axes[1].fill_between(
            x,
            ratio_summary.ratio_q25,
            ratio_summary.ratio_q75,
            color="#D55E00",
            alpha=0.18,
        )
        axes[1].set_ylabel("Method difference / subject variation")
        axes[2].plot(x, cross_summary.pct_below_0_8, label="r < 0.8", color="#009E73")
        axes[2].plot(x, cross_summary.pct_below_0_9, label="r < 0.9", color="#CC79A7")
        axes[2].set_ylabel("Vertices below threshold (%)")
        axes[2].legend(frameon=False, fontsize=7)
        for ax in axes:
            ax.set_xlabel("Cortical depth (%)")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "Preservation of Cross-Subject MD Variation Across Interpolation Orders",
            fontsize=10,
        )
        fig.tight_layout()
        save_figure(fig, "cross_subject_preservation")
        """
    ),
    markdown(
        """
        ## Figure 3: Dependence of Interpolation-Order Differences on Spatial Group Size

        Spatial group membership was generated from template-surface adjacency
        within annotation regions and is shared by both methods. The regional
        mean is shown as a separate category.
        """
    ),
    code(
        """
        group_summary = group_subject.groupby("group_size", as_index=False).agg(
            profile_correlation_median=("profile_correlation_median", "median"),
            profile_correlation_q25=("profile_correlation_median", lambda values: values.quantile(0.25)),
            profile_correlation_q75=("profile_correlation_median", lambda values: values.quantile(0.75)),
            profile_snmae_median=("profile_snmae_median", "median"),
            profile_snmae_q25=("profile_snmae_median", lambda values: values.quantile(0.25)),
            profile_snmae_q75=("profile_snmae_median", lambda values: values.quantile(0.75)),
        )
        cross_group_summary = group_cross.groupby(
            "group_size", as_index=False
        ).agg(
            cross_subject_correlation_median=(
                "cross_subject_correlation_median",
                "median",
            ),
            cross_subject_correlation_q25=(
                "cross_subject_correlation_median",
                lambda values: values.quantile(0.25),
            ),
            cross_subject_correlation_q75=(
                "cross_subject_correlation_median",
                lambda values: values.quantile(0.75),
            ),
        )
        group_plot = group_summary.merge(cross_group_summary, on="group_size")
        group_order = ["1", "50", "100", "200", "400", "regional_mean"]
        group_plot["group_size"] = pd.Categorical(
            group_plot.group_size,
            categories=group_order,
            ordered=True,
        )
        group_plot = group_plot.sort_values("group_size")
        x = range(len(group_plot))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        panels = [
            ("profile_correlation", "Profile correlation", "#0072B2"),
            ("profile_snmae", "Symmetric NMAE (%)", "#D55E00"),
            ("cross_subject_correlation", "Cross-subject correlation", "#009E73"),
        ]
        for ax, (prefix, label, color) in zip(axes, panels):
            median = group_plot[f"{prefix}_median"]
            q25 = group_plot[f"{prefix}_q25"]
            q75 = group_plot[f"{prefix}_q75"]
            ax.plot(list(x), median, marker="o", color=color)
            ax.fill_between(list(x), q25, q75, color=color, alpha=0.18)
            ax.set_ylabel(label)
            ax.set_xticks(list(x))
            ax.set_xticklabels(group_plot.group_size)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes:
            ax.set_xlabel("Spatial group size")
        fig.suptitle(
            "Dependence of Interpolation-Order Differences on Spatial Group Size",
            fontsize=10,
        )
        fig.tight_layout()
        save_figure(fig, "group_size_comparison")
        """
    ),
    markdown(
        """
        ## 50% depth difference maps

        These are subject-level median maps of `O3-Coordinate - O2-Column` at
        50% cortical depth. They are generated by Python and can be loaded as
        generic FreeSurfer overlays on the corresponding `fsaverage` surface.
        """
    ),
    code(
        """
        difference_maps = sorted(FIGURE_ROOT.glob("*depth50_o3_minus_o2.mgh"))
        display(pd.DataFrame({"freeview_overlay": [str(path) for path in difference_maps]}))
        """
    ),
    markdown(
        """
        ## Preservation of Local Spatial Structure

        Adjacent vertices are defined by edges of the fsaverage white-surface
        mesh. The supplementary nearby-versus-distant comparison uses the
        fixed deterministic pair sampling recorded by the Python script.
        Lower adjacent-vertex difference means a smoother map; it is not by
        itself evidence of greater accuracy.
        """
    ),
    code(
        """
        spatial_summary = spatial_structure.groupby("depth", as_index=False).agg(
            o2_adjacent=("o2_adjacent_difference_median", "median"),
            o3_adjacent=("o3_adjacent_difference_median", "median"),
            o2_near_minus_distant=("nearby_minus_distant_o2", "median"),
            o3_near_minus_distant=("nearby_minus_distant_o3", "median"),
        )
        x = spatial_summary.depth * 100
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
        axes[0].plot(x, spatial_summary.o2_adjacent, label="O2-Column")
        axes[0].plot(x, spatial_summary.o3_adjacent, label="O3-Coordinate")
        axes[0].set_ylabel("Median adjacent-vertex MD difference")
        axes[0].legend(frameon=False, fontsize=7)
        axes[1].plot(
            x,
            spatial_summary.o2_near_minus_distant,
            label="O2-Column",
        )
        axes[1].plot(
            x,
            spatial_summary.o3_near_minus_distant,
            label="O3-Coordinate",
        )
        axes[1].set_ylabel("Nearby - distant profile correlation")
        axes[1].legend(frameon=False, fontsize=7)
        for ax in axes:
            ax.set_xlabel("Cortical depth (%)")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "Preservation of Local Spatial Relationships Across Interpolation Orders",
            fontsize=10,
        )
        fig.tight_layout()
        save_figure(fig, "spatial_structure_preservation")
        """
    ),
    markdown(
        """
        ## Supplementary Comparison With Standard FreeSurfer Surface Resampling

        O2-FS is a sensitivity baseline only. It is not included in the main
        O2-Column versus O3-Coordinate decision.
        """
    ),
    code(
        """
        x = o2fs_summary.depth * 100
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
        axes[0].plot(x, o2fs_summary.pearson_r_median, color="#0072B2")
        axes[0].fill_between(
            x,
            o2fs_summary.pearson_r_q25,
            o2fs_summary.pearson_r_q75,
            color="#0072B2",
            alpha=0.18,
        )
        axes[0].set_ylabel("O2-FS vs O2-Column correlation")
        axes[1].plot(x, o2fs_summary.snmae_pct_median, color="#D55E00")
        axes[1].fill_between(
            x,
            o2fs_summary.snmae_pct_q25,
            o2fs_summary.snmae_pct_q75,
            color="#D55E00",
            alpha=0.18,
        )
        axes[1].set_ylabel("Symmetric NMAE (%)")
        for ax in axes:
            ax.set_xlabel("Cortical depth (%)")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "Supplementary Comparison With Standard FreeSurfer Surface Resampling",
            fontsize=10,
        )
        fig.tight_layout()
        save_figure(fig, "o2fs_vs_o2column")
        display(o2fs_regional.head())
        """
    ),
    markdown(
        """
        ## Saved tables and change log

        Reused: existing O2-Column/O3-Coordinate profiles, validated target
        mappings, and the common 30-subject manifest.

        Recalculated in Python: common validity QC, depth-wise Pearson/MAE/
        RMSE/bias/sNMAE, profile metrics, cross-subject metrics, the updated
        method-difference ratio, fixed-group metrics, and 50% difference maps.

        Skipped here: all volume sampling, surface mapping, profile generation,
        and other expensive vertex-level calculations. Priority-2 analyses such
        as O2-FS sensitivity and additional spatial-smoothness summaries remain
        supplementary and do not block the primary comparison.

        The previous notebook with in-cell calculations is preserved as
        `surface_resampling/analysis/archive/compare_md_interpolation_order_fsaverage.pre_python_backup.ipynb` and
        must not be used as the final summary notebook.
        """
    ),
    code(
        """
        summary_files = sorted(METRICS_ROOT.glob("*.csv"))
        display(decision_summary)
        display(pd.DataFrame({"saved_summary_table": [str(path) for path in summary_files]}))
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)

output = Path(
    "surface_resampling/template_registration_comparison/notebooks/compare_md_interpolation_order_fsaverage.ipynb"
)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    nbf.write(notebook, handle)
print(f"Wrote {output}")
