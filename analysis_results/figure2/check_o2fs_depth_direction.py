#!/usr/bin/env python3

"""Check O2-FS depth direction and quantify method-comparison scale."""

import argparse
import gc
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_method_comparison import (
    CANONICAL_DEPTHS,
    build_pair_data,
    load_cortex_mask,
    load_method_profiles,
    profile_correlation,
)


COMPARISONS = {
    "O2Column_vs_O3": "O2-Column",
    "O2FS_vs_O3": "O2-FS",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-file", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--freesurfer-subjects-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--representative-vertices", type=int, default=20)
    return parser.parse_args()


def read_subjects(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def symmetric_nmae(method_a, method_b):
    mae = np.mean(np.abs(method_b - method_a))
    denominator = np.median(np.abs(method_a)) + np.median(np.abs(method_b))
    return 200.0 * mae / denominator


def audit_o2fs_manifests(args, subjects):
    rows = []
    for subject in subjects:
        for hemisphere in ("lh", "rh"):
            path = (
                args.results_root
                / "qc"
                / "o2_fs"
                / subject
                / args.template
                / hemisphere
                / "vol2surf_manifest.json"
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            outputs = manifest["outputs"]
            o3_path = (
                args.results_root
                / "fsaverage_result"
                / "columns"
                / subject
                / "profiles"
                / f"{subject}_{hemisphere}_md_profiles.npz"
            )
            with np.load(o3_path, allow_pickle=True) as data:
                o3_metric_image = str(data["metric_image"].item())
                o3_registration = str(data["registration"].item())
                o3_direction = str(data["depth_direction"].item())
                target_vertex_id = np.asarray(
                    data["target_vertex_id"],
                    dtype=np.int64,
                )
            production_transform = (
                args.results_root
                / "fsaverage_result"
                / "columns"
                / subject
                / "registration"
                / f"{subject}_DWI2T1.dat"
            )
            rows.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "manifest_path": str(path),
                    "depth_direction": manifest["depth_direction"],
                    "projection_definition": manifest["projection_definition"],
                    "n_depths": len(outputs),
                    "first_depth": outputs[0]["depth"],
                    "first_projfrac": outputs[0]["projfrac"],
                    "middle_depth": outputs[10]["depth"],
                    "middle_projfrac": outputs[10]["projfrac"],
                    "last_depth": outputs[-1]["depth"],
                    "last_projfrac": outputs[-1]["projfrac"],
                    "o3_depth_direction": o3_direction,
                    "same_depth_direction": (
                        manifest["depth_direction"] == o3_direction
                    ),
                    "o2fs_metric_image": manifest["metric_image"],
                    "o3_metric_image": o3_metric_image,
                    "same_metric_image": manifest["metric_image"]
                    == o3_metric_image,
                    "o2fs_registration": manifest["registration"],
                    "o3_registration_metadata": o3_registration,
                    "production_transform": str(production_transform),
                    "o2fs_uses_production_transform": (
                        Path(manifest["registration"]).resolve()
                        == production_transform.resolve()
                    ),
                    "n_o3_target_vertices": target_vertex_id.size,
                    "o3_target_vertex_ids_unique": (
                        np.unique(target_vertex_id).size
                        == target_vertex_id.size
                    ),
                    "o3_target_vertex_id_min": target_vertex_id.min(),
                    "o3_target_vertex_id_max": target_vertex_id.max(),
                }
            )
    return pd.DataFrame(rows)


def summarize_profile_pair(method_a, method_b, valid):
    a = method_a[valid]
    b = method_b[valid]
    profile_r = profile_correlation(a, b)
    profile_mae = np.mean(np.abs(b - a), axis=1)
    return {
        "n_vertices": int(valid.sum()),
        "median_profile_r": np.nanmedian(profile_r),
        "q25_profile_r": np.nanquantile(profile_r, 0.25),
        "q75_profile_r": np.nanquantile(profile_r, 0.75),
        "median_profile_mae": np.nanmedian(profile_mae),
        "mean_profile_mae": np.nanmean(profile_mae),
        "symmetric_nmae_pct": symmetric_nmae(a, b),
    }


def collect_subject_diagnostics(args, subjects, cortex_masks):
    direction_rows = []
    endpoint_rows = []
    relative_rows = []
    depth_rows = []

    for subject_index, subject in enumerate(subjects, start=1):
        for hemisphere in ("lh", "rh"):
            cortex = cortex_masks[hemisphere]
            o3 = load_method_profiles(
                args.results_root,
                "O3",
                subject,
                hemisphere,
                args.template,
                cortex.size,
            )
            o2fs = load_method_profiles(
                args.results_root,
                "O2-FS",
                subject,
                hemisphere,
                args.template,
                cortex.size,
            )
            o2column = load_method_profiles(
                args.results_root,
                "O2-Column",
                subject,
                hemisphere,
                args.template,
                cortex.size,
            )

            fs_pair = build_pair_data(o2fs, o3, cortex)
            column_pair = build_pair_data(o2column, o3, cortex)
            fs_valid = fs_pair.pairwise_valid
            column_valid = column_pair.pairwise_valid

            for orientation, fs_profiles in (
                ("original", o2fs.profiles),
                ("reversed", o2fs.profiles[:, ::-1]),
            ):
                row = {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "orientation": orientation,
                }
                row.update(
                    summarize_profile_pair(fs_profiles, o3.profiles, fs_valid)
                )
                direction_rows.append(row)

            endpoint_rows.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "n_vertices": int(fs_valid.sum()),
                    "pial_matched_mae": np.mean(
                        np.abs(
                            o2fs.profiles[fs_valid, 0]
                            - o3.profiles[fs_valid, 0]
                        )
                    ),
                    "pial_swapped_mae": np.mean(
                        np.abs(
                            o2fs.profiles[fs_valid, 0]
                            - o3.profiles[fs_valid, -1]
                        )
                    ),
                    "middle_matched_mae": np.mean(
                        np.abs(
                            o2fs.profiles[fs_valid, 10]
                            - o3.profiles[fs_valid, 10]
                        )
                    ),
                    "white_matched_mae": np.mean(
                        np.abs(
                            o2fs.profiles[fs_valid, -1]
                            - o3.profiles[fs_valid, -1]
                        )
                    ),
                    "white_swapped_mae": np.mean(
                        np.abs(
                            o2fs.profiles[fs_valid, -1]
                            - o3.profiles[fs_valid, 0]
                        )
                    ),
                }
            )

            for comparison, method_a, valid in (
                ("O2Column_vs_O3", o2column.profiles, column_valid),
                ("O2FS_vs_O3", o2fs.profiles, fs_valid),
            ):
                summary = summarize_profile_pair(method_a, o3.profiles, valid)
                relative_rows.append(
                    {
                        "subject_id": subject,
                        "hemisphere": hemisphere,
                        "comparison": comparison,
                        **summary,
                    }
                )
                for depth_index, depth in enumerate(CANONICAL_DEPTHS):
                    a = method_a[valid, depth_index]
                    b = o3.profiles[valid, depth_index]
                    depth_rows.append(
                        {
                            "subject_id": subject,
                            "hemisphere": hemisphere,
                            "comparison": comparison,
                            "depth": depth,
                            "n_vertices": int(valid.sum()),
                            "mae": np.mean(np.abs(b - a)),
                            "symmetric_nmae_pct": symmetric_nmae(a, b),
                        }
                    )

        logging.info(
            "Subject diagnostics %d/%d: %s",
            subject_index,
            len(subjects),
            subject,
        )

    return (
        pd.DataFrame(direction_rows),
        pd.DataFrame(endpoint_rows),
        pd.DataFrame(relative_rows),
        pd.DataFrame(depth_rows),
    )


def cohort_summary(table, group_columns, value_columns):
    aggregations = {}
    for column in value_columns:
        aggregations[f"median_{column}"] = (column, "median")
        aggregations[f"q25_{column}"] = (
            column,
            lambda values: values.quantile(0.25),
        )
        aggregations[f"q75_{column}"] = (
            column,
            lambda values: values.quantile(0.75),
        )
    return table.groupby(group_columns, as_index=False).agg(
        n_subjects=("subject_id", "nunique"),
        **aggregations,
    )


def summarize_direction_diagnostic(direction_table):
    cohort = cohort_summary(
        direction_table,
        ["orientation"],
        [
            "median_profile_r",
            "median_profile_mae",
            "symmetric_nmae_pct",
        ],
    )
    paired = direction_table.pivot(
        index=["subject_id", "hemisphere"],
        columns="orientation",
        values=["median_profile_r", "median_profile_mae"],
    )
    diagnostic = pd.DataFrame(
        [
            {
                "n_subject_hemispheres": len(paired),
                "fraction_original_higher_profile_r": np.mean(
                    paired["median_profile_r"]["original"]
                    > paired["median_profile_r"]["reversed"]
                ),
                "fraction_original_lower_profile_mae": np.mean(
                    paired["median_profile_mae"]["original"]
                    < paired["median_profile_mae"]["reversed"]
                ),
                "median_original_minus_reversed_profile_r": np.median(
                    paired["median_profile_r"]["original"]
                    - paired["median_profile_r"]["reversed"]
                ),
                "median_original_minus_reversed_profile_mae": np.median(
                    paired["median_profile_mae"]["original"]
                    - paired["median_profile_mae"]["reversed"]
                ),
            }
        ]
    )
    return cohort, diagnostic


def summarize_endpoints(endpoint_table):
    value_columns = [
        "pial_matched_mae",
        "pial_swapped_mae",
        "middle_matched_mae",
        "white_matched_mae",
        "white_swapped_mae",
    ]
    return cohort_summary(endpoint_table, ["hemisphere"], value_columns)


def load_comparison_arrays(
    args,
    subjects,
    hemisphere,
    cortex,
    method_a_name,
):
    shape = (len(subjects), cortex.size, CANONICAL_DEPTHS.size)
    method_a_array = np.empty(shape, dtype=np.float32)
    o3_array = np.empty(shape, dtype=np.float32)

    for subject_index, subject in enumerate(subjects):
        method_a = load_method_profiles(
            args.results_root,
            method_a_name,
            subject,
            hemisphere,
            args.template,
            cortex.size,
        )
        o3 = load_method_profiles(
            args.results_root,
            "O3",
            subject,
            hemisphere,
            args.template,
            cortex.size,
        )
        pair = build_pair_data(method_a, o3, cortex)
        method_a_array[subject_index] = method_a.profiles
        o3_array[subject_index] = o3.profiles
        method_a_array[subject_index, ~pair.pairwise_valid] = np.nan
        o3_array[subject_index, ~pair.pairwise_valid] = np.nan

    return method_a_array, o3_array


def cross_subject_metrics(method_a, o3, chunk_size):
    n_vertices = method_a.shape[1]
    n_depths = method_a.shape[2]
    ratio = np.full((n_vertices, n_depths), np.nan, dtype=np.float32)
    correlation = np.full((n_vertices, n_depths), np.nan, dtype=np.float32)
    n_valid = np.zeros((n_vertices, n_depths), dtype=np.int16)

    for start in range(0, n_vertices, chunk_size):
        stop = min(start + chunk_size, n_vertices)
        a = method_a[:, start:stop, :].astype(np.float64)
        b = o3[:, start:stop, :].astype(np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        count = valid.sum(axis=0)
        n_valid[start:stop] = count

        a[~valid] = np.nan
        b[~valid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            difference = np.nanmedian(np.abs(b - a), axis=0)
            subject_sd = np.nanstd((a + b) / 2.0, axis=0, ddof=1)
        chunk_ratio = np.divide(
            difference,
            subject_sd,
            out=np.full_like(difference, np.nan),
            where=subject_sd > 0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_a = np.nanmean(a, axis=0)
            mean_b = np.nanmean(b, axis=0)
        centered_a = a - mean_a
        centered_b = b - mean_b
        numerator = np.nansum(centered_a * centered_b, axis=0)
        denominator = np.sqrt(
            np.nansum(centered_a**2, axis=0)
            * np.nansum(centered_b**2, axis=0)
        )
        chunk_correlation = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=(denominator > 0) & (count >= 3),
        )

        ratio[start:stop] = chunk_ratio
        correlation[start:stop] = chunk_correlation

    return ratio, correlation, n_valid


def summarize_cross_subject_arrays(
    comparison,
    hemisphere,
    ratio,
    correlation,
    n_valid,
    minimum_subjects,
):
    rows = []
    for depth_index, depth in enumerate(CANONICAL_DEPTHS):
        eligible = n_valid[:, depth_index] >= minimum_subjects
        ratio_values = ratio[eligible, depth_index]
        correlation_values = correlation[eligible, depth_index]
        ratio_values = ratio_values[np.isfinite(ratio_values)]
        correlation_values = correlation_values[np.isfinite(correlation_values)]
        rows.append(
            {
                "comparison": comparison,
                "hemisphere": hemisphere,
                "depth": depth,
                "minimum_subjects_required": minimum_subjects,
                "n_vertices_ratio": ratio_values.size,
                "median_method_difference_subject_variation_ratio": np.median(
                    ratio_values
                ),
                "q25_method_difference_subject_variation_ratio": np.quantile(
                    ratio_values, 0.25
                ),
                "q75_method_difference_subject_variation_ratio": np.quantile(
                    ratio_values, 0.75
                ),
                "pct_ratio_at_least_0_5": 100.0 * np.mean(ratio_values >= 0.5),
                "pct_ratio_at_least_1": 100.0 * np.mean(ratio_values >= 1.0),
                "n_vertices_cross_subject_r": correlation_values.size,
                "median_cross_subject_r": np.median(correlation_values),
                "q25_cross_subject_r": np.quantile(correlation_values, 0.25),
                "q75_cross_subject_r": np.quantile(correlation_values, 0.75),
                "pct_cross_subject_r_below_0_8": 100.0
                * np.mean(correlation_values < 0.8),
                "pct_cross_subject_r_below_0_9": 100.0
                * np.mean(correlation_values < 0.9),
            }
        )
    return pd.DataFrame(rows)


def calculate_cross_subject_results(args, subjects, cortex_masks):
    summaries = []
    for comparison, method_a_name in COMPARISONS.items():
        for hemisphere in ("lh", "rh"):
            logging.info("Cross-subject metrics: %s %s", comparison, hemisphere)
            method_a, o3 = load_comparison_arrays(
                args,
                subjects,
                hemisphere,
                cortex_masks[hemisphere],
                method_a_name,
            )
            ratio, correlation, n_valid = cross_subject_metrics(
                method_a,
                o3,
                args.chunk_size,
            )
            np.savez_compressed(
                args.output_dir
                / f"{comparison}_{hemisphere}_cross_subject_metrics.npz",
                target_vertex_id=np.arange(ratio.shape[0], dtype=np.int64),
                depths=CANONICAL_DEPTHS,
                method_difference_subject_variation_ratio=ratio,
                cross_subject_correlation=correlation,
                n_subjects_valid=n_valid,
            )
            summaries.append(
                summarize_cross_subject_arrays(
                    comparison,
                    hemisphere,
                    ratio,
                    correlation,
                    n_valid,
                    int(np.ceil(0.80 * len(subjects))),
                )
            )
            del method_a, o3, ratio, correlation, n_valid
            gc.collect()
    return pd.concat(summaries, ignore_index=True)


def select_representative_subject(direction_table):
    original = direction_table.loc[direction_table["orientation"].eq("original")]
    by_subject = original.groupby("subject_id")["median_profile_r"].median()
    cohort_median = by_subject.median()
    return (by_subject - cohort_median).abs().idxmin()


def representative_profiles(args, subject, cortex_masks):
    values = []
    panels = []
    n_per_hemisphere = args.representative_vertices // 2

    for hemisphere in ("lh", "rh"):
        cortex = cortex_masks[hemisphere]
        methods = {
            method: load_method_profiles(
                args.results_root,
                method,
                subject,
                hemisphere,
                args.template,
                cortex.size,
            )
            for method in ("O2-Column", "O2-FS", "O3")
        }
        valid = cortex.copy()
        for method in methods.values():
            valid &= method.declared_valid & method.finite_profile

        vertex_ids = np.flatnonzero(valid)
        correlations = profile_correlation(
            methods["O2-FS"].profiles[valid],
            methods["O3"].profiles[valid],
        )
        quantiles = np.linspace(0.05, 0.95, n_per_hemisphere)
        targets = np.quantile(correlations[np.isfinite(correlations)], quantiles)
        selected = []
        for target in targets:
            order = np.argsort(np.abs(correlations - target))
            chosen = next(
                vertex_ids[index]
                for index in order
                if vertex_ids[index] not in selected
            )
            selected.append(chosen)

        for vertex_id in selected:
            profiles = {
                "O2-Column": methods["O2-Column"].profiles[vertex_id],
                "O2-FS original": methods["O2-FS"].profiles[vertex_id],
                "O2-FS reversed": methods["O2-FS"].profiles[vertex_id, ::-1],
                "O3": methods["O3"].profiles[vertex_id],
            }
            panels.append((hemisphere, vertex_id, profiles))
            for method_name, profile in profiles.items():
                for depth, md_value in zip(CANONICAL_DEPTHS, profile):
                    values.append(
                        {
                            "subject_id": subject,
                            "hemisphere": hemisphere,
                            "target_vertex_id": vertex_id,
                            "method": method_name,
                            "depth": depth,
                            "md": md_value,
                        }
                    )

    n_columns = 5
    n_rows = int(np.ceil(len(panels) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(15, 2.8 * n_rows),
        sharex=True,
    )
    styles = {
        "O2-Column": ("#009E73", "-"),
        "O2-FS original": ("#D55E00", "-"),
        "O2-FS reversed": ("#CC79A7", "--"),
        "O3": ("#0072B2", "-"),
    }
    for axis, (hemisphere, vertex_id, profiles) in zip(axes.ravel(), panels):
        for method_name, profile in profiles.items():
            color, linestyle = styles[method_name]
            axis.plot(
                CANONICAL_DEPTHS,
                profile,
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
                label=method_name,
            )
        axis.set_title(f"{subject} {hemisphere} vertex {vertex_id}", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Cortical depth (pial 0 → white 1)")
    for axis in axes[:, 0]:
        axis.set_ylabel("MD")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
    )
    fig.suptitle("Representative depth profiles", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        args.output_dir / "representative_depth_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        args.output_dir / "representative_depth_profiles.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    return pd.DataFrame(values)


def surface_correlation_distribution(args):
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for axis, comparison in zip(axes, COMPARISONS):
        pooled = []
        for hemisphere in ("lh", "rh"):
            path = (
                args.results_root
                / "qc"
                / "template_registration_comparison"
                / "figure2_method_comparison_fsaverage"
                / f"figure2_{comparison}_vertex_agreement_map_{hemisphere}.csv"
            )
            table = pd.read_csv(path)
            values = table.loc[
                table["valid_subject_fraction"] >= 0.80,
                "mean_profile_r_fisher",
            ].dropna()
            pooled.append(values.to_numpy())
        values = np.concatenate(pooled)
        rows.append(
            {
                "comparison": comparison,
                "n_vertices": values.size,
                "minimum": np.min(values),
                "p01": np.quantile(values, 0.01),
                "p05": np.quantile(values, 0.05),
                "p25": np.quantile(values, 0.25),
                "median": np.median(values),
                "p75": np.quantile(values, 0.75),
                "p95": np.quantile(values, 0.95),
                "p99": np.quantile(values, 0.99),
                "maximum": np.max(values),
                "pct_below_0_9": 100.0 * np.mean(values < 0.9),
                "pct_below_0_8": 100.0 * np.mean(values < 0.8),
                "pct_below_0": 100.0 * np.mean(values < 0.0),
            }
        )
        axis.hist(values, bins=80, color="#0072B2", edgecolor="none")
        axis.axvline(0.8, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(0.9, color="black", linestyle=":", linewidth=0.8)
        title = (
            comparison.replace("O2Column", "O2-Column")
            .replace("O2FS", "O2-FS")
            .replace("_", " ")
        )
        axis.set(
            title=title,
            xlabel="Fisher-averaged profile correlation",
            ylabel="Template vertices",
        )
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "surface_profile_correlation_histograms.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "surface_profile_correlation_histograms.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = read_subjects(args.subjects_file)
    cortex_masks = {
        hemisphere: load_cortex_mask(
            args.freesurfer_subjects_dir,
            args.template,
            hemisphere,
        )
        for hemisphere in ("lh", "rh")
    }

    manifest_audit = audit_o2fs_manifests(args, subjects)
    manifest_audit.to_csv(
        args.output_dir / "o2fs_manifest_direction_audit.csv",
        index=False,
    )

    (
        direction_table,
        endpoint_table,
        relative_table,
        depth_table,
    ) = collect_subject_diagnostics(args, subjects, cortex_masks)
    direction_table.to_csv(
        args.output_dir / "depth_direction_subject_summary.csv",
        index=False,
    )
    direction_cohort, direction_diagnostic = summarize_direction_diagnostic(
        direction_table
    )
    direction_cohort.to_csv(
        args.output_dir / "depth_direction_cohort_summary.csv",
        index=False,
    )
    direction_diagnostic.to_csv(
        args.output_dir / "depth_direction_diagnostic.csv",
        index=False,
    )

    endpoint_table.to_csv(
        args.output_dir / "endpoint_correspondence_subject_summary.csv",
        index=False,
    )
    summarize_endpoints(endpoint_table).to_csv(
        args.output_dir / "endpoint_correspondence_cohort_summary.csv",
        index=False,
    )

    relative_table.to_csv(
        args.output_dir / "relative_scale_subject_summary.csv",
        index=False,
    )
    cohort_summary(
        relative_table,
        ["comparison"],
        ["median_profile_r", "median_profile_mae", "symmetric_nmae_pct"],
    ).to_csv(
        args.output_dir / "relative_scale_cohort_summary.csv",
        index=False,
    )
    depth_table.to_csv(
        args.output_dir / "depthwise_relative_scale_subject.csv",
        index=False,
    )
    cohort_summary(
        depth_table,
        ["comparison", "depth"],
        ["mae", "symmetric_nmae_pct"],
    ).to_csv(
        args.output_dir / "depthwise_relative_scale_cohort.csv",
        index=False,
    )

    representative_subject = select_representative_subject(direction_table)
    representative_profiles(args, representative_subject, cortex_masks).to_csv(
        args.output_dir / "representative_depth_profile_values.csv",
        index=False,
    )
    surface_correlation_distribution(args).to_csv(
        args.output_dir / "surface_profile_correlation_distribution.csv",
        index=False,
    )

    cross_subject_summary = calculate_cross_subject_results(
        args,
        subjects,
        cortex_masks,
    )
    cross_subject_summary.to_csv(
        args.output_dir / "cross_subject_metrics_by_depth.csv",
        index=False,
    )

    run_summary = {
        "template": args.template,
        "n_subjects": len(subjects),
        "canonical_depth_direction": "pial_to_white",
        "representative_subject": representative_subject,
        "comparisons": list(COMPARISONS),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info("Saved diagnostic results to %s", args.output_dir)


if __name__ == "__main__":
    main()
