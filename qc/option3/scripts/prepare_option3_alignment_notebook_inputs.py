#!/usr/bin/env python3
"""Prepare Option 3 Alignment QC notebook tables from completed cohort caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/qc/"
    "full_cohort_spatial_scale/option3_qc/cohort"
)
DEFAULT_OUTPUT = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/qc/"
    "full_cohort_spatial_scale/option3_alignment_qc_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage-review-pct", type=float, default=95.0)
    return parser.parse_args()


def weighted_mean(group: pd.DataFrame, column: str, weight: str) -> float:
    valid = group[column].notna() & group[weight].notna() & group[weight].gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(group.loc[valid, column], weights=group.loc[valid, weight]))


def distribution(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        table[columns]
        .agg(
            [
                "median",
                lambda values: values.quantile(0.25),
                lambda values: values.quantile(0.75),
                "min",
                "max",
            ]
        )
        .set_axis(["median", "Q25", "Q75", "minimum", "maximum"])
        .rename_axis("statistic")
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    source = args.source_dir
    output = args.output_dir
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    status = pd.read_csv(
        source / "full_cohort_status.csv", dtype={"subject_id": str}
    )
    cached = pd.read_csv(
        source / "option3_mapping_qc_subject_hemi.csv",
        dtype={"subject_id": str},
    )
    depth = pd.read_csv(
        source / "option3_invalid_md_by_depth.csv",
        dtype={"subject_id": str},
    )
    registration = pd.read_csv(
        source / "registration_cost_by_subject.csv",
        dtype={"subject_id": str},
    )

    if status["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject IDs in full_cohort_status.csv")
    if cached.duplicated(["subject_id", "hemisphere"]).any():
        raise ValueError("Duplicate subject/hemisphere rows in mapping QC cache")

    hemisphere = pd.DataFrame(
        {
            "subject_id": cached["subject_id"],
            "hemisphere": cached["hemisphere"],
            "n_mask": cached["n_analysis_mask_vertices"].astype(int),
            "mapping_full_pct": cached["mapping_valid_percentage"],
            # Mapping was complete for the cached cohort. The cache does not
            # store a separate mask-specific mapping count.
            "mapping_mask_pct": cached["mapping_valid_percentage"],
            "geometry_pct": 100
            * (
                cached["n_analysis_mask_vertices"]
                - cached["zero_length_count_analysis_mask"]
            )
            / cached["n_analysis_mask_vertices"],
            "final_pct": cached["complete_valid_profile_pct_analysis_mask"],
            "zero_full_pct": cached["zero_length_pct_full_mesh"],
            "zero_mask_pct": cached["zero_length_pct_analysis_mask"],
            # The completed cache has separate point-event rates but not their
            # union within the analysis mask. Leave this undefined rather than
            # summing overlapping events.
            "point_invalid_pct": np.nan,
            "column_invalid_pct": (
                100
                * (
                    cached["n_analysis_mask_vertices"]
                    - cached["zero_length_count_analysis_mask"]
                )
                / cached["n_analysis_mask_vertices"]
                - cached["complete_valid_profile_pct_analysis_mask"]
            ).clip(lower=0),
            "pearson_r": cached["length_thickness_pearson_r"],
            "spearman_rho": cached["length_thickness_spearman_rho"],
            "mae_mm": cached["length_thickness_mae_mm"],
            "median_difference_mm": cached[
                "length_minus_thickness_median_mm"
            ],
            "weight_sum_error_max": cached["weight_sum_error_max"],
            "reconstruction_error_max": cached["mapping_error_max"],
            "endpoint_pial_error_mm": cached["max_pial_endpoint_error_mm"],
            "endpoint_white_error_mm": cached["max_white_endpoint_error_mm"],
            "line_error_mm": cached["max_column_formula_error_mm"],
            "equal_spacing_error_mm": np.nan,
        }
    )
    relative = cached["length_thickness_relative_difference_median"]
    hemisphere["median_length_thickness_ratio"] = (2 + relative) / (2 - relative)
    hemisphere["sampling_retention_geometry_pct"] = (
        100 * hemisphere["final_pct"] / hemisphere["geometry_pct"]
    )

    pooled_rows: list[dict[str, object]] = []
    weighted_columns = [
        "mapping_mask_pct",
        "geometry_pct",
        "final_pct",
        "zero_mask_pct",
        "column_invalid_pct",
        "pearson_r",
        "spearman_rho",
        "mae_mm",
        "median_difference_mm",
        "median_length_thickness_ratio",
    ]
    for subject_id, group in hemisphere.groupby("subject_id", sort=False):
        row: dict[str, object] = {"subject_id": subject_id}
        for column in weighted_columns:
            row[column] = weighted_mean(group, column, "n_mask")
        row["point_invalid_pct"] = np.nan
        row["coverage_review_flag"] = (
            row["final_pct"] < args.coverage_review_pct
        )
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)

    mask_sizes = hemisphere[["subject_id", "hemisphere", "n_mask"]]
    depth = depth.merge(
        mask_sizes,
        on=["subject_id", "hemisphere"],
        how="left",
        validate="many_to_one",
    )
    depth_rows: list[dict[str, object]] = []
    for (subject_id, depth_fraction), group in depth.groupby(
        ["subject_id", "depth_fraction_pial_to_white"], sort=False
    ):
        depth_rows.append(
            {
                "subject_id": subject_id,
                "depth": depth_fraction,
                "oob_pct": weighted_mean(
                    group, "point_oob_percentage", "n_mask"
                ),
                "nan_pct": np.nan,
                "nonfinite_pct": weighted_mean(
                    group, "point_nonfinite_md_percentage", "n_mask"
                ),
                "zero_pct": weighted_mean(
                    group, "point_zero_md_percentage", "n_mask"
                ),
            }
        )
    depth_subject = pd.DataFrame(depth_rows)

    label_qc = pd.DataFrame(
        {
            "subject_id": cached["subject_id"],
            "hemisphere": cached["hemisphere"],
            "same_label_pct": (
                cached["valid_label_triangle_percentage"]
                - cached["mixed_label_triangle_percentage"]
            ).clip(lower=0),
            "mixed_pct": cached["mixed_label_triangle_percentage"],
            "majority_agreement_pct": np.nan,
            "mixed_near_template_boundary_pct": cached[
                "mixed_within_one_boundary_hop_percentage"
            ],
        }
    )

    registration_qc = registration.rename(
        columns={"automatic_cost_outlier": "cost_review_flag"}
    ).copy()
    registration_qc["registration_error"] = ""
    registration_qc["exclusion_reason"] = ""
    registration_qc["gm_wm_boundary_offset_mm"] = np.nan
    registration_qc["excluded"] = ~registration_qc[
        "final_exclusion_status"
    ].fillna("not_excluded").eq("not_excluded")
    registration_qc = registration_qc[
        [
            "subject_id",
            "bbregister_mincost",
            "mean_white_signal",
            "mean_gray_signal",
            "percent_contrast",
            "registration_error",
            "robust_z",
            "cost_review_flag",
            "visual_qc_status",
            "exclusion_reason",
            "gm_wm_boundary_offset_mm",
            "excluded",
        ]
    ]
    review_queue = registration_qc.loc[
        registration_qc["cost_review_flag"]
        | registration_qc["visual_qc_status"].eq("not_reviewed")
    ].copy()

    subject_qc = (
        status.merge(pooled, on="subject_id", how="left", validate="one_to_one")
        .merge(
            registration_qc,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )
    )

    status.to_csv(tables / "status.csv", index=False)
    (
        status.groupby("status")
        .size()
        .rename("n_subjects")
        .reset_index()
        .to_csv(tables / "status_counts.csv", index=False)
    )
    hemisphere.to_csv(tables / "hemisphere_qc.csv", index=False)
    subject_qc.to_csv(tables / "subject_qc.csv", index=False)
    depth_subject.to_csv(tables / "depth_qc.csv", index=False)
    label_qc.to_csv(tables / "label_qc.csv", index=False)
    registration_qc.to_csv(tables / "registration_qc.csv", index=False)
    review_queue.to_csv(tables / "registration_review_queue.csv", index=False)

    successful = pooled.loc[pooled["subject_id"].isin(
        status.loc[status["status"].eq("complete"), "subject_id"]
    )]
    distribution(
        successful, ["mapping_mask_pct", "geometry_pct", "final_pct"]
    ).to_csv(tables / "coverage_summary.csv", index=False)
    distribution(
        successful,
        [
            "pearson_r",
            "spearman_rho",
            "mae_mm",
            "median_difference_mm",
            "median_length_thickness_ratio",
        ],
    ).to_csv(tables / "geometry_summary.csv", index=False)
    distribution(
        successful,
        ["point_invalid_pct", "column_invalid_pct", "geometry_pct", "final_pct"],
    ).to_csv(tables / "sampling_summary.csv", index=False)
    subject_qc.nsmallest(5, "final_pct").to_csv(
        tables / "lowest_coverage_subjects.csv", index=False
    )
    hemisphere.groupby("hemisphere")[[
        "endpoint_pial_error_mm",
        "endpoint_white_error_mm",
        "line_error_mm",
        "equal_spacing_error_mm",
    ]].max().reset_index().to_csv(
        tables / "trajectory_summary.csv", index=False
    )

    mapping_geometry = hemisphere.groupby("hemisphere")[[
        "mapping_full_pct",
        "weight_sum_error_max",
        "reconstruction_error_max",
        "zero_full_pct",
        "zero_mask_pct",
    ]].agg(["median", "max"])
    mapping_geometry.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in mapping_geometry.columns
    ]
    mapping_geometry.reset_index().to_csv(
        tables / "mapping_geometry_by_hemisphere.csv", index=False
    )

    for metric in ["final_pct", "geometry_pct", "zero_mask_pct"]:
        wide = hemisphere.pivot(
            index="subject_id", columns="hemisphere", values=metric
        )
        difference = (wide["lh"] - wide["rh"]).describe()
        (
            difference.rename_axis("statistic")
            .rename(f"{metric}_lh_minus_rh")
            .reset_index()
            .to_csv(
                tables / f"hemisphere_difference_{metric}.csv", index=False
            )
        )

    metadata = {
        "status": "complete",
        "source_mode": "completed_option3_cohort_cache",
        "source_directory": str(source),
        "n_requested": int(len(status)),
        "n_successful": int(status["status"].eq("complete").sum()),
        "n_failed": int(status["status"].eq("failed").sum()),
        "coverage_review_pct": args.coverage_review_pct,
        "table_directory": str(tables),
        "figure_directory": str(figures),
        "figures": [],
        "limitations": [
            "The completed cohort cache does not contain the union of point-level invalid events within the analysis mask; point_invalid_pct is left missing.",
            "Depth-specific point rates use the denominators stored by the completed cohort cache.",
            "Bilateral length-thickness statistics are mask-size-weighted hemisphere summaries rather than correlations refit after pooling vertices.",
            "Equal-spacing error was not retained in the completed cohort cache and is left missing.",
        ],
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
