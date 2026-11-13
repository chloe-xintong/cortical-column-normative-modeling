#!/usr/bin/env python3
"""Summarize real bilateral DK-region results for frozen ico5 K=15 pairs.

Description
-----------
This stage intentionally contains no region-level null inference.  It reports
real D(d), subject-level exponential fits, and sampling/coverage diagnostics.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-file`` command-line input
- ``--results-root`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/summarize_ico5_region_real.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_correlation_lengths import fit_curve


MIN_SUBJECT_FRACTION = 0.80
K_TARGET = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-fit-r2", type=float, default=0.5)
    return parser.parse_args()


def quantile25(values: pd.Series) -> float:
    return values.quantile(0.25)


def quantile75(values: pd.Series) -> float:
    return values.quantile(0.75)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    subject_ids = manifest["subject_id"].drop_duplicates().tolist()
    minimum_subjects = int(np.ceil(MIN_SUBJECT_FRACTION * len(subject_ids)))

    tables = []
    status_rows = []
    for subject_id in subject_ids:
        subject_dir = args.results_root / "subjects" / subject_id
        completion = subject_dir / "completion.json"
        failure = subject_dir / "failure.json"
        table_path = subject_dir / "figure4_region_real_subject_bins.csv"
        if completion.is_file() and table_path.is_file():
            tables.append(pd.read_csv(table_path, dtype={"subject_id": str}))
            status_rows.append(
                {"subject_id": subject_id, "status": "complete", "reason": ""}
            )
        elif failure.is_file():
            failure_data = json.loads(failure.read_text())
            status_rows.append(
                {
                    "subject_id": subject_id,
                    "status": "failed",
                    "reason": str(failure_data.get("error", "")),
                }
            )
        else:
            status_rows.append(
                {
                    "subject_id": subject_id,
                    "status": "incomplete",
                    "reason": "No completed regional table",
                }
            )
    status = pd.DataFrame(status_rows)
    status.to_csv(args.output_dir / "figure4_region_real_status.csv", index=False)
    if not tables:
        raise RuntimeError("No completed regional subject tables were found")

    subject = pd.concat(tables, ignore_index=True)
    subject.to_parquet(
        args.output_dir / "figure4_region_real_subject_bins.parquet",
        index=False,
        compression="zstd",
    )
    keys = [
        "anchor_dk_label", "anchor_region", "distance_bin_index", "distance_bin"
    ]
    group = (
        subject.groupby(keys, as_index=False, observed=True)
        .agg(
            n_subjects=("median_dissimilarity", "count"),
            dissimilarity_median=("median_dissimilarity", "median"),
            dissimilarity_q25=("median_dissimilarity", quantile25),
            dissimilarity_q75=("median_dissimilarity", quantile75),
            profile_mae_median=("median_profile_mae", "median"),
            profile_mae_q25=("median_profile_mae", quantile25),
            profile_mae_q75=("median_profile_mae", quantile75),
            median_distance_mm=("median_geodesic_distance_mm", "median"),
            expected_anchor_count_median=("expected_anchor_count", "median"),
            valid_anchor_count_median=("valid_anchor_count", "median"),
            valid_pair_count_median=("valid_pair_count", "median"),
            median_k_effective=("median_k_effective", "median"),
            q25_k_effective=("median_k_effective", quantile25),
            q75_k_effective=("median_k_effective", quantile75),
            full_k15_coverage_median=("full_k15_coverage", "median"),
            full_k15_coverage_q25=("full_k15_coverage", quantile25),
            full_k15_coverage_q75=("full_k15_coverage", quantile75),
            any_valid_anchor_coverage_median=(
                "any_valid_anchor_coverage", "median"
            ),
        )
    )
    group["coverage_pass"] = group["n_subjects"].ge(minimum_subjects)
    group.to_csv(args.output_dir / "figure4_region_real_group_curves.csv", index=False)

    fits = []
    for (subject_id, label, region), frame in subject.groupby(
        ["subject_id", "anchor_dk_label", "anchor_region"], observed=True
    ):
        fit_input = frame.rename(
            columns={"median_dissimilarity": "median_dissimilarity"}
        )
        fits.append(
            {
                "subject_id": subject_id,
                "anchor_dk_label": label,
                "anchor_region": region,
                **fit_curve(fit_input, min_fit_r2=args.min_fit_r2),
            }
        )
    fit_table = pd.DataFrame(fits)
    fit_table.to_csv(
        args.output_dir / "figure4_region_real_subject_correlation_lengths.csv",
        index=False,
    )
    acceptable = fit_table.loc[fit_table["fit_acceptable"]].copy()
    fit_group = (
        acceptable.groupby(
            ["anchor_dk_label", "anchor_region"], as_index=False, observed=True
        )
        .agg(
            n_acceptable=("lambda_mm", "count"),
            lambda_median_mm=("lambda_mm", "median"),
            lambda_q25_mm=("lambda_mm", quantile25),
            lambda_q75_mm=("lambda_mm", quantile75),
            d_infinity_median=("d_infinity", "median"),
            amplitude_median=("amplitude", "median"),
            fit_r2_median=("fit_r2", "median"),
        )
    )
    attempted = (
        fit_table.groupby(
            ["anchor_dk_label", "anchor_region"], as_index=False, observed=True
        )
        .agg(
            n_subjects_attempted=("subject_id", "nunique"),
            n_fit_success=("fit_success", "sum"),
            n_boundary=("hit_parameter_boundary", "sum"),
        )
    )
    fit_group = attempted.merge(
        fit_group,
        on=["anchor_dk_label", "anchor_region"],
        how="left",
        validate="one_to_one",
    )
    fit_group["lambda_coverage_pass"] = fit_group["n_acceptable"].ge(
        minimum_subjects
    )
    fit_group.to_csv(
        args.output_dir / "figure4_region_real_group_correlation_lengths.csv",
        index=False,
    )

    pairs = pd.read_parquet(
        args.pair_file,
        columns=[
            "hemisphere", "anchor_vertex", "anchor_dk_label", "anchor_region",
            "neighbor_vertex", "distance_bin_index", "distance_bin", "n_candidates",
        ],
    )
    static_anchor = (
        pairs.groupby(
            [
                "hemisphere", "anchor_vertex", "anchor_dk_label", "anchor_region",
                "distance_bin_index", "distance_bin",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            fixed_sampled_k=("neighbor_vertex", "size"),
            candidate_count=("n_candidates", "first"),
        )
    )
    static_coverage = (
        static_anchor.groupby(keys, as_index=False, observed=True)
        .agg(
            fixed_anchor_count=("anchor_vertex", "size"),
            fixed_pair_count=("fixed_sampled_k", "sum"),
            fixed_full_k15_coverage=(
                "fixed_sampled_k", lambda values: values.eq(K_TARGET).mean()
            ),
            candidate_count_median=("candidate_count", "median"),
            candidate_count_minimum=("candidate_count", "min"),
        )
    )
    static_coverage.to_csv(
        args.output_dir / "figure4_region_fixed_pair_coverage.csv", index=False
    )

    local = group.loc[group["distance_bin_index"].lt(5) & group["coverage_pass"]]
    inventory = (
        local.groupby(
            ["anchor_dk_label", "anchor_region"], as_index=False, observed=True
        )
        .agg(
            mean_real_dissimilarity_0_10=("dissimilarity_median", "mean"),
            minimum_subjects_0_10=("n_subjects", "min"),
            minimum_full_k15_coverage_0_10=("full_k15_coverage_median", "min"),
            minimum_median_k_effective_0_10=("median_k_effective", "min"),
        )
        .merge(
            fit_group,
            on=["anchor_dk_label", "anchor_region"],
            how="left",
            validate="one_to_one",
        )
        .sort_values("mean_real_dissimilarity_0_10")
    )
    inventory.to_csv(args.output_dir / "figure4_region_inventory.csv", index=False)

    report = {
        "status": "complete" if status["status"].eq("complete").all() else "incomplete",
        "subjects_expected": len(subject_ids),
        "subjects_completed": int(status["status"].eq("complete").sum()),
        "bilateral_regions": int(subject["anchor_region"].nunique()),
        "distance_bins": int(subject["distance_bin_index"].nunique()),
        "minimum_subject_fraction": MIN_SUBJECT_FRACTION,
        "minimum_subjects_for_coverage": minimum_subjects,
        "k": K_TARGET,
        "null_included": False,
        "null_dependent_outputs_deferred": [
            "regional_null_curve", "regional_delta_L", "regional_residual_AUC"
        ],
        "aggregation": (
            "pair -> hemisphere x anchor x bin median -> subject x bilateral "
            "region x bin median -> group median/IQR"
        ),
    }
    (args.output_dir / "figure4_region_real_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
