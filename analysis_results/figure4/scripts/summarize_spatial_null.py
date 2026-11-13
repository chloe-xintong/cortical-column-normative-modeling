#!/usr/bin/env python3
"""Aggregate completed Figure 4 null runs without executing a notebook.

Description
-----------
Aggregate completed Figure 4 null runs without executing a notebook.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--results-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/summarize_spatial_null.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    statuses = []
    for subject_id in manifest["subject_id"]:
        subject_dir = args.results_root / "subjects" / subject_id
        completion = subject_dir / "completion.json"
        failure = subject_dir / "failure.json"
        if completion.is_file():
            tables.append(pd.read_csv(subject_dir / "figure4_null_subject_bins.csv"))
            status = "complete"
            reason = ""
        elif failure.is_file():
            failure_data = json.loads(failure.read_text(encoding="utf-8"))
            status = "failed"
            reason = str(failure_data.get("error", ""))
        else:
            status = "incomplete"
            reason = "No completion JSON"
        statuses.append({"subject_id": subject_id, "status": status, "reason": reason})
    pd.DataFrame(statuses).to_csv(
        args.output_dir / "figure4_null_status.csv",
        index=False,
    )
    if not tables:
        raise RuntimeError("No completed Figure 4 null results were found")
    subject_bins = pd.concat(tables, ignore_index=True)
    subject_bins.to_csv(
        args.output_dir / "figure4_null_subject_replicate_bins.csv",
        index=False,
    )
    metric_column = (
        "null_dissimilarity"
        if "null_dissimilarity" in subject_bins.columns
        else "median_dissimilarity"
    )
    group_columns = [
                "subject_id",
                "distance_bin_index",
                "distance_bin",
                "null_method",
                "psf_fwhm_mm",
                "preliminary",
            ]
    if "laminar_recolored" in subject_bins.columns:
        group_columns.append("laminar_recolored")
    grouped = subject_bins.groupby(
            group_columns,
            observed=True,
        )
    subject_replicate = grouped[metric_column].median().rename(
        "null_median_dissimilarity"
    ).reset_index()
    if {"real_dissimilarity", "null_minus_real"}.issubset(subject_bins.columns):
        real_delta = grouped[["real_dissimilarity", "null_minus_real"]].median().reset_index()
        subject_replicate = subject_replicate.merge(
            real_delta,
            on=group_columns,
            validate="one_to_one",
        )
    extra_metrics = [
        column for column in (
            "real_local_excess_similarity",
            "null_local_excess_similarity",
            "real_minus_null_local_excess",
        ) if column in subject_bins.columns
    ]
    if extra_metrics:
        extra = grouped[extra_metrics].median().reset_index()
        subject_replicate = subject_replicate.merge(
            extra, on=group_columns, validate="one_to_one"
        )
    cohort_group_columns = [
        column for column in group_columns if column != "subject_id"
    ]
    group_summary = (
        subject_replicate.groupby(
            cohort_group_columns,
            observed=True,
        )["null_median_dissimilarity"]
        .agg(
            n_subjects="count",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    group_summary.to_csv(
        args.output_dir / "figure4_null_group_summary.csv",
        index=False,
    )
    if "null_minus_real" in subject_replicate.columns:
        delta_summary = (
            subject_replicate.groupby(
                cohort_group_columns,
                observed=True,
            )["null_minus_real"]
            .agg(
                n_subjects="count",
                median="median",
                q25=lambda values: values.quantile(.25),
                q75=lambda values: values.quantile(.75),
            )
            .reset_index()
        )
        delta_summary.to_csv(
            args.output_dir / "figure4_null_minus_real_group_summary.csv",
            index=False,
        )
    if extra_metrics:
        rows = []
        for keys, table in subject_replicate.groupby(
            cohort_group_columns, observed=True
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(cohort_group_columns, keys))
            row["n_subjects"] = table.subject_id.nunique()
            for metric in extra_metrics:
                row[f"{metric}_median"] = table[metric].median()
                row[f"{metric}_q25"] = table[metric].quantile(.25)
                row[f"{metric}_q75"] = table[metric].quantile(.75)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            args.output_dir / "figure4_local_excess_similarity_group_summary.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
