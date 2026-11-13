#!/usr/bin/env python3
"""Aggregate the all-pair, region-balanced DK effect across subjects.

Description
-----------
Aggregate the all-pair, region-balanced DK effect across subjects.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-table`` command-line input
- ``--results-root`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/summarize_dk_all_pairs_effect.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser.parse_args()


def bootstrap_intervals(
    subject: pd.DataFrame,
    repetitions: int,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(20261021)
    rows = []
    for bin_index, table in subject.groupby("distance_bin_index"):
        values = table[value_column].dropna().to_numpy(dtype=float)
        boot = np.median(
            rng.choice(values, size=(repetitions, len(values)), replace=True), axis=1
        )
        rows.append(
            {
                "distance_bin_index": int(bin_index),
                f"{prefix}_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                f"{prefix}_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    expected = manifest.loc[manifest["input_complete"].astype(bool), "subject_id"].tolist()
    paths = sorted(
        args.results_root.glob("subjects/*/figure4_dk_all_pairs_subject_bins.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No subject summaries under {args.results_root}")

    subject = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    group = (
        subject.groupby(
            ["distance_bin_index", "distance_bin"], as_index=False
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            within_group_median=("within_dissimilarity", "median"),
            within_q25=("within_dissimilarity", lambda x: x.quantile(.25)),
            within_q75=("within_dissimilarity", lambda x: x.quantile(.75)),
            across_group_median=("across_dissimilarity", "median"),
            across_q25=("across_dissimilarity", lambda x: x.quantile(.25)),
            across_q75=("across_dissimilarity", lambda x: x.quantile(.75)),
            delta_group_median=("across_minus_within", "median"),
            delta_q25=("across_minus_within", lambda x: x.quantile(.25)),
            delta_q75=("across_minus_within", lambda x: x.quantile(.75)),
            within_profile_mae_group_median=("within_profile_mae", "median"),
            within_profile_mae_q25=("within_profile_mae", lambda x: x.quantile(.25)),
            within_profile_mae_q75=("within_profile_mae", lambda x: x.quantile(.75)),
            across_profile_mae_group_median=("across_profile_mae", "median"),
            across_profile_mae_q25=("across_profile_mae", lambda x: x.quantile(.25)),
            across_profile_mae_q75=("across_profile_mae", lambda x: x.quantile(.75)),
            mae_delta_group_median=("across_minus_within_mae", "median"),
            mae_delta_q25=("across_minus_within_mae", lambda x: x.quantile(.25)),
            mae_delta_q75=("across_minus_within_mae", lambda x: x.quantile(.75)),
            matched_region_subbin_count_total=(
                "matched_region_subbin_count", "sum"
            ),
            matched_region_count_median_subject=("matched_region_count", "median"),
            within_anchor_subbin_count_total=("within_anchor_subbin_count", "sum"),
            across_anchor_subbin_count_total=("across_anchor_subbin_count", "sum"),
            within_pair_count_total=("within_pair_count", "sum"),
            across_pair_count_total=("across_pair_count", "sum"),
            within_distance_median=("within_distance_median", "median"),
            across_distance_median=("across_distance_median", "median"),
            distance_difference_median_mm=(
                "distance_difference_median_mm", "median"
            ),
        )
    )
    group = group.merge(
        bootstrap_intervals(
            subject,
            args.bootstrap_repetitions,
            "across_minus_within",
            "delta",
        ),
        on="distance_bin_index",
        how="left",
    )
    group = group.merge(
        bootstrap_intervals(
            subject,
            args.bootstrap_repetitions,
            "across_minus_within_mae",
            "mae_delta",
        ),
        on="distance_bin_index",
        how="left",
    )

    region_paths = sorted(
        args.results_root.glob("subjects/*/figure4_dk_all_pairs_region_subbins.parquet")
    )
    region = pd.concat([pd.read_parquet(path) for path in region_paths], ignore_index=True)
    subject.to_csv(
        args.output_dir / "figure4_dk_all_pairs_subject_bins.csv", index=False
    )
    group.to_csv(
        args.output_dir / "figure4_dk_all_pairs_group_summary.csv", index=False
    )
    region.to_parquet(
        args.output_dir / "figure4_dk_all_pairs_region_subbins.parquet", index=False
    )

    pair_table = pd.read_parquet(
        args.pair_table,
        columns=["pair_type", "distance_bin_index", "distance_bin", "geodesic_distance_mm"],
    )
    pair_qc = (
        pair_table.groupby(
            ["distance_bin_index", "distance_bin", "pair_type"], as_index=False
        )
        .agg(
            fixed_pair_count=("geodesic_distance_mm", "size"),
            distance_median=("geodesic_distance_mm", "median"),
            distance_q25=("geodesic_distance_mm", lambda x: x.quantile(.25)),
            distance_q75=("geodesic_distance_mm", lambda x: x.quantile(.75)),
        )
    )
    pair_qc.to_csv(
        args.output_dir / "figure4_dk_all_pairs_fixed_pair_distance_qc.csv",
        index=False,
    )

    completed = sorted(subject["subject_id"].unique().tolist())
    failed = sorted(set(expected) - set(completed))
    status = {
        "status": "complete" if not failed else "incomplete",
        "subjects_expected": len(expected),
        "subjects_completed": len(completed),
        "subjects_failed": len(failed),
        "failed_subject_ids": failed,
        "fixed_pair_count": int(len(pair_table)),
        "within_fixed_pair_count": int((pair_table["pair_type"] == "within_region").sum()),
        "across_fixed_pair_count": int((pair_table["pair_type"] == "across_boundary").sum()),
        "distance_subbin_width_mm": 0.5,
        "aggregation": (
            "pair -> anchor x 0.5-mm sub-bin median -> hemisphere x DK region "
            "x sub-bin median -> subject median -> group median/IQR"
        ),
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    (args.output_dir / "figure4_dk_all_pairs_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
