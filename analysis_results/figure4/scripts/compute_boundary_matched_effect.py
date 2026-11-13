#!/usr/bin/env python3
"""Compute Figure 4D profile effects from frozen subject matched pairs.

Description
-----------
Compute Figure 4D profile effects from frozen subject matched pairs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-root`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_boundary_matched_effect.py --help
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compute_real_spatial_curves import load_profiles, select_subject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    return parser.parse_args()


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1), denominator,
        out=np.full(len(left), np.nan, dtype=float),
        where=denominator > 0,
    )


def add_profile_metrics(
    matched: pd.DataFrame,
    profiles: np.ndarray,
) -> pd.DataFrame:
    anchor = matched["anchor_vertex"].to_numpy(np.int64)
    within = matched["within_candidate_vertex"].to_numpy(np.int64)
    across = matched["across_candidate_vertex"].to_numpy(np.int64)
    anchor_profiles = profiles[anchor]
    within_profiles = profiles[within]
    across_profiles = profiles[across]
    r_within = row_correlation(anchor_profiles, within_profiles)
    r_across = row_correlation(anchor_profiles, across_profiles)
    result = matched.copy()
    result["r_within"] = r_within.astype(np.float32)
    result["r_across"] = r_across.astype(np.float32)
    result["d_within"] = (1.0 - r_within).astype(np.float32)
    result["d_across"] = (1.0 - r_across).astype(np.float32)
    result["delta_d"] = (result["d_across"] - result["d_within"]).astype(np.float32)
    result["within_profile_mae"] = np.mean(
        np.abs(anchor_profiles - within_profiles), axis=1
    ).astype(np.float32)
    result["across_profile_mae"] = np.mean(
        np.abs(anchor_profiles - across_profiles), axis=1
    ).astype(np.float32)
    result["delta_profile_mae"] = (
        result["across_profile_mae"] - result["within_profile_mae"]
    ).astype(np.float32)
    if not np.isfinite(result[["r_within", "r_across", "delta_d"]]).all().all():
        raise ValueError("Frozen matched pairs produced non-finite correlations")
    return result


def aggregate_metrics(metrics: pd.DataFrame, subject_id: str):
    anchor_keys = [
        "hemisphere", "anchor_vertex", "anchor_ico5_id", "anchor_dk_label",
        "anchor_region", "distance_bin_index", "distance_bin",
    ]
    anchor = (
        metrics.groupby(anchor_keys, as_index=False, observed=True)
        .agg(
            d_within=("d_within", "mean"),
            d_across=("d_across", "mean"),
            delta_d=("delta_d", "mean"),
            within_profile_mae=("within_profile_mae", "mean"),
            across_profile_mae=("across_profile_mae", "mean"),
            delta_profile_mae=("delta_profile_mae", "mean"),
            within_distance_mm=("within_distance_mm", "mean"),
            across_distance_mm=("across_distance_mm", "mean"),
            absolute_distance_difference_mm=(
                "absolute_distance_difference_mm", "mean"
            ),
            n_matches=("match_id", "size"),
        )
    )
    region = (
        anchor.groupby(
            [
                "anchor_dk_label", "anchor_region",
                "distance_bin_index", "distance_bin",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            d_within=("d_within", "mean"),
            d_across=("d_across", "mean"),
            delta_d=("delta_d", "mean"),
            within_profile_mae=("within_profile_mae", "mean"),
            across_profile_mae=("across_profile_mae", "mean"),
            delta_profile_mae=("delta_profile_mae", "mean"),
            within_distance_mm=("within_distance_mm", "mean"),
            across_distance_mm=("across_distance_mm", "mean"),
            absolute_distance_difference_mm=(
                "absolute_distance_difference_mm", "mean"
            ),
            anchor_count=("anchor_vertex", "size"),
            match_count=("n_matches", "sum"),
            hemisphere_count=("hemisphere", "nunique"),
        )
    )
    subject = (
        region.groupby(
            ["distance_bin_index", "distance_bin"],
            as_index=False,
            observed=True,
        )
        .agg(
            d_within=("d_within", "mean"),
            d_across=("d_across", "mean"),
            delta_d=("delta_d", "mean"),
            within_profile_mae=("within_profile_mae", "mean"),
            across_profile_mae=("across_profile_mae", "mean"),
            delta_profile_mae=("delta_profile_mae", "mean"),
            within_distance_mm=("within_distance_mm", "mean"),
            across_distance_mm=("across_distance_mm", "mean"),
            absolute_distance_difference_mm=(
                "absolute_distance_difference_mm", "mean"
            ),
            region_count=("anchor_dk_label", "nunique"),
            anchor_count=("anchor_count", "sum"),
            match_count=("match_count", "sum"),
        )
    )
    for table in (anchor, region, subject):
        table.insert(0, "subject_id", subject_id)
    return anchor, region, subject


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    started = time.time()
    try:
        pair_path = (
            args.pair_root / "subjects" / subject_id
            / "figure4D_boundary_matched_pairs.parquet"
        )
        matched = pd.read_parquet(pair_path)
        metric_parts = []
        for hemisphere in ("lh", "rh"):
            profiles, _ = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            part = matched.loc[matched["hemisphere"].eq(hemisphere)].reset_index(drop=True)
            metric_parts.append(add_profile_metrics(part, profiles))
        metrics = pd.concat(metric_parts, ignore_index=True)
        anchor, region, subject = aggregate_metrics(metrics, subject_id)
        metrics.to_parquet(
            output_dir / "figure4D_matched_pair_metrics.parquet",
            index=False, compression="zstd",
        )
        anchor.to_parquet(
            output_dir / "figure4D_anchor_bin_effects.parquet",
            index=False, compression="zstd",
        )
        region.to_parquet(
            output_dir / "figure4D_region_bin_effects.parquet",
            index=False, compression="zstd",
        )
        subject.to_csv(output_dir / "figure4D_subject_bin_effects.csv", index=False)
        completion.write_text(
            json.dumps(
                {
                    "status": "complete", "subject_id": subject_id,
                    "stage": "boundary_profile_effect",
                    "matched_pair_rows": int(len(metrics)),
                    "anchor_bin_rows": int(len(anchor)),
                    "region_bin_rows": int(len(region)),
                    "subject_bin_rows": int(len(subject)),
                    "aggregation": (
                        "matched-pair mean -> anchor-bin mean -> bilateral "
                        "subject-region-bin mean -> equal-region subject-bin mean"
                    ),
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                },
                indent=2,
            )
            + "\n"
        )
        failure.unlink(missing_ok=True)
    except Exception as error:
        failure.write_text(
            json.dumps(
                {
                    "status": "failed", "subject_id": subject_id,
                    "stage": "boundary_profile_effect",
                    "error_type": type(error).__name__, "error": str(error),
                    "runtime_seconds": time.time() - started,
                },
                indent=2,
            )
            + "\n"
        )
        completion.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
