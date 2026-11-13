#!/usr/bin/env python3
"""Compute one pilot subject's nested-K ico5 within-region summaries.

Description
-----------
Compute one pilot subject's nested-K ico5 within-region summaries.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--subject-list`` command-line input
- ``--pair-dir`` command-line input
- ``--output-root`` command-line input
- ``--index`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_ico5_k_subject.py --help
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compute_real_spatial_curves import load_profiles


K_VALUES = (1, 3, 5, 10, 15, 20, 25, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subject-list", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--existing-results-root", type=Path)
    parser.add_argument("--new-k-values", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--diagnostic-k10", action="store_true")
    parser.add_argument("--index", type=int, required=True)
    return parser.parse_args()


def pair_metrics(profiles: np.ndarray, valid: np.ndarray, pairs: pd.DataFrame) -> pd.DataFrame:
    anchors = pairs["anchor_vertex"].to_numpy(np.int64)
    neighbors = pairs["neighbor_vertex"].to_numpy(np.int64)
    keep = valid[anchors] & valid[neighbors]
    pairs = pairs.loc[keep].copy()
    left = profiles[anchors[keep]]
    right = profiles[neighbors[keep]]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
        right_centered, axis=1
    )
    correlation = np.divide(
        np.sum(left_centered * right_centered, axis=1), denominator,
        out=np.full(len(left), np.nan), where=denominator > 0,
    )
    mae = np.mean(np.abs(left - right), axis=1)
    srmae_denominator = np.median(np.abs(left), axis=1) + np.median(
        np.abs(right), axis=1
    )
    srmae = np.divide(
        200.0 * mae, srmae_denominator, out=np.full(len(mae), np.nan),
        where=np.isfinite(srmae_denominator) & (srmae_denominator > 0),
    )
    pairs["profile_dissimilarity"] = 1.0 - correlation
    pairs["profile_mae"] = mae
    pairs["symmetric_relative_mae_pct"] = srmae
    return pairs.loc[np.isfinite(pairs[
        ["profile_dissimilarity", "profile_mae", "symmetric_relative_mae_pct"]
    ]).all(axis=1)]


def main() -> None:
    args = parse_args()
    subject_ids = [x.strip() for x in args.subject_list.read_text().splitlines() if x.strip()]
    subject_id = subject_ids[args.index]
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    selected = manifest.loc[manifest["subject_id"].eq(subject_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one manifest row for {subject_id}")
    row = selected.iloc[0]
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    new_k_values = tuple(sorted(set(args.new_k_values)))
    calculation_k_values = tuple(sorted(set(
        new_k_values + ((10,) if args.diagnostic_k10 else tuple())
    )))
    profiles_by_hemi = {
        hemi: load_profiles(Path(row[f"clean_profile_{hemi}"]))
        for hemi in ("lh", "rh")
    }
    subject_parts: list[pd.DataFrame] = []
    region_parts: list[pd.DataFrame] = []
    full20_subject_parts: list[pd.DataFrame] = []
    full20_region_parts: list[pd.DataFrame] = []
    full30_subject_parts: list[pd.DataFrame] = []
    full30_region_parts: list[pd.DataFrame] = []
    maximum_rank_needed = max(calculation_k_values)

    for pair_path in sorted(args.pair_dir.glob("seed_*.parquet")):
        seed = int(pair_path.stem.split("_")[-1])
        pairs = pd.read_parquet(
            pair_path, filters=[("sample_rank", "<=", maximum_rank_needed)]
        )
        metric_parts = []
        for hemisphere in ("lh", "rh"):
            hemi_pairs = pairs.loc[pairs["hemisphere"].eq(hemisphere)].reset_index(drop=True)
            profiles, valid = profiles_by_hemi[hemisphere]
            metric_parts.append(pair_metrics(profiles, valid, hemi_pairs))
        metrics = pd.concat(metric_parts, ignore_index=True)
        for k in calculation_k_values:
            subset = metrics.loc[metrics["sample_rank"].le(k)]
            anchor = subset.groupby(
                ["hemisphere", "anchor_vertex", "anchor_dk_label", "anchor_region",
                 "distance_bin_index", "distance_bin"],
                as_index=False, observed=True,
            ).agg(
                median_dissimilarity=("profile_dissimilarity", "median"),
                median_profile_mae=("profile_mae", "median"),
                median_symmetric_relative_mae_pct=("symmetric_relative_mae_pct", "median"),
                median_geodesic_distance_mm=("geodesic_distance_mm", "median"),
                n_neighbors_used=("profile_dissimilarity", "size"),
                n_candidates=("n_candidates", "first"),
            )
            if k in new_k_values:
                whole = anchor.groupby(
                    ["distance_bin_index", "distance_bin"], as_index=False, observed=True
                ).agg(
                    profile_dissimilarity=("median_dissimilarity", "median"),
                    profile_mae=("median_profile_mae", "median"),
                    symmetric_relative_mae_pct=("median_symmetric_relative_mae_pct", "median"),
                    median_geodesic_distance_mm=("median_geodesic_distance_mm", "median"),
                    anchor_count=("anchor_vertex", "size"),
                    pair_count=("n_neighbors_used", "sum"),
                )
                whole.insert(0, "k", k)
                whole.insert(0, "seed", seed)
                whole.insert(0, "subject_id", subject_id)
                subject_parts.append(whole)

                region = anchor.groupby(
                    ["anchor_region", "distance_bin_index", "distance_bin"],
                    as_index=False, observed=True,
                ).agg(
                    profile_dissimilarity=("median_dissimilarity", "median"),
                    profile_mae=("median_profile_mae", "median"),
                    symmetric_relative_mae_pct=("median_symmetric_relative_mae_pct", "median"),
                    median_geodesic_distance_mm=("median_geodesic_distance_mm", "median"),
                    anchor_count=("anchor_vertex", "size"),
                    pair_count=("n_neighbors_used", "sum"),
                    hemispheres_contributing=("hemisphere", "nunique"),
                )
                region.insert(0, "k", k)
                region.insert(0, "seed", seed)
                region.insert(0, "subject_id", subject_id)
                region_parts.append(region)

            if k >= 10:
                eligible = anchor.loc[anchor["n_candidates"].ge(20)]
                full20_whole = eligible.groupby(
                    ["distance_bin_index", "distance_bin"],
                    as_index=False, observed=True,
                ).agg(
                    profile_dissimilarity=("median_dissimilarity", "median"),
                    profile_mae=("median_profile_mae", "median"),
                    symmetric_relative_mae_pct=(
                        "median_symmetric_relative_mae_pct", "median"
                    ),
                    median_geodesic_distance_mm=(
                        "median_geodesic_distance_mm", "median"
                    ),
                    anchor_count=("anchor_vertex", "size"),
                    pair_count=("n_neighbors_used", "sum"),
                )
                full20_whole.insert(0, "k", k)
                full20_whole.insert(0, "seed", seed)
                full20_whole.insert(0, "subject_id", subject_id)
                full20_subject_parts.append(full20_whole)

                full20_region = eligible.groupby(
                    ["anchor_region", "distance_bin_index", "distance_bin"],
                    as_index=False, observed=True,
                ).agg(
                    profile_dissimilarity=("median_dissimilarity", "median"),
                    profile_mae=("median_profile_mae", "median"),
                    symmetric_relative_mae_pct=(
                        "median_symmetric_relative_mae_pct", "median"
                    ),
                    median_geodesic_distance_mm=(
                        "median_geodesic_distance_mm", "median"
                    ),
                    anchor_count=("anchor_vertex", "size"),
                    pair_count=("n_neighbors_used", "sum"),
                    hemispheres_contributing=("hemisphere", "nunique"),
                )
                full20_region.insert(0, "k", k)
                full20_region.insert(0, "seed", seed)
                full20_region.insert(0, "subject_id", subject_id)
                full20_region_parts.append(full20_region)

            if k >= 20:
                eligible30 = anchor.loc[anchor["n_candidates"].ge(30)]
                full30_whole = eligible30.groupby(
                    ["distance_bin_index", "distance_bin"],
                    as_index=False, observed=True,
                ).agg(
                    profile_dissimilarity=("median_dissimilarity", "median"),
                    profile_mae=("median_profile_mae", "median"),
                    symmetric_relative_mae_pct=(
                        "median_symmetric_relative_mae_pct", "median"
                    ),
                    median_geodesic_distance_mm=(
                        "median_geodesic_distance_mm", "median"
                    ),
                    anchor_count=("anchor_vertex", "size"),
                    pair_count=("n_neighbors_used", "sum"),
                )
                full30_whole.insert(0, "k", k)
                full30_whole.insert(0, "seed", seed)
                full30_whole.insert(0, "subject_id", subject_id)
                full30_subject_parts.append(full30_whole)

                full30_region = eligible30.groupby(
                    ["anchor_region", "distance_bin_index", "distance_bin"],
                    as_index=False, observed=True,
                ).agg(
                    profile_dissimilarity=("median_dissimilarity", "median"),
                    profile_mae=("median_profile_mae", "median"),
                    symmetric_relative_mae_pct=(
                        "median_symmetric_relative_mae_pct", "median"
                    ),
                    median_geodesic_distance_mm=(
                        "median_geodesic_distance_mm", "median"
                    ),
                    anchor_count=("anchor_vertex", "size"),
                    pair_count=("n_neighbors_used", "sum"),
                    hemispheres_contributing=("hemisphere", "nunique"),
                )
                full30_region.insert(0, "k", k)
                full30_region.insert(0, "seed", seed)
                full30_region.insert(0, "subject_id", subject_id)
                full30_region_parts.append(full30_region)

    reused_k_values: list[int] = []
    if args.existing_results_root is not None:
        existing_dir = args.existing_results_root / "subjects" / subject_id
        existing_subject = pd.read_csv(existing_dir / "k_convergence_subject_bins.csv")
        existing_region = pd.read_csv(existing_dir / "k_convergence_region_bins.csv")
        reused_k_values = sorted(set(existing_subject["k"].astype(int)) - set(new_k_values))
        if reused_k_values != [1, 3, 5, 10]:
            raise RuntimeError(
                f"Expected reusable K=[1,3,5,10] for {subject_id}, found {reused_k_values}"
            )
        subject_parts.insert(
            0, existing_subject.loc[existing_subject["k"].isin(reused_k_values)].copy()
        )
        region_parts.insert(
            0, existing_region.loc[existing_region["k"].isin(reused_k_values)].copy()
        )

    pd.concat(subject_parts, ignore_index=True).to_csv(
        output_dir / "k_convergence_subject_bins.csv", index=False
    )
    pd.concat(region_parts, ignore_index=True).to_csv(
        output_dir / "k_convergence_region_bins.csv", index=False
    )
    pd.concat(full20_subject_parts, ignore_index=True).to_csv(
        output_dir / "k_convergence_full20_subject_bins.csv", index=False
    )
    pd.concat(full20_region_parts, ignore_index=True).to_csv(
        output_dir / "k_convergence_full20_region_bins.csv", index=False
    )
    pd.concat(full30_subject_parts, ignore_index=True).to_csv(
        output_dir / "k_convergence_full30_subject_bins.csv", index=False
    )
    pd.concat(full30_region_parts, ignore_index=True).to_csv(
        output_dir / "k_convergence_full30_region_bins.csv", index=False
    )
    (output_dir / "completion.json").write_text(json.dumps({
        "status": "complete", "subject_id": subject_id,
        "k_values": sorted(reused_k_values + list(new_k_values)),
        "new_k_values_computed": list(new_k_values),
        "reused_k_values": reused_k_values,
        "existing_results_root": (
            str(args.existing_results_root) if args.existing_results_root else None
        ),
        "diagnostic_k10_recomputed_only_for_n_candidates_ge_20": bool(
            args.diagnostic_k10
        ),
        "n_seeds": len(list(args.pair_dir.glob("seed_*.parquet"))),
        "runtime_seconds": time.time() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
