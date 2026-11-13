#!/usr/bin/env python3
"""Compute real, bilateral DK-region curves for the frozen ico5 K=15 design.

Description
-----------
The hierarchy is pair -> hemisphere x anchor x distance-bin median ->
subject x bilateral DK region x distance-bin median.  Every fixed anchor-bin,
including those with zero valid profile pairs for a subject, contributes to the
K-effective coverage diagnostics.  Pair counts never weight a curve estimate.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-file`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_ico5_region_real.py --help
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
from spatial_metrics import pair_metrics


K_TARGET = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    return parser.parse_args()


def anchor_bins_with_coverage(
    profiles: np.ndarray,
    valid_vertices: np.ndarray,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "hemisphere", "anchor_vertex", "anchor_dk_label", "anchor_region",
        "distance_bin_index", "distance_bin",
    ]
    fixed = (
        pairs.groupby(keys, as_index=False, observed=True)
        .agg(
            fixed_sampled_k=("neighbor_vertex", "size"),
            candidate_count=("n_candidates", "first"),
        )
    )
    metrics = pair_metrics(profiles, valid_vertices, pairs)
    valid = metrics.loc[metrics["pair_valid"]].copy()
    observed = (
        valid.groupby(keys, as_index=False, observed=True)
        .agg(
            median_dissimilarity=("profile_dissimilarity", "median"),
            median_profile_mae=("profile_mae", "median"),
            median_profile_rmse=("profile_rmse", "median"),
            median_geodesic_distance_mm=("geodesic_distance_mm", "median"),
            k_effective=("profile_dissimilarity", "size"),
        )
    )
    anchor = fixed.merge(observed, on=keys, how="left", validate="one_to_one")
    anchor["k_effective"] = anchor["k_effective"].fillna(0).astype(np.int16)
    anchor["has_valid_pair"] = anchor["k_effective"].gt(0)
    anchor["full_k15"] = anchor["k_effective"].eq(K_TARGET)
    return anchor


def bilateral_region_bins(anchor: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    keys = [
        "anchor_dk_label", "anchor_region", "distance_bin_index", "distance_bin"
    ]
    rows = []
    for values, frame in anchor.groupby(keys, observed=True, sort=True):
        valid = frame.loc[np.isfinite(frame["median_dissimilarity"])]
        k_effective = frame["k_effective"].to_numpy(dtype=float)
        row = dict(zip(keys, values))
        row.update(
            {
                "subject_id": subject_id,
                "median_dissimilarity": valid["median_dissimilarity"].median(),
                "median_profile_mae": valid["median_profile_mae"].median(),
                "median_profile_rmse": valid["median_profile_rmse"].median(),
                "median_geodesic_distance_mm": valid[
                    "median_geodesic_distance_mm"
                ].median(),
                "expected_anchor_count": int(len(frame)),
                "valid_anchor_count": int(frame["has_valid_pair"].sum()),
                "valid_pair_count": int(frame["k_effective"].sum()),
                "median_k_effective": float(np.median(k_effective)),
                "q25_k_effective": float(np.quantile(k_effective, 0.25)),
                "q75_k_effective": float(np.quantile(k_effective, 0.75)),
                "mean_k_effective": float(np.mean(k_effective)),
                "full_k15_anchor_count": int(frame["full_k15"].sum()),
                "full_k15_coverage": float(frame["full_k15"].mean()),
                "any_valid_anchor_coverage": float(frame["has_valid_pair"].mean()),
                "hemispheres_expected": int(frame["hemisphere"].nunique()),
                "hemispheres_contributing": int(valid["hemisphere"].nunique()),
            }
        )
        rows.append(row)
    columns = ["subject_id", *keys]
    result = pd.DataFrame(rows)
    return result[columns + [column for column in result if column not in columns]]


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
        pairs = pd.read_parquet(args.pair_file)
        required = {
            "hemisphere", "anchor_vertex", "neighbor_vertex", "anchor_dk_label",
            "anchor_region", "distance_bin_index", "distance_bin", "n_candidates",
            "pair_type", "sample_rank",
        }
        missing = required.difference(pairs.columns)
        if missing:
            raise ValueError(f"Formal pair table lacks columns: {sorted(missing)}")
        if not pairs["pair_type"].eq("within_region").all():
            raise ValueError("Region analysis requires only within-region pairs")
        if int(pairs["sample_rank"].max()) != K_TARGET:
            raise ValueError(f"Expected frozen K={K_TARGET}")

        anchor_parts = []
        for hemisphere in ("lh", "rh"):
            profiles, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            current = pairs.loc[pairs["hemisphere"].eq(hemisphere)].reset_index(
                drop=True
            )
            anchor_parts.append(anchor_bins_with_coverage(profiles, valid, current))
        anchor = pd.concat(anchor_parts, ignore_index=True)
        anchor.insert(0, "subject_id", subject_id)
        anchor.to_parquet(
            output_dir / "figure4_region_real_anchor_bins.parquet",
            index=False,
            compression="zstd",
        )
        subject_region = bilateral_region_bins(anchor, subject_id)
        subject_region.to_csv(
            output_dir / "figure4_region_real_subject_bins.csv", index=False
        )
        completion.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "subject_id": subject_id,
                    "anchor_definition": "official_fsaverage5_fixed",
                    "candidate_definition": "within_DK_region",
                    "k": K_TARGET,
                    "pair_seed": int(pairs["seed"].iat[0]),
                    "bilateral_regions": int(subject_region["anchor_region"].nunique()),
                    "aggregation": (
                        "pair -> hemisphere x anchor x bin median -> "
                        "subject x bilateral DK region x bin median"
                    ),
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    ),
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
                    "status": "failed",
                    "subject_id": subject_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
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
