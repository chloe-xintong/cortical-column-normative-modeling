#!/usr/bin/env python3
"""Compute a whole-profile spatial-permutation null for Figure 4B.

Description
-----------
Compute a whole-profile spatial-permutation null for Figure 4B.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-dir`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_whole_profile_permutation_null.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compute_spatial_null import subject_curve
from surface_pca_surrogate import load_real_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    return parser.parse_args()


def select_subject(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.index is not None:
        return manifest.iloc[args.index]
    rows = manifest.loc[manifest["subject_id"].eq(args.subject_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {args.subject_id}")
    return rows.iloc[0]


def stable_subject_seed(base_seed: int, subject_id: str) -> int:
    digest = hashlib.sha256(subject_id.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1))


def permute_complete_profiles(
    profiles: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Permute intact 21-depth rows bijectively within each hemisphere."""
    permuted: dict[str, np.ndarray] = {}
    qc_rows: list[dict[str, object]] = []
    for hemisphere in ("lh", "rh"):
        valid_ids = np.flatnonzero(valid[hemisphere])
        source_ids = rng.permutation(valid_ids)
        output = np.full_like(profiles[hemisphere], np.nan)
        output[valid_ids] = profiles[hemisphere][source_ids]
        permuted[hemisphere] = output
        qc_rows.append({
            "hemisphere": hemisphere,
            "valid_vertex_count": int(len(valid_ids)),
            "unique_source_count": int(np.unique(source_ids).size),
            "permutation_is_bijection": bool(
                np.array_equal(np.sort(source_ids), valid_ids)
            ),
            "fixed_point_count": int(np.count_nonzero(source_ids == valid_ids)),
        })
    return permuted, qc_rows


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    started = time.time()

    try:
        if not bool(row["input_complete"]):
            raise FileNotFoundError(f"Incomplete inputs: {row['missing_inputs']}")
        pairs = pd.read_parquet(args.pair_dir / "figure4_geodesic_pairs_all.parquet")
        profiles: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        for hemisphere in ("lh", "rh"):
            profiles[hemisphere], valid[hemisphere] = load_real_profiles(
                Path(row[f"clean_profile_{hemisphere}"])
            )

        real_curve = subject_curve(profiles, valid, pairs).rename(columns={
            "median_dissimilarity": "real_dissimilarity",
            "n_anchors": "real_anchor_count",
            "n_pairs": "real_pair_count",
        })
        subject_seed = stable_subject_seed(args.seed, subject_id)
        rng = np.random.default_rng(subject_seed)
        curve_rows = []
        qc_rows = []
        for replicate in range(args.replicates):
            logging.info(
                "[%s] Whole-profile permutation %d/%d",
                subject_id, replicate + 1, args.replicates,
            )
            null_profiles, replicate_qc = permute_complete_profiles(
                profiles, valid, rng
            )
            for qc in replicate_qc:
                qc.update({"subject_id": subject_id, "replicate": replicate})
            qc_rows.extend(replicate_qc)
            summary = subject_curve(null_profiles, valid, pairs).rename(columns={
                "median_dissimilarity": "null_dissimilarity",
                "n_anchors": "anchor_count",
                "n_pairs": "pair_count",
            })
            summary = summary.merge(
                real_curve[[
                    "distance_bin_index", "distance_bin", "pair_type",
                    "real_dissimilarity", "real_anchor_count", "real_pair_count",
                ]],
                on=["distance_bin_index", "distance_bin", "pair_type"],
                validate="one_to_one",
            )
            summary.insert(0, "replicate", replicate)
            summary.insert(0, "subject_id", subject_id)
            summary["null_minus_real"] = (
                summary["null_dissimilarity"] - summary["real_dissimilarity"]
            )
            far = summary["distance_bin_index"].between(10, 14)
            real_far = float(summary.loc[far, "real_dissimilarity"].median())
            null_far = float(summary.loc[far, "null_dissimilarity"].median())
            summary["real_local_excess_similarity"] = (
                real_far - summary["real_dissimilarity"]
            )
            summary["null_local_excess_similarity"] = (
                null_far - summary["null_dissimilarity"]
            )
            summary["real_minus_null_local_excess"] = (
                summary["real_local_excess_similarity"]
                - summary["null_local_excess_similarity"]
            )
            curve_rows.append(summary)

        output = pd.concat(curve_rows, ignore_index=True)
        output["null_method"] = "whole_profile_permutation"
        output["permutation_scope"] = "within_hemisphere_valid_vertices"
        output.to_csv(output_dir / "figure4_whole_profile_subject_bins.csv", index=False)
        pd.DataFrame(qc_rows).to_csv(
            output_dir / "figure4_whole_profile_permutation_qc.csv", index=False
        )
        if not pd.DataFrame(qc_rows)["permutation_is_bijection"].all():
            raise RuntimeError("A generated profile permutation was not bijective")
        completion.write_text(json.dumps({
            "subject_id": subject_id,
            "status": "complete",
            "null_method": "whole_profile_permutation",
            "permutation_unit": "complete_21_depth_profile",
            "permutation_scope": "within_hemisphere_valid_vertices",
            "base_seed": args.seed,
            "subject_seed": subject_seed,
            "replicates": args.replicates,
            "runtime_seconds": time.time() - started,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }, indent=2) + "\n", encoding="utf-8")
        failure.unlink(missing_ok=True)
    except Exception as error:
        failure.write_text(json.dumps({
            "subject_id": subject_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "runtime_seconds": time.time() - started,
        }, indent=2) + "\n", encoding="utf-8")
        completion.unlink(missing_ok=True)
        logging.exception("[%s] Whole-profile permutation null failed", subject_id)
        raise


if __name__ == "__main__":
    main()
