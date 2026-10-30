#!/usr/bin/env python3
"""Generate subject-specific Figure 4D matched within/across pair tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_matching import (
    deterministic_distance_match,
    filter_candidate_frame,
    label_candidate_pools,
    valid_profile_mask,
)
from compute_real_spatial_curves import load_profiles, select_subject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--caliper-mm", type=float, default=0.1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def match_subject_hemisphere(
    subject_id: str,
    hemisphere: str,
    candidates: pd.DataFrame,
    profile_valid: np.ndarray,
    k: int,
    caliper_mm: float,
) -> pd.DataFrame:
    candidates = filter_candidate_frame(candidates, profile_valid)
    keys = [
        "anchor_ico5_id", "anchor_vertex", "anchor_dk_label", "anchor_region",
        "distance_bin_index", "distance_bin",
    ]
    parts = []
    for values, frame in candidates.groupby(keys, observed=True, sort=True):
        metadata = dict(zip(keys, values))
        within, across = label_candidate_pools(
            frame, int(metadata["anchor_dk_label"])
        )
        n_within = len(within)
        n_across = len(across)
        requested = min(k, n_within, n_across)
        if requested == 0:
            continue
        matched = deterministic_distance_match(
            within, across, maximum_matches=requested, caliper_mm=caliper_mm
        )
        if matched.empty:
            continue
        matched.insert(0, "subject_id", subject_id)
        matched.insert(1, "hemisphere", hemisphere)
        insertion = 2
        for column in keys:
            matched.insert(insertion, column, metadata[column])
            insertion += 1
        matched["n_within_candidates"] = n_within
        matched["n_across_candidates"] = n_across
        matched["k_effective_requested"] = requested
        matched["n_matched"] = len(matched)
        parts.append(matched)
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    ordered = [
        "subject_id", "hemisphere", "anchor_vertex", "anchor_ico5_id",
        "anchor_dk_label", "anchor_region", "distance_bin_index", "distance_bin",
        "match_id", "within_candidate_vertex", "within_candidate_dk_label",
        "within_distance_mm", "across_candidate_vertex",
        "across_candidate_dk_label", "across_distance_mm",
        "absolute_distance_difference_mm", "n_within_candidates",
        "n_across_candidates", "k_effective_requested", "n_matched",
    ]
    return result[ordered]


def exploded_directed_pairs(matched: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for pair_type, candidate_column in (
        ("within", "within_candidate_vertex"),
        ("across", "across_candidate_vertex"),
    ):
        part = matched[
            [
                "hemisphere", "anchor_vertex", "distance_bin_index",
                "distance_bin", candidate_column,
            ]
        ].rename(columns={candidate_column: "candidate_vertex"})
        part["pair_type"] = pair_type
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def audit_matches(matched: pd.DataFrame, caliper_mm: float) -> tuple[dict, pd.DataFrame]:
    anchor_bin_keys = [
        "hemisphere", "anchor_vertex", "distance_bin_index", "distance_bin"
    ]
    directed = exploded_directed_pairs(matched)
    duplicate_rows = int(matched.duplicated().sum())
    self_pairs = int(
        matched["anchor_vertex"].eq(matched["within_candidate_vertex"]).sum()
        + matched["anchor_vertex"].eq(matched["across_candidate_vertex"]).sum()
    )
    within_duplicate = int(
        matched.duplicated(anchor_bin_keys + ["within_candidate_vertex"]).sum()
    )
    across_duplicate = int(
        matched.duplicated(anchor_bin_keys + ["across_candidate_vertex"]).sum()
    )
    cross_bin = (
        directed.groupby(
            ["hemisphere", "pair_type", "anchor_vertex", "candidate_vertex"],
            observed=True,
        )["distance_bin_index"].nunique()
    )
    cross_bin_duplicates = int(cross_bin.gt(1).sum())
    directed_duplicates = int(
        directed.duplicated(
            [
                "hemisphere", "pair_type", "anchor_vertex", "candidate_vertex",
                "distance_bin_index",
            ]
        ).sum()
    )
    within_violations = int(
        matched["within_candidate_dk_label"].ne(matched["anchor_dk_label"]).sum()
    )
    across_violations = int(
        matched["across_candidate_dk_label"].eq(matched["anchor_dk_label"]).sum()
    )
    directed_keys = set(
        zip(
            directed["hemisphere"], directed["pair_type"],
            directed["anchor_vertex"], directed["candidate_vertex"],
        )
    )
    reciprocal_directed_rows = sum(
        (hemisphere, pair_type, candidate, anchor) in directed_keys
        for hemisphere, pair_type, anchor, candidate in directed_keys
    )
    maximum_mismatch = float(matched["absolute_distance_difference_mm"].max())
    audit = {
        "self_pairs": self_pairs,
        "duplicate_rows": duplicate_rows,
        "within_candidate_duplicates_within_anchor_bin": within_duplicate,
        "across_candidate_duplicates_within_anchor_bin": across_duplicate,
        "duplicate_directed_pairs": directed_duplicates,
        "cross_bin_duplicates": cross_bin_duplicates,
        "within_label_violations": within_violations,
        "across_label_violations": across_violations,
        "maximum_distance_difference_mm": maximum_mismatch,
        "caliper_mm": caliper_mm,
        "reciprocal_directed_rows": int(reciprocal_directed_rows),
        "reciprocal_pair_count": int(reciprocal_directed_rows // 2),
    }
    assert self_pairs == 0
    assert duplicate_rows == 0
    assert within_duplicate == 0
    assert across_duplicate == 0
    assert directed_duplicates == 0
    assert cross_bin_duplicates == 0
    assert within_violations == 0
    assert across_violations == 0
    assert maximum_mismatch <= caliper_mm + 1e-8

    anchor_bins = matched.drop_duplicates(anchor_bin_keys + ["n_matched"])
    rows = []
    for keys, frame in matched.groupby(
        ["distance_bin_index", "distance_bin"], observed=True, sort=True
    ):
        bin_index, bin_label = keys
        current_anchor_bins = anchor_bins.loc[
            anchor_bins["distance_bin_index"].eq(bin_index)
        ]
        n_matched = current_anchor_bins["n_matched"].to_numpy(float)
        rows.append(
            {
                "distance_bin_index": bin_index,
                "distance_bin": bin_label,
                "matched_pair_rows": len(frame),
                "matched_anchor_count": len(current_anchor_bins),
                "matched_region_count": frame["anchor_dk_label"].nunique(),
                "n_matched_median": np.median(n_matched),
                "n_matched_q25": np.quantile(n_matched, 0.25),
                "n_matched_q75": np.quantile(n_matched, 0.75),
                "full_k15_anchor_coverage": np.mean(n_matched == 15),
                "distance_mismatch_median_mm": frame[
                    "absolute_distance_difference_mm"
                ].median(),
                "distance_mismatch_q95_mm": frame[
                    "absolute_distance_difference_mm"
                ].quantile(0.95),
                "distance_mismatch_maximum_mm": frame[
                    "absolute_distance_difference_mm"
                ].max(),
                "within_distance_median_mm": frame["within_distance_mm"].median(),
                "within_distance_q25_mm": frame["within_distance_mm"].quantile(0.25),
                "within_distance_q75_mm": frame["within_distance_mm"].quantile(0.75),
                "across_distance_median_mm": frame["across_distance_mm"].median(),
                "across_distance_q25_mm": frame["across_distance_mm"].quantile(0.25),
                "across_distance_q75_mm": frame["across_distance_mm"].quantile(0.75),
            }
        )
    return audit, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.k <= 0 or args.caliper_mm <= 0:
        raise ValueError("K and caliper must be positive")
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    started = time.time()
    try:
        parts = []
        mask_rows = []
        for hemisphere in ("lh", "rh"):
            profiles, base_valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            profile_valid = valid_profile_mask(profiles, base_valid)
            candidate_path = (
                args.candidate_dir / f"figure4D_dense_candidates_{hemisphere}.parquet"
            )
            candidates = pd.read_parquet(candidate_path)
            parts.append(
                match_subject_hemisphere(
                    subject_id, hemisphere, candidates, profile_valid,
                    args.k, args.caliper_mm,
                )
            )
            mask_rows.append(
                {
                    "subject_id": subject_id,
                    "hemisphere": hemisphere,
                    "base_valid_vertices": int(base_valid.sum()),
                    "nonconstant_valid_vertices": int(profile_valid.sum()),
                    "constant_or_nonfinite_excluded": int(
                        base_valid.sum() - profile_valid.sum()
                    ),
                }
            )
        matched = pd.concat([part for part in parts if not part.empty], ignore_index=True)
        if matched.empty:
            raise RuntimeError("No matched within/across comparisons")
        audit, bin_qc = audit_matches(matched, args.caliper_mm)
        output_path = output_dir / "figure4D_boundary_matched_pairs.parquet"
        matched.to_parquet(output_path, index=False, compression="zstd")
        bin_qc.insert(0, "subject_id", subject_id)
        bin_qc.to_csv(output_dir / "figure4D_boundary_matching_qc_by_bin.csv", index=False)
        pd.DataFrame(mask_rows).to_csv(
            output_dir / "figure4D_profile_validity_qc.csv", index=False
        )
        audit.update(
            {
                "status": "pass",
                "subject_id": subject_id,
                "matched_rows": int(len(matched)),
                "matched_anchor_bins": int(
                    matched[
                        ["hemisphere", "anchor_vertex", "distance_bin_index"]
                    ].drop_duplicates().shape[0]
                ),
                "k": args.k,
                "pair_table": str(output_path),
                "pair_table_sha256": sha256(output_path),
            }
        )
        (output_dir / "figure4D_boundary_matching_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        completion.write_text(
            json.dumps(
                {
                    "status": "complete", "subject_id": subject_id,
                    "stage": "boundary_pair_generation", "k": args.k,
                    "caliper_mm": args.caliper_mm,
                    "matched_rows": int(len(matched)),
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
                    "stage": "boundary_pair_generation",
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
