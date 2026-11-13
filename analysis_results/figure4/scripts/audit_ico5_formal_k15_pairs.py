#!/usr/bin/env python3
"""Audit and freeze provenance for the formal ico5 within-DK K=15 pairs.

Description
-----------
Audit and freeze provenance for the formal ico5 within-DK K=15 pairs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--pair-file`` command-line input
- ``--mapping-file`` command-line input
- ``--candidate-counts`` command-line input
- ``--fsaverage-dir`` command-line input
- ``--real-results-root`` command-line input
- ``--null-results-root`` command-line input
- ``--manifest`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/audit_ico5_formal_k15_pairs.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--candidate-counts", type=Path, required=True)
    parser.add_argument("--fsaverage-dir", type=Path, required=True)
    parser.add_argument("--real-results-root", type=Path, required=True)
    parser.add_argument("--null-results-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20261021)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(args.pair_file)
    mapping = pd.read_csv(args.mapping_file)
    candidates = pd.read_parquet(args.candidate_counts)

    self_pairs = int(pairs.anchor_vertex.eq(pairs.neighbor_vertex).sum())
    duplicate_candidate_rows = int(pairs.duplicated([
        "hemisphere", "anchor_vertex", "distance_bin_index", "neighbor_vertex"
    ]).sum())
    directed_bin_counts = pairs.groupby(
        ["hemisphere", "anchor_vertex", "neighbor_vertex"], observed=True
    ).distance_bin_index.nunique()
    directed_pairs_across_bins = int(directed_bin_counts.gt(1).sum())
    mapping_ico_collisions = int(mapping.duplicated([
        "hemisphere", "ico5_vertex_id"
    ]).sum())
    mapping_full_collisions = int(mapping.duplicated([
        "hemisphere", "full_resolution_vertex_id"
    ]).sum())

    cross_region_pairs = 0
    for hemisphere in ("lh", "rh"):
        labels, _, _ = fsio.read_annot(
            str(args.fsaverage_dir / "label" / f"{hemisphere}.aparc.annot")
        )
        current = pairs.loc[pairs.hemisphere.eq(hemisphere)]
        cross_region_pairs += int((
            labels[current.anchor_vertex.to_numpy(dtype=int)]
            != labels[current.neighbor_vertex.to_numpy(dtype=int)]
        ).sum())

    effective = (
        pairs.groupby([
            "hemisphere", "anchor_vertex", "anchor_region",
            "distance_bin_index", "distance_bin",
        ], as_index=False, observed=True)
        .agg(k_effective=("neighbor_vertex", "size"))
    )
    expected_anchor_bins = candidates.loc[candidates.n_candidates.gt(0), [
        "hemisphere", "anchor_vertex", "anchor_region",
        "distance_bin_index", "distance_bin", "n_candidates",
    ]]
    effective = expected_anchor_bins.merge(
        effective,
        on=["hemisphere", "anchor_vertex", "anchor_region",
            "distance_bin_index", "distance_bin"],
        how="left", validate="one_to_one",
    )
    effective["k_effective"] = effective.k_effective.fillna(0).astype(int)
    expected_k = np.minimum(15, effective.n_candidates.to_numpy(dtype=int))
    effective_k_mismatch = int((effective.k_effective.to_numpy() != expected_k).sum())
    effective.to_parquet(args.output_dir / "formal_k_effective_anchor_bins.parquet", index=False)
    coverage = (
        effective.groupby(["distance_bin_index", "distance_bin"], as_index=False)
        .agg(
            anchor_bins=("anchor_vertex", "size"),
            k_effective_min=("k_effective", "min"),
            k_effective_q25=("k_effective", lambda x: x.quantile(.25)),
            k_effective_median=("k_effective", "median"),
            k_effective_q75=("k_effective", lambda x: x.quantile(.75)),
            k_effective_max=("k_effective", "max"),
            full_k15_fraction=("k_effective", lambda x: (x == 15).mean()),
        )
    )
    coverage.to_csv(args.output_dir / "formal_k15_coverage_by_distance.csv", index=False)
    distribution = (
        effective.groupby(["distance_bin_index", "distance_bin", "k_effective"],
                          as_index=False, observed=True)
        .size().rename(columns={"size": "anchor_bin_count"})
    )
    distribution["fraction"] = distribution.anchor_bin_count / distribution.groupby(
        "distance_bin_index"
    ).anchor_bin_count.transform("sum")
    distribution.to_csv(args.output_dir / "formal_k_effective_distribution.csv", index=False)

    anchors = pairs[[
        "hemisphere", "anchor_vertex", "ico5_vertex_id",
        "anchor_dk_label", "anchor_region",
    ]].drop_duplicates()
    rng = np.random.default_rng(args.split_seed)
    split_parts = []
    for _, table in anchors.groupby(
        ["hemisphere", "anchor_dk_label"], observed=True, sort=True
    ):
        table = table.sort_values("anchor_vertex").copy()
        order = rng.permutation(len(table))
        halves = np.empty(len(table), dtype=np.int8)
        halves[order[::2]] = 1
        halves[order[1::2]] = 2
        table["split_half"] = halves
        split_parts.append(table)
    assignment = pd.concat(split_parts, ignore_index=True)
    assignment.to_csv(args.output_dir / "formal_fixed_stratified_split_assignment.csv", index=False)
    split_balance_max_difference = int(
        assignment.groupby(["hemisphere", "anchor_dk_label", "split_half"], observed=True)
        .size().unstack(fill_value=0).diff(axis=1).abs().max().max()
    )

    expected = pd.read_csv(args.manifest, dtype={"subject_id": str}).subject_id
    real_null_curve_max_abs_difference = 0.0
    count_mismatch_rows = 0
    compared_subjects = 0
    for subject_id in expected:
        real_path = args.real_results_root / "subjects" / subject_id / "figure4_ico5_k15_real_subject_bins.csv"
        null_path = args.null_results_root / "subjects" / subject_id / "figure4_null_subject_bins.csv"
        if not real_path.is_file() or not null_path.is_file():
            continue
        real = pd.read_csv(real_path).sort_values("distance_bin_index")
        null = pd.read_csv(null_path)
        first = null.loc[null.replicate.eq(0)].sort_values("distance_bin_index")
        real_null_curve_max_abs_difference = max(
            real_null_curve_max_abs_difference,
            float(np.max(np.abs(
                real.median_dissimilarity.to_numpy()
                - first.real_dissimilarity.to_numpy()
            ))),
        )
        count_mismatch_rows += int((
            (real.n_anchors.to_numpy() != first.real_anchor_count.to_numpy())
            | (real.n_pairs.to_numpy() != first.real_pair_count.to_numpy())
            | (first.anchor_count.to_numpy() != first.real_anchor_count.to_numpy())
            | (first.pair_count.to_numpy() != first.real_pair_count.to_numpy())
        ).sum())
        compared_subjects += 1

    checks = {
        "self_pair_count": self_pairs,
        "duplicate_candidate_within_anchor_bin_count": duplicate_candidate_rows,
        "directed_pair_repeated_across_bins_count": directed_pairs_across_bins,
        "ico5_to_full_mapping_collision_count": mapping_ico_collisions,
        "full_to_ico5_mapping_collision_count": mapping_full_collisions,
        "cross_region_pair_count": cross_region_pairs,
        "effective_k_mismatch_count": effective_k_mismatch,
        "real_null_subjects_compared": compared_subjects,
        "real_null_count_mismatch_rows": count_mismatch_rows,
        "real_null_curve_max_absolute_difference": real_null_curve_max_abs_difference,
        "split_assignment_anchor_count": int(len(assignment)),
        "split_assignment_max_stratum_count_difference": split_balance_max_difference,
    }
    zero_required = [
        self_pairs, duplicate_candidate_rows, directed_pairs_across_bins,
        mapping_ico_collisions, mapping_full_collisions, cross_region_pairs,
        effective_k_mismatch, count_mismatch_rows,
    ]
    report = {
        "status": "pass" if all(value == 0 for value in zero_required) else "fail",
        "formal_pair_file": str(args.pair_file),
        "anchor_definition": "official_fsaverage5_fixed",
        "candidate_definition": "within_DK_region",
        "k": 15,
        "pair_seed": int(pairs.seed.iat[0]),
        "directed_pair_count": int(len(pairs)),
        "unique_anchor_count": int(len(anchors)),
        "split_seed": args.split_seed,
        **checks,
    }
    (args.output_dir / "formal_pair_qc.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise RuntimeError("Formal pair QC failed")


if __name__ == "__main__":
    main()
