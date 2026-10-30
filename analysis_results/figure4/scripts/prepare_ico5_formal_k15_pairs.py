#!/usr/bin/env python3
"""Freeze the formal Figure 4 ico5 within-region K=15 pair table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pair-file", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--pair-seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    validation = json.loads(args.source_validation.read_text())
    if not validation.get("all_old_rows_identical", False):
        raise RuntimeError("The extended nested pair table did not pass rank validation")
    pairs = pd.read_parquet(
        args.source_pair_file,
        filters=[("seed", "==", args.pair_seed), ("sample_rank", "<=", args.k)],
    )
    if pairs.empty:
        raise RuntimeError("No formal pairs were selected")
    if not pairs["seed"].eq(args.pair_seed).all():
        raise RuntimeError("Unexpected pair-sampling seed")
    if pairs["sample_rank"].max() > args.k:
        raise RuntimeError("Pair table contains ranks above formal K")
    duplicate_key = [
        "hemisphere", "anchor_vertex", "distance_bin_index", "sample_rank"
    ]
    if pairs.duplicated(duplicate_key).any():
        raise RuntimeError("Duplicate nested ranks in formal pair table")
    pairs["pair_type"] = "within_region"
    hemisphere_offset = pairs["hemisphere"].map({"lh": 0, "rh": 1}).astype(int)
    pairs["split_half"] = 1 + (
        (pairs["ico5_vertex_id"].astype(int) + hemisphere_offset) % 2
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "figure4_geodesic_pairs_all.parquet"
    pairs.to_parquet(output, index=False, compression="zstd")
    anchor_counts = (
        pairs.assign(
            hemisphere_anchor=(
                pairs["hemisphere"].astype(str)
                + ":"
                + pairs["anchor_vertex"].astype(str)
            )
        )
        .groupby(
            ["distance_bin_index", "distance_bin", "hemisphere_anchor"],
            as_index=False, observed=True,
        )
        .agg(
            pair_count=("neighbor_vertex", "size"),
            median_distance_mm=("geodesic_distance_mm", "median"),
        )
    )
    counts = (
        anchor_counts
        .groupby(["distance_bin_index", "distance_bin"], as_index=False)
        .agg(
            anchor_count=("hemisphere_anchor", "nunique"),
            pair_count=("pair_count", "sum"),
            median_distance_mm=("median_distance_mm", "median"),
            minimum_neighbors=("pair_count", "min"),
            median_neighbors=("pair_count", "median"),
        )
    )
    counts.to_csv(args.output_dir / "formal_pair_counts.csv", index=False)
    report = {
        "status": "complete",
        "anchor_definition": "official_fsaverage5_vertices_mapped_one_to_one_to_fsaverage",
        "candidate_definition": "same_hemisphere_same_DK_region_within_2mm_distance_bin",
        "pair_sampling": "nested_without_replacement",
        "pair_seed": args.pair_seed,
        "k": args.k,
        "distance_bins_mm": list(range(0, 32, 2)),
        "n_pairs": int(len(pairs)),
        "n_anchors": int(
            pairs[["hemisphere", "anchor_vertex"]].drop_duplicates().shape[0]
        ),
        "source_pair_file": str(args.source_pair_file),
        "source_validation": str(args.source_validation),
        "source_old_rows_identical": True,
        "output_pair_file": str(output),
        "output_sha256": sha256(output),
    }
    (args.output_dir / "formal_pair_table.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
