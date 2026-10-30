#!/usr/bin/env python3
"""Extend existing nested ico5 pairs without changing any existing ranks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse.csgraph import dijkstra

from build_geodesic_pair_tables import DISTANCE_EDGES, build_graph, valid_template_mask
from build_ico5_within_pairs import (
    HEMISPHERES, annotation_names, ico5_mapping, stable_rng,
)


VALIDATION_COLUMNS = [
    "seed", "sample_rank", "hemisphere", "ico5_vertex_id", "anchor_vertex",
    "neighbor_vertex", "geodesic_distance_mm", "distance_bin_index",
    "distance_bin", "anchor_dk_label", "anchor_region", "n_candidates",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--fsaverage5-dir", type=Path, required=True)
    parser.add_argument("--old-pair-dir", type=Path, required=True)
    parser.add_argument("--output-pair-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20261020)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--old-max-rank", type=int, default=10)
    parser.add_argument("--new-max-rank", type=int, default=20)
    return parser.parse_args()


def generated_batch(
    batch: np.ndarray,
    batch_ico: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    hemisphere: str,
    hemisphere_index: int,
    n_seeds: int,
    base_seed: int,
    maximum_rank: int,
) -> dict[int, pd.DataFrame]:
    by_seed: dict[int, list[pd.DataFrame]] = {seed: [] for seed in range(n_seeds)}
    for local_index, (ico_id, anchor) in enumerate(zip(batch_ico, batch)):
        current = distances[local_index]
        region_label = int(labels[anchor])
        for bin_index, (lower, upper) in enumerate(
            zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])
        ):
            candidates = np.flatnonzero(
                mask & (labels == region_label)
                & (current > lower) & (current <= upper)
            )
            candidates = candidates[candidates != anchor]
            if len(candidates) == 0:
                continue
            maximum = min(maximum_rank, len(candidates))
            for seed in range(n_seeds):
                rng = stable_rng(base_seed, seed, hemisphere_index, int(anchor), bin_index)
                selected = candidates[rng.permutation(len(candidates))[:maximum]]
                by_seed[seed].append(pd.DataFrame({
                    "seed": seed,
                    "sample_rank": np.arange(1, maximum + 1, dtype=np.int8),
                    "hemisphere": hemisphere,
                    "ico5_vertex_id": int(ico_id),
                    "anchor_vertex": int(anchor),
                    "neighbor_vertex": selected.astype(np.int32),
                    "geodesic_distance_mm": current[selected].astype(np.float32),
                    "distance_bin_index": np.int8(bin_index),
                    "distance_bin": f"{lower:.0f}-{upper:.0f}",
                    "anchor_dk_label": region_label,
                    "anchor_region": names[region_label],
                    "n_candidates": int(len(candidates)),
                }))
    return {
        seed: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=VALIDATION_COLUMNS)
        for seed, parts in by_seed.items()
    }


def assert_old_rows_unchanged(old: pd.DataFrame, regenerated: pd.DataFrame,
                              seed: int, row_group: int,
                              old_max_rank: int) -> None:
    expected = regenerated.loc[
        regenerated["sample_rank"].le(old_max_rank), VALIDATION_COLUMNS
    ]
    observed = old.loc[:, VALIDATION_COLUMNS]
    if len(expected) != len(observed):
        raise RuntimeError(
            f"seed={seed} row_group={row_group}: old/regenerated row counts differ "
            f"({len(observed)} vs {len(expected)})"
        )
    for column in VALIDATION_COLUMNS:
        left = observed[column].to_numpy()
        right = expected[column].to_numpy()
        if column == "geodesic_distance_mm":
            identical = np.array_equal(left.astype(np.float32), right.astype(np.float32))
        else:
            identical = np.array_equal(left, right)
        if not identical:
            mismatch = int(np.flatnonzero(left != right)[0])
            raise RuntimeError(
                f"seed={seed} row_group={row_group} column={column} "
                f"first mismatch row={mismatch}: {left[mismatch]!r} != {right[mismatch]!r}"
            )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_pair_dir.mkdir(parents=True, exist_ok=True)
    if args.new_max_rank <= args.old_max_rank:
        raise ValueError("new-max-rank must be greater than old-max-rank")
    template_root = args.subjects_dir / args.template
    readers = {
        seed: pq.ParquetFile(args.old_pair_dir / f"seed_{seed:02d}.parquet")
        for seed in range(args.n_seeds)
    }
    writers = {
        seed: pq.ParquetWriter(
            args.output_pair_dir / f"seed_{seed:02d}.parquet",
            readers[seed].schema_arrow,
            compression="zstd",
        )
        for seed in range(args.n_seeds)
    }
    row_group = 0
    old_rows_validated = 0
    added_rows = 0
    try:
        for hemisphere_index, (hemisphere, side) in enumerate(HEMISPHERES):
            white, faces = fsio.read_geometry(
                str(template_root / "surf" / f"{hemisphere}.white")
            )
            mask, labels = valid_template_mask(template_root, hemisphere)
            _, names = annotation_names(template_root, hemisphere)
            mapped, _, _ = ico5_mapping(
                template_root, args.fsaverage5_dir, hemisphere, side
            )
            anchors = mapped[mask[mapped]]
            ico_ids = np.flatnonzero(mask[mapped])
            graph = build_graph(white, faces)
            for batch_start in range(0, len(anchors), args.batch_size):
                batch = anchors[batch_start:batch_start + args.batch_size]
                batch_ico = ico_ids[batch_start:batch_start + args.batch_size]
                distances = dijkstra(graph, directed=False, indices=batch, limit=30.0)
                regenerated = generated_batch(
                    batch, batch_ico, distances, mask, labels, names,
                    hemisphere, hemisphere_index, args.n_seeds, args.base_seed,
                    args.new_max_rank,
                )
                for seed in range(args.n_seeds):
                    if row_group >= readers[seed].num_row_groups:
                        raise RuntimeError(f"seed={seed}: missing old row group {row_group}")
                    old = readers[seed].read_row_group(row_group).to_pandas()
                    assert_old_rows_unchanged(
                        old, regenerated[seed], seed, row_group, args.old_max_rank
                    )
                    extension = regenerated[seed].loc[
                        regenerated[seed]["sample_rank"].gt(args.old_max_rank)
                    ]
                    combined = pd.concat([old, extension], ignore_index=True)
                    table = pa.Table.from_pandas(
                        combined,
                        schema=readers[seed].schema_arrow,
                        preserve_index=False,
                        safe=True,
                    )
                    writers[seed].write_table(table)
                    old_rows_validated += len(old)
                    added_rows += len(extension)
                row_group += 1
                logging.info(
                    "%s anchors %d/%d; validated row group %d",
                    hemisphere, min(batch_start + len(batch), len(anchors)),
                    len(anchors), row_group,
                )
        for seed, reader in readers.items():
            if row_group != reader.num_row_groups:
                raise RuntimeError(
                    f"seed={seed}: consumed {row_group}/{reader.num_row_groups} row groups"
                )
    finally:
        for writer in writers.values():
            writer.close()

    args.report.write_text(json.dumps({
        "status": "complete",
        "old_pair_dir": str(args.old_pair_dir),
        "output_pair_dir": str(args.output_pair_dir),
        "old_ranks_validated": [1, args.old_max_rank],
        "new_ranks_added": [args.old_max_rank + 1, args.new_max_rank],
        "validation_columns": VALIDATION_COLUMNS,
        "n_seeds": args.n_seeds,
        "n_row_groups_per_seed": row_group,
        "old_rows_validated_across_all_seeds": old_rows_validated,
        "new_rows_added_across_all_seeds": added_rows,
        "all_old_rows_identical": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
