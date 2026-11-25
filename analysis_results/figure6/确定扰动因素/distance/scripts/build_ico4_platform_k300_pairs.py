#!/usr/bin/env python3
"""Extend the frozen platform-bin geometry sampling from K=30 to K=300."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse.csgraph import dijkstra


REPO = Path(__file__).resolve().parents[5]
FORMAL_SCRIPTS = REPO / "analysis_results/figure4/scripts"
sys.path.insert(0, str(FORMAL_SCRIPTS))
from build_geodesic_pair_tables import DISTANCE_EDGES, build_graph, valid_template_mask  # noqa: E402


BASE_SEED = 20261119
SEEDS = range(5)
PLATFORM_BINS = range(10, 15)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects-dir", type=Path, default=Path(
        "/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"))
    p.add_argument("--template", default="fsaverage")
    p.add_argument("--distance-root", type=Path, default=REPO /
        "analysis_results/figure6/确定扰动因素/distance")
    p.add_argument("--batch-size", type=int, default=24)
    return p.parse_args()


def rng_for(seed: int, hemi_index: int, anchor: int, bin_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [BASE_SEED, seed, hemi_index, int(anchor), bin_index]
    ))


def main() -> None:
    a = parse_args()
    source = a.distance_root / "local_similarity_stability"
    output = a.distance_root / "platform_sampling_sensitivity"
    pair_dir = output / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(source / "ico4_nodes.csv")
    counts = pd.read_parquet(source / "anchor_bin_template_counts.parquet")
    old_paths = {s: source / "pairs" / f"seed_{s:02d}_k30_nested.parquet" for s in SEEDS}
    new_paths = {s: pair_dir / f"seed_{s:02d}_platform_k300_nested.parquet" for s in SEEDS}
    writers: dict[int, pq.ParquetWriter] = {}
    rows_by_seed = {s: 0 for s in SEEDS}
    validated_k30_rows = 0
    template_root = a.subjects_dir / a.template
    try:
        for hemi_index, hemi in enumerate(("lh", "rh")):
            vertices, faces = fsio.read_geometry(template_root / "surf" / f"{hemi}.white")
            mask, labels = valid_template_mask(template_root, hemi)
            graph = build_graph(vertices, faces)
            hemi_nodes = nodes.loc[nodes.hemisphere.eq(hemi)].sort_values("node_order")
            anchors = hemi_nodes.fsaverage_vertex_id.to_numpy(np.int64)
            old = {
                s: pd.read_parquet(old_paths[s], filters=[("hemisphere", "==", hemi),
                    ("distance_bin_index", ">=", 10)])
                for s in SEEDS
            }
            for start in range(0, len(anchors), a.batch_size):
                batch = anchors[start:start + a.batch_size]
                distances = dijkstra(graph, directed=False, indices=batch, limit=30.0)
                seed_parts: dict[int, list[pd.DataFrame]] = {s: [] for s in SEEDS}
                for local, anchor in enumerate(batch):
                    node = hemi_nodes.iloc[start + local]
                    current = distances[local]
                    for b in PLATFORM_BINS:
                        lower, upper = DISTANCE_EDGES[b:b + 2]
                        candidates = np.flatnonzero(mask & (current > lower) & (current <= upper))
                        candidates = candidates[candidates != anchor]
                        expected_count = counts.loc[
                            counts.node_order.eq(node.node_order) & counts.distance_bin_index.eq(b),
                            "n_geometry",
                        ]
                        if len(expected_count) != 1 or int(expected_count.iat[0]) != len(candidates):
                            raise RuntimeError("Geometry candidate count differs from frozen pilot")
                        for seed in SEEDS:
                            selected = candidates[
                                rng_for(seed, hemi_index, int(anchor), b).permutation(len(candidates))[:300]
                            ]
                            previous = old[seed].loc[
                                old[seed].node_order.eq(node.node_order)
                                & old[seed].distance_bin_index.eq(b)
                            ].sort_values("sample_rank")
                            n_old = min(30, len(candidates))
                            if not np.array_equal(
                                selected[:n_old], previous.neighbor_vertex.to_numpy(np.int64)
                            ):
                                raise RuntimeError(
                                    f"Nested K30 validation failed: {hemi} node={node.node_order} bin={b} seed={seed}"
                                )
                            validated_k30_rows += n_old
                            if not len(selected):
                                continue
                            seed_parts[seed].append(pd.DataFrame({
                                "seed": np.int8(seed),
                                "sample_rank": np.arange(1, len(selected) + 1, dtype=np.int16),
                                "node_order": np.int32(node.node_order), "hemisphere": hemi,
                                "ico4_node_id": np.int32(node.ico_node_id),
                                "anchor_vertex": np.int32(anchor),
                                "neighbor_vertex": selected.astype(np.int32),
                                "geodesic_distance_mm": current[selected].astype(np.float32),
                                "distance_bin_index": np.int8(b),
                                "distance_bin": f"({lower:.0f},{upper:.0f}]",
                                "anchor_dk_label": np.int16(labels[anchor]),
                                "neighbor_dk_label": labels[selected].astype(np.int16),
                                "pair_class": np.where(labels[selected] == labels[anchor], "within", "across"),
                                "n_geometry": np.int32(len(candidates)),
                            }))
                for seed, parts in seed_parts.items():
                    if not parts:
                        continue
                    frame = pd.concat(parts, ignore_index=True)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if seed not in writers:
                        writers[seed] = pq.ParquetWriter(new_paths[seed], table.schema, compression="zstd")
                    writers[seed].write_table(table)
                    rows_by_seed[seed] += len(frame)
    finally:
        for writer in writers.values():
            writer.close()
    report = {
        "status": "complete", "base_seed": BASE_SEED, "seeds": list(SEEDS),
        "platform_bins": [f"({DISTANCE_EDGES[b]:.0f},{DISTANCE_EDGES[b+1]:.0f}]" for b in PLATFORM_BINS],
        "platform_k": [30, 100, 300], "nested": True, "replacement": False,
        "old_k30_rows_validated": validated_k30_rows,
        "pair_rows_by_seed": rows_by_seed,
        "source_local_pilot": str(source.resolve()),
    }
    (output / "pair_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"output_dir={output.resolve()}")


if __name__ == "__main__":
    main()
