#!/usr/bin/env python3
"""Extend local 0--10 mm pair samples to nested K=300, validating old K=30."""

from __future__ import annotations

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
DISTANCE = REPO / "analysis_results/figure6/确定扰动因素/distance"
sys.path.insert(0, str(REPO / "analysis_results/figure4/scripts"))
from build_geodesic_pair_tables import DISTANCE_EDGES, build_graph, valid_template_mask  # noqa: E402


BASE_SEED = 20261119
SEEDS = range(3)


def rng_for(seed: int, hemi_index: int, anchor: int, bin_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [BASE_SEED, seed, hemi_index, int(anchor), bin_index]))


def main() -> None:
    source = DISTANCE / "local_similarity_stability"
    pilot = DISTANCE / "null_adjusted_E_stability/local_K_convergence"
    pair_dir = pilot / "pairs"; pair_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(source / "ico4_nodes.csv").sort_values("node_order")
    counts = pd.read_parquet(source / "anchor_bin_template_counts.parquet")
    template_root = Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/fsaverage")
    writers = {}; rows_by_seed = {s: 0 for s in SEEDS}; validated = 0
    try:
        for hemi_index, hemi in enumerate(("lh", "rh")):
            vertices, faces = fsio.read_geometry(template_root / "surf" / f"{hemi}.white")
            mask, labels = valid_template_mask(template_root, hemi)
            graph = build_graph(vertices, faces)
            hemi_nodes = nodes.loc[nodes.hemisphere.eq(hemi)].reset_index(drop=True)
            anchors = hemi_nodes.fsaverage_vertex_id.to_numpy(np.int64)
            old = {s: pd.read_parquet(source / "pairs" / f"seed_{s:02d}_k30_nested.parquet",
                                     filters=[("hemisphere", "==", hemi),
                                              ("distance_bin_index", "<", 5)]) for s in SEEDS}
            for start in range(0, len(anchors), 24):
                batch = anchors[start:start+24]
                distances = dijkstra(graph, directed=False, indices=batch, limit=10.0)
                parts = {s: [] for s in SEEDS}
                for local, anchor in enumerate(batch):
                    node = hemi_nodes.iloc[start+local]; current = distances[local]
                    for b in range(5):
                        lower, upper = DISTANCE_EDGES[b:b+2]
                        candidates = np.flatnonzero(mask & (current > lower) & (current <= upper))
                        candidates = candidates[candidates != anchor]
                        expected = counts.loc[counts.node_order.eq(node.node_order)
                            & counts.distance_bin_index.eq(b), "n_geometry"]
                        if len(expected) != 1 or int(expected.iat[0]) != len(candidates):
                            raise RuntimeError("Frozen geometry candidate count mismatch")
                        for seed in SEEDS:
                            selected = candidates[rng_for(seed, hemi_index, int(anchor), b)
                                                  .permutation(len(candidates))[:300]]
                            previous = old[seed].loc[old[seed].node_order.eq(node.node_order)
                                & old[seed].distance_bin_index.eq(b)].sort_values("sample_rank")
                            n_old = min(30, len(candidates))
                            if not np.array_equal(selected[:n_old], previous.neighbor_vertex.to_numpy(np.int64)):
                                raise RuntimeError(f"K30 mismatch node={node.node_order} bin={b} seed={seed}")
                            validated += n_old
                            if not len(selected): continue
                            parts[seed].append(pd.DataFrame({
                                "seed": np.int8(seed), "sample_rank": np.arange(1,len(selected)+1,dtype=np.int16),
                                "node_order": np.int32(node.node_order), "hemisphere": hemi,
                                "ico4_node_id": np.int32(node.ico_node_id), "anchor_vertex": np.int32(anchor),
                                "neighbor_vertex": selected.astype(np.int32),
                                "geodesic_distance_mm": current[selected].astype(np.float32),
                                "distance_bin_index": np.int8(b), "distance_bin": f"({lower:.0f},{upper:.0f}]",
                                "anchor_dk_label": np.int16(labels[anchor]),
                                "neighbor_dk_label": labels[selected].astype(np.int16),
                                "pair_class": np.where(labels[selected]==labels[anchor],"within","across"),
                                "n_geometry": np.int32(len(candidates)),
                            }))
                for seed, tables in parts.items():
                    if not tables: continue
                    frame = pd.concat(tables, ignore_index=True)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    path = pair_dir / f"seed_{seed:02d}_local_k300_nested.parquet"
                    if seed not in writers:
                        writers[seed] = pq.ParquetWriter(path, table.schema, compression="zstd")
                    writers[seed].write_table(table); rows_by_seed[seed] += len(frame)
    finally:
        for writer in writers.values(): writer.close()
    report = {"status":"complete","base_seed":BASE_SEED,"seeds":list(SEEDS),
              "local_bins":["(0,2]","(2,4]","(4,6]","(6,8]","(8,10]"],
              "local_k":[30,100,300],"nested":True,"replacement":False,
              "old_k30_rows_validated":validated,"pair_rows_by_seed":rows_by_seed}
    (pilot/"pair_build.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2)); print(f"output_dir={pilot.resolve()}")


if __name__ == "__main__": main()
