#!/usr/bin/env python3
"""Build fixed nested K=15/30 full-cortex pair tables for ico4 anchors."""

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
SCRIPT_DIR = REPO / "analysis_results/figure4/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from build_geodesic_pair_tables import DISTANCE_EDGES, build_graph, valid_template_mask  # noqa: E402


HEMISPHERES = ("lh", "rh")
SEEDS = (0, 1, 2, 3, 4)
BASE_SEED = 20261119


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects-dir", type=Path, default=Path(
        "/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"))
    p.add_argument("--template", default="fsaverage")
    p.add_argument("--ico4-node-table", type=Path, default=Path(
        "/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/template/ico4_vertex_order.csv"))
    p.add_argument("--output-dir", type=Path, default=REPO /
        "analysis_results/figure6/确定扰动因素/distance/local_similarity_stability")
    p.add_argument("--batch-size", type=int, default=24)
    return p.parse_args()


def rng_for(seed: int, hemi_index: int, anchor: int, bin_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [BASE_SEED, seed, hemi_index, int(anchor), bin_index]
    ))


def main() -> None:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = a.output_dir / "pairs"
    pair_dir.mkdir(exist_ok=True)
    nodes = pd.read_csv(a.ico4_node_table)
    nodes = nodes.loc[nodes.in_analysis_mask.astype(bool)].copy()
    nodes = nodes.sort_values(["hemisphere", "ico_node_id"], kind="stable").reset_index(drop=True)
    if len(nodes) != 4687 or not nodes.ico_level.eq(4).all():
        raise RuntimeError("Expected the frozen 4,687-node masked ico4 table")
    nodes.insert(0, "node_order", np.arange(len(nodes), dtype=np.int32))
    nodes.to_csv(a.output_dir / "ico4_nodes.csv", index=False)

    paths = {s: pair_dir / f"seed_{s:02d}_k30_nested.parquet" for s in SEEDS}
    writers: dict[int, pq.ParquetWriter] = {}
    count_rows: list[dict[str, object]] = []
    total_rows = {s: 0 for s in SEEDS}
    template_root = a.subjects_dir / a.template
    try:
        for hemi_index, hemi in enumerate(HEMISPHERES):
            vertices, faces = fsio.read_geometry(template_root / "surf" / f"{hemi}.white")
            mask, labels = valid_template_mask(template_root, hemi)
            graph = build_graph(vertices, faces)
            hemi_nodes = nodes.loc[nodes.hemisphere.eq(hemi)].reset_index(drop=True)
            anchors = hemi_nodes.fsaverage_vertex_id.to_numpy(np.int64)
            if not mask[anchors].all():
                raise RuntimeError(f"{hemi}: anchor outside template analysis mask")
            if not np.array_equal(labels[anchors], hemi_nodes.dk_label_id.to_numpy(np.int64)):
                raise RuntimeError(f"{hemi}: node table DK labels disagree with template")
            for start in range(0, len(anchors), a.batch_size):
                batch = anchors[start:start + a.batch_size]
                distances = dijkstra(graph, directed=False, indices=batch, limit=30.0)
                seed_parts: dict[int, list[pd.DataFrame]] = {s: [] for s in SEEDS}
                for local, anchor in enumerate(batch):
                    node = hemi_nodes.iloc[start + local]
                    current = distances[local]
                    for b, (lower, upper) in enumerate(zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])):
                        candidates = np.flatnonzero(mask & (current > lower) & (current <= upper))
                        candidates = candidates[candidates != anchor]
                        count_rows.append({
                            "node_order": int(node.node_order), "hemisphere": hemi,
                            "ico4_node_id": int(node.ico_node_id), "anchor_vertex": int(anchor),
                            "anchor_dk_label": int(labels[anchor]),
                            "anchor_dk_region": str(node.dk_label_name),
                            "distance_bin_index": b, "distance_bin": f"({lower:.0f},{upper:.0f}]",
                            "n_geometry": int(len(candidates)),
                            "n_template_k15": int(min(15, len(candidates))),
                            "n_template_k30": int(min(30, len(candidates))),
                        })
                        for seed in SEEDS:
                            selected = candidates[rng_for(seed, hemi_index, int(anchor), b).permutation(len(candidates))[:30]]
                            if not len(selected):
                                continue
                            seed_parts[seed].append(pd.DataFrame({
                                "seed": np.int8(seed),
                                "sample_rank": np.arange(1, len(selected) + 1, dtype=np.int8),
                                "node_order": np.int32(node.node_order),
                                "hemisphere": hemi,
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
                        writers[seed] = pq.ParquetWriter(paths[seed], table.schema, compression="zstd")
                    writers[seed].write_table(table)
                    total_rows[seed] += len(frame)
    finally:
        for writer in writers.values():
            writer.close()

    counts = pd.DataFrame(count_rows).sort_values(["node_order", "distance_bin_index"])
    if len(counts) != 4687 * 15 or counts.duplicated(["node_order", "distance_bin_index"]).any():
        raise RuntimeError("Anchor-bin inventory is incomplete or duplicated")
    counts.to_parquet(a.output_dir / "anchor_bin_template_counts.parquet", index=False, compression="zstd")
    report = {
        "status": "complete", "base_seed": BASE_SEED, "seeds": list(SEEDS),
        "k_values": [15, 30], "nested": True, "replacement": False,
        "surface": "full-resolution fsaverage white", "neighbor_scope": "same hemisphere full-resolution analysis mask, all DK labels",
        "distance_bins": [f"({x:.0f},{y:.0f}]" for x, y in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])],
        "n_nodes": len(nodes), "n_anchor_bins": len(counts),
        "pair_rows_by_seed": total_rows,
    }
    (a.output_dir / "pair_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"output_dir={a.output_dir.resolve()}")


if __name__ == "__main__":
    main()
