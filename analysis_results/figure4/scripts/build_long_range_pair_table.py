#!/usr/bin/env python3
"""Build a sparse fsaverage white-surface pair table from 20 to 100 mm."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from build_geodesic_pair_tables import (
    build_graph,
    sample_candidates,
    select_uniform_anchors,
    valid_template_mask,
)


HEMISPHERES = ("lh", "rh")
DISTANCE_EDGES_MM = np.asarray([20.0, 30.0, 45.0, 60.0, 80.0, 100.0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors-per-hemisphere", type=int, default=256)
    parser.add_argument("--neighbors-per-anchor-bin", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_root = args.subjects_dir / args.template
    tables = []
    inventory = []
    for hemisphere_index, hemisphere in enumerate(HEMISPHERES):
        white, faces = fsio.read_geometry(
            str(template_root / "surf" / f"{hemisphere}.white")
        )
        sphere, _ = fsio.read_geometry(
            str(template_root / "surf" / f"{hemisphere}.sphere")
        )
        mask, labels = valid_template_mask(template_root, hemisphere)
        anchors = select_uniform_anchors(
            sphere, mask, args.anchors_per_hemisphere
        )
        graph = build_graph(white, faces)
        rng = np.random.default_rng(args.seed + hemisphere_index)
        rows = []
        for start in range(0, len(anchors), args.batch_size):
            batch = anchors[start : start + args.batch_size]
            distances = dijkstra(
                graph, directed=False, indices=batch,
                limit=float(DISTANCE_EDGES_MM[-1]),
            )
            logging.info(
                "%s anchors %d/%d", hemisphere,
                min(start + len(batch), len(anchors)), len(anchors),
            )
            for local_index, anchor in enumerate(batch):
                anchor_distances = distances[local_index]
                for bin_index, (lower, upper) in enumerate(
                    zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:])
                ):
                    candidates = np.flatnonzero(
                        mask
                        & (anchor_distances > lower)
                        & (anchor_distances <= upper)
                    )
                    selected = sample_candidates(
                        candidates, args.neighbors_per_anchor_bin, rng
                    )
                    for neighbor in selected:
                        rows.append({
                            "hemisphere": hemisphere,
                            "anchor_vertex": int(anchor),
                            "neighbor_vertex": int(neighbor),
                            "geodesic_distance_mm": float(
                                anchor_distances[neighbor]
                            ),
                            "distance_bin_index": bin_index,
                            "distance_bin": f"{lower:.0f}-{upper:.0f}",
                            "anchor_dk_label": int(labels[anchor]),
                            "neighbor_dk_label": int(labels[neighbor]),
                            "pair_type": "all",
                            "split_half": 1 + (int(anchor) % 2),
                        })
        table = pd.DataFrame(rows)
        tables.append(table)
        inventory.append({
            "hemisphere": hemisphere,
            "analysis_mask_vertices": int(mask.sum()),
            "anchors": int(table.anchor_vertex.nunique()),
            "pairs": len(table),
        })
    pairs = pd.concat(tables, ignore_index=True)
    pairs.to_parquet(
        args.output_dir / "figure4_long_range_geodesic_pairs.parquet",
        index=False,
    )
    pd.DataFrame(inventory).to_csv(
        args.output_dir / "figure4_long_range_pair_inventory.csv", index=False
    )
    counts = (
        pairs.groupby(["hemisphere", "distance_bin"], observed=True)
        .agg(
            anchors=("anchor_vertex", "nunique"),
            pairs=("neighbor_vertex", "size"),
            median_distance_mm=("geodesic_distance_mm", "median"),
        )
        .reset_index()
    )
    counts.to_csv(
        args.output_dir / "figure4_long_range_pair_counts.csv", index=False
    )
    (args.output_dir / "completion.json").write_text(
        json.dumps({
            "status": "complete",
            "surface": "fsaverage white",
            "distance": "mesh-edge geodesic distance in millimeters",
            "distance_edges_mm": DISTANCE_EDGES_MM.tolist(),
            "anchors_per_hemisphere": args.anchors_per_hemisphere,
            "neighbors_per_anchor_bin": args.neighbors_per_anchor_bin,
            "seed": args.seed,
            "n_pairs": len(pairs),
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
