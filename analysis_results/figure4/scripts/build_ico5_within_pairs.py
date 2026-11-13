#!/usr/bin/env python3
"""Build ico5-anchor/full-resolution within-DK geodesic K-pilot pairs.

Description
-----------
Build ico5-anchor/full-resolution within-DK geodesic K-pilot pairs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subjects-dir`` command-line input
- ``--fsaverage5-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/build_ico5_within_pairs.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from build_geodesic_pair_tables import (
    DISTANCE_EDGES,
    build_graph,
    valid_template_mask,
)


HEMISPHERES = (("lh", "left"), ("rh", "right"))
K_VALUES = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--fsaverage5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20261020)
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


def normalized(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - np.median(vertices, axis=0)
    return centered / np.linalg.norm(centered, axis=1, keepdims=True)


def ico5_mapping(
    template_root: Path, fsaverage5_dir: Path, hemisphere: str, side: str
) -> tuple[np.ndarray, np.ndarray, float]:
    high, _ = fsio.read_geometry(str(template_root / "surf" / f"{hemisphere}.sphere"))
    low = np.asarray(
        nib.load(fsaverage5_dir / f"sphere_{side}.gii.gz").darrays[0].data,
        dtype=float,
    )
    distance, mapped = cKDTree(normalized(high)).query(normalized(low), k=1)
    if len(low) != 10_242:
        raise ValueError(f"Expected 10,242 ico5 vertices, found {len(low)}")
    if np.unique(mapped).size != len(mapped):
        raise RuntimeError(f"{hemisphere}: ico5 mapping is not one-to-one")
    angular_error = np.degrees(2.0 * np.arcsin(np.clip(distance / 2.0, 0, 1)))
    return mapped.astype(np.int64), angular_error, float(angular_error.max())


def annotation_names(template_root: Path, hemisphere: str) -> tuple[np.ndarray, list[str]]:
    annotation, _, names = fsio.read_annot(
        str(template_root / "label" / f"{hemisphere}.aparc.annot")
    )
    decoded = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return annotation, decoded


def stable_rng(base_seed: int, seed_index: int, hemisphere_index: int,
               anchor: int, bin_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [base_seed, seed_index, hemisphere_index, int(anchor), bin_index]
    )
    return np.random.default_rng(sequence)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_root = args.subjects_dir / args.template
    mapping_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    pair_dir = args.output_dir / "ico5_within_nested_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair_writers: dict[int, pq.ParquetWriter] = {}
    n_pair_rows = 0
    mapping_qc: dict[str, object] = {}

    for hemisphere_index, (hemisphere, side) in enumerate(HEMISPHERES):
        white, faces = fsio.read_geometry(str(template_root / "surf" / f"{hemisphere}.white"))
        mask, labels = valid_template_mask(template_root, hemisphere)
        annotation, names = annotation_names(template_root, hemisphere)
        mapped, angular_error, maximum_error = ico5_mapping(
            template_root, args.fsaverage5_dir, hemisphere, side
        )
        anchors = mapped[mask[mapped]]
        ico_ids = np.flatnonzero(mask[mapped])
        graph = build_graph(white, faces)
        mapping_qc[hemisphere] = {
            "n_ico5_vertices": int(len(mapped)),
            "n_unique_full_resolution_vertices": int(np.unique(mapped).size),
            "n_masked_anchors": int(len(anchors)),
            "maximum_angular_mapping_error_degrees": maximum_error,
            "mapping_is_identity": bool(np.array_equal(mapped, np.arange(len(mapped)))),
        }
        for ico_id, full_id in zip(ico_ids, anchors):
            label = int(labels[full_id])
            mapping_rows.append({
                "hemisphere": hemisphere,
                "ico5_vertex_id": int(ico_id),
                "full_resolution_vertex_id": int(full_id),
                "dk_label": label,
                "dk_region": names[label],
                "angular_mapping_error_degrees": float(angular_error[ico_id]),
                "in_analysis_mask": True,
            })

        for batch_start in range(0, len(anchors), args.batch_size):
            batch = anchors[batch_start:batch_start + args.batch_size]
            batch_ico = ico_ids[batch_start:batch_start + args.batch_size]
            distances = dijkstra(graph, directed=False, indices=batch, limit=30.0)
            batch_pair_parts: list[pd.DataFrame] = []
            for local_index, (ico_id, anchor) in enumerate(zip(batch_ico, batch)):
                current = distances[local_index]
                region_label = int(labels[anchor])
                for bin_index, (lower, upper) in enumerate(
                    zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])
                ):
                    candidates = np.flatnonzero(
                        mask
                        & (labels == region_label)
                        & (current > lower)
                        & (current <= upper)
                    )
                    candidates = candidates[candidates != anchor]
                    count_rows.append({
                        "hemisphere": hemisphere,
                        "ico5_vertex_id": int(ico_id),
                        "anchor_vertex": int(anchor),
                        "anchor_dk_label": region_label,
                        "anchor_region": names[region_label],
                        "distance_bin_index": bin_index,
                        "distance_bin": f"{lower:.0f}-{upper:.0f}",
                        "n_candidates": int(len(candidates)),
                    })
                    if len(candidates) == 0:
                        continue
                    selected_rows = []
                    maximum = min(max(K_VALUES), len(candidates))
                    for seed_index in range(args.n_seeds):
                        rng = stable_rng(
                            args.base_seed, seed_index, hemisphere_index,
                            int(anchor), bin_index,
                        )
                        order = rng.permutation(len(candidates))[:maximum]
                        selected = candidates[order]
                        selected_rows.append(pd.DataFrame({
                            "seed": seed_index,
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
                    batch_pair_parts.extend(selected_rows)
            if batch_pair_parts:
                batch_pairs = pd.concat(batch_pair_parts, ignore_index=True)
                for seed_index, seed_pairs in batch_pairs.groupby("seed", sort=True):
                    table = pa.Table.from_pandas(seed_pairs, preserve_index=False)
                    seed_index = int(seed_index)
                    if seed_index not in pair_writers:
                        pair_writers[seed_index] = pq.ParquetWriter(
                            pair_dir / f"seed_{seed_index:02d}.parquet",
                            table.schema,
                            compression="zstd",
                        )
                    pair_writers[seed_index].write_table(table)
                n_pair_rows += len(batch_pairs)
            logging.info(
                "%s anchors %d/%d", hemisphere,
                min(batch_start + len(batch), len(anchors)), len(anchors),
            )

    for writer in pair_writers.values():
        writer.close()
    mapping = pd.DataFrame(mapping_rows)
    counts = pd.DataFrame(count_rows)
    mapping.to_csv(args.output_dir / "ico5_anchor_mapping.csv", index=False)
    counts.to_parquet(args.output_dir / "ico5_within_candidate_counts.parquet", index=False)

    summary_rows = []
    group_columns = ["anchor_region", "distance_bin_index", "distance_bin"]
    for keys, part in counts.groupby(group_columns, observed=True, sort=True):
        values = part["n_candidates"].to_numpy()
        current = dict(zip(group_columns, keys))
        current.update({
            "anchor_bin_total": len(values),
            "candidate_median": float(np.median(values)),
            "candidate_q25": float(np.quantile(values, .25)),
            "candidate_q75": float(np.quantile(values, .75)),
            "candidate_min": int(values.min()),
            "candidate_max": int(values.max()),
            "candidate_total": int(values.sum()),
        })
        for k in K_VALUES:
            current[f"fraction_at_least_{k}"] = float(np.mean(values >= k))
            current[f"pairs_used_k{k}"] = int(np.minimum(values, k).sum())
            current[f"anchors_retained_k{k}"] = int(np.sum(values > 0))
        summary_rows.append(current)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "ico5_within_candidate_summary.csv", index=False)
    (args.output_dir / "ico5_pair_build.json").write_text(json.dumps({
        "status": "complete",
        "surface": "full-resolution fsaverage white mesh",
        "anchor_source": "official Nilearn/FreeSurfer fsaverage5 sphere",
        "distance_bins_mm": [float(x) for x in DISTANCE_EDGES],
        "same_dk_region_only": True,
        "directed_pairs": True,
        "k_values": list(K_VALUES),
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "mapping_qc": mapping_qc,
        "n_anchor_bin_rows": int(len(counts)),
        "n_nested_pair_rows": int(n_pair_rows),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
