#!/usr/bin/env python3
"""Build fixed, reusable fsaverage white-surface geodesic pair tables.

Description
-----------
Build fixed, reusable fsaverage white-surface geodesic pair tables.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subjects-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/build_geodesic_pair_tables.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


HEMISPHERES = ("lh", "rh")
DISTANCE_EDGES = np.arange(0.0, 32.0, 2.0)
EXCLUDED_LABELS = {"unknown", "corpuscallosum", "medial_wall", "medialwall"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors-per-hemisphere", type=int, default=2048)
    parser.add_argument("--max-neighbors-per-anchor-bin", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fibonacci_directions(n_points: int) -> np.ndarray:
    indices = np.arange(n_points, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (indices + 0.5) / n_points
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden_angle * indices
    return np.column_stack((np.cos(theta) * radius, y, np.sin(theta) * radius))


def select_uniform_anchors(
    sphere_vertices: np.ndarray,
    mask: np.ndarray,
    n_anchors: int,
) -> np.ndarray:
    """Choose a deterministic farthest-point sample on the masked sphere.

    This maximises the distance to the nearest earlier anchor and therefore
    directly guarantees surface coverage.  It avoids depending on vertex-ID
    ordering or on truncating a Fibonacci candidate sequence.
    """
    directions = sphere_vertices - np.median(sphere_vertices, axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    valid_ids = np.flatnonzero(mask)
    if n_anchors > len(valid_ids):
        raise ValueError("Requested more anchors than valid mask vertices")
    valid_directions = directions[valid_ids]
    start_direction = fibonacci_directions(1)[0]
    selected = np.empty(n_anchors, dtype=int)
    selected[0] = np.argmin(np.sum((valid_directions - start_direction) ** 2, axis=1))
    nearest_squared_distance = np.sum(
        (valid_directions - valid_directions[selected[0]]) ** 2, axis=1
    )
    for index in range(1, n_anchors):
        selected[index] = np.argmax(nearest_squared_distance)
        squared_distance = np.sum(
            (valid_directions - valid_directions[selected[index]]) ** 2, axis=1
        )
        nearest_squared_distance = np.minimum(nearest_squared_distance, squared_distance)
    return valid_ids[selected]


def build_graph(vertices: np.ndarray, faces: np.ndarray):
    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((weights, weights))
    return coo_matrix(
        (values, (rows, cols)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()


def valid_template_mask(
    template_root: Path, hemisphere: str
) -> tuple[np.ndarray, np.ndarray]:
    annotation, _, names = fsio.read_annot(
        str(template_root / "label" / f"{hemisphere}.aparc.annot")
    )
    cortex_ids = read_label(
        str(template_root / "label" / f"{hemisphere}.cortex.label"),
        read_scalars=False,
    ).astype(int)
    mask = np.zeros(len(annotation), dtype=bool)
    mask[cortex_ids] = True
    normalized_names = np.asarray(
        [
            (name.decode() if isinstance(name, bytes) else str(name))
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            for name in names
        ]
    )
    valid_annotation = annotation >= 0
    for label_id, name in enumerate(normalized_names):
        if name in EXCLUDED_LABELS:
            valid_annotation &= annotation != label_id
    return mask & valid_annotation, annotation


def sample_candidates(
    candidates: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(candidates) <= maximum:
        return candidates
    return np.sort(rng.choice(candidates, size=maximum, replace=False))


def pair_rows_for_hemisphere(
    template_root: Path,
    hemisphere: str,
    n_anchors: int,
    maximum: int,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    white, faces = fsio.read_geometry(
        str(template_root / "surf" / f"{hemisphere}.white")
    )
    sphere, _ = fsio.read_geometry(str(template_root / "surf" / f"{hemisphere}.sphere"))
    mask, labels = valid_template_mask(template_root, hemisphere)
    anchors = select_uniform_anchors(sphere, mask, n_anchors)
    graph = build_graph(white, faces)
    rng = np.random.default_rng(seed + (0 if hemisphere == "lh" else 1))
    all_rows: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []

    for batch_start in range(0, len(anchors), batch_size):
        batch = anchors[batch_start : batch_start + batch_size]
        distances = dijkstra(
            graph,
            directed=False,
            indices=batch,
            limit=30.0,
        )
        logging.info(
            "%s anchors %d/%d",
            hemisphere,
            min(batch_start + len(batch), len(anchors)),
            len(anchors),
        )
        for local_index, anchor in enumerate(batch):
            anchor_distances = distances[local_index]
            for bin_index in range(len(DISTANCE_EDGES) - 1):
                lower = DISTANCE_EDGES[bin_index]
                upper = DISTANCE_EDGES[bin_index + 1]
                candidates = np.flatnonzero(
                    mask & (anchor_distances > lower) & (anchor_distances <= upper)
                )
                candidates = candidates[candidates != anchor]
                selected = sample_candidates(candidates, maximum, rng)
                for neighbor in selected:
                    pair_type = (
                        "within_region"
                        if labels[anchor] == labels[neighbor]
                        else "across_boundary"
                    )
                    all_rows.append(
                        {
                            "hemisphere": hemisphere,
                            "anchor_vertex": int(anchor),
                            "neighbor_vertex": int(neighbor),
                            "geodesic_distance_mm": float(anchor_distances[neighbor]),
                            "distance_bin_index": bin_index,
                            "distance_bin": f"{lower:.0f}-{upper:.0f}",
                            "anchor_dk_label": int(labels[anchor]),
                            "neighbor_dk_label": int(labels[neighbor]),
                            "pair_type": pair_type,
                            "split_half": 1 + (int(anchor) + bin_index) % 2,
                        }
                    )

                subbin_edges = np.arange(lower, upper + 0.5, 0.5)
                maximum_per_subbin = max(1, int(np.ceil(maximum / 4)))
                for subbin_index in range(len(subbin_edges) - 1):
                    sub_lower = subbin_edges[subbin_index]
                    sub_upper = subbin_edges[subbin_index + 1]
                    in_subbin = candidates[
                        (anchor_distances[candidates] > sub_lower)
                        & (anchor_distances[candidates] <= sub_upper)
                    ]
                    within = in_subbin[labels[in_subbin] == labels[anchor]]
                    across = in_subbin[labels[in_subbin] != labels[anchor]]
                    matched_count = min(
                        maximum_per_subbin,
                        len(within),
                        len(across),
                    )
                    if matched_count == 0:
                        continue
                    for pair_type, pool in (
                        ("within_region", within),
                        ("across_boundary", across),
                    ):
                        stratified = sample_candidates(pool, matched_count, rng)
                        for neighbor in stratified:
                            boundary_rows.append(
                                {
                                    "hemisphere": hemisphere,
                                    "anchor_vertex": int(anchor),
                                    "neighbor_vertex": int(neighbor),
                                    "geodesic_distance_mm": float(
                                        anchor_distances[neighbor]
                                    ),
                                    "distance_bin_index": bin_index,
                                    "distance_bin": f"{lower:.0f}-{upper:.0f}",
                                    "distance_subbin": (
                                        f"{sub_lower:.1f}-{sub_upper:.1f}"
                                    ),
                                    "anchor_dk_label": int(labels[anchor]),
                                    "neighbor_dk_label": int(labels[neighbor]),
                                    "pair_type": pair_type,
                                    "split_half": (1 + (int(anchor) + bin_index) % 2),
                                }
                            )

    information = {
        "hemisphere": hemisphere,
        "n_template_vertices": len(white),
        "n_analysis_mask_vertices": int(mask.sum()),
        "n_anchors": len(anchors),
        "n_all_pairs": len(all_rows),
        "n_boundary_stratified_pairs": len(boundary_rows),
    }
    return pd.DataFrame(all_rows), pd.DataFrame(boundary_rows), information


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "figure4_geodesic_pairs_all.parquet"
    boundary_path = args.output_dir / "figure4_geodesic_pairs_boundary.parquet"
    completion_path = args.output_dir / "pair_table_completion.json"
    if (
        args.resume
        and all_path.is_file()
        and boundary_path.is_file()
        and completion_path.is_file()
    ):
        print(f"Pair tables already complete: {args.output_dir}")
        return

    template_root = args.subjects_dir / args.template
    all_tables = []
    boundary_tables = []
    information = []
    for hemisphere in HEMISPHERES:
        all_pairs, boundary_pairs, current = pair_rows_for_hemisphere(
            template_root,
            hemisphere,
            args.anchors_per_hemisphere,
            args.max_neighbors_per_anchor_bin,
            args.batch_size,
            args.seed,
        )
        all_tables.append(all_pairs)
        boundary_tables.append(boundary_pairs)
        information.append(current)

    all_table = pd.concat(all_tables, ignore_index=True)
    boundary_table = pd.concat(boundary_tables, ignore_index=True)
    all_table.to_parquet(all_path, index=False)
    boundary_table.to_parquet(boundary_path, index=False)
    estimate = pd.DataFrame(information)
    estimate.to_csv(args.output_dir / "figure4_pair_table_inventory.csv", index=False)
    pair_counts = (
        boundary_table.groupby(["hemisphere", "distance_bin", "pair_type"])
        .size()
        .rename("n_pairs")
        .reset_index()
    )
    pair_counts.to_csv(args.output_dir / "figure4_pair_counts.csv", index=False)
    completion_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "surface": "fsaverage white",
                "distance": "mesh-edge geodesic distance in millimeters",
                "anchors_per_hemisphere": args.anchors_per_hemisphere,
                "maximum_neighbors_per_anchor_bin": (args.max_neighbors_per_anchor_bin),
                "seed": args.seed,
                "n_all_pairs": len(all_table),
                "n_boundary_pairs": len(boundary_table),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
