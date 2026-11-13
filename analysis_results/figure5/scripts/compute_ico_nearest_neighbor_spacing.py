#!/usr/bin/env python3
"""Measure masked ico3/4/5 nearest-neighbor spacing on fsaverage white.

Description
-----------
Measure masked ico3/4/5 nearest-neighbor spacing on fsaverage white.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--vertex-order-dir`` command-line input
- ``--subjects-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/compute_ico_nearest_neighbor_spacing.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE4_SCRIPT_DIR = REPO_ROOT / "analysis_results" / "figure4" / "scripts"
sys.path.insert(0, str(FIGURE4_SCRIPT_DIR))
from build_geodesic_pair_tables import build_graph, valid_template_mask  # noqa: E402


LEVELS = (3, 4, 5)
HEMISPHERES = ("lh", "rh")
SUMMARY_COLUMNS = (
    "level",
    "hemisphere",
    "n_nodes",
    "mean_nn_mm",
    "sd_nn_mm",
    "median_nn_mm",
    "q1_nn_mm",
    "q3_nn_mm",
    "p5_nn_mm",
    "p95_nn_mm",
    "min_nn_mm",
    "max_nn_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex-order-dir", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--distance-limit-mm", type=float, default=20.0)
    parser.add_argument("--figure4-pair-cache", type=Path)
    parser.add_argument("--validation-pairs-per-hemisphere", type=int, default=32)
    parser.add_argument("--validation-tolerance-mm", type=float, default=1e-9)
    return parser.parse_args()


def load_node_tables(
    vertex_order_dir: Path, template_root: Path
) -> dict[tuple[int, str], pd.DataFrame]:
    tables: dict[tuple[int, str], pd.DataFrame] = {}
    template_masks = {
        hemisphere: valid_template_mask(template_root, hemisphere)[0]
        for hemisphere in HEMISPHERES
    }
    required = {
        "hemisphere",
        "ico_level",
        "ico_node_id",
        "fsaverage_vertex_id",
        "in_analysis_mask",
    }
    for level in LEVELS:
        path = vertex_order_dir / f"ico{level}_vertex_order.csv"
        table = pd.read_csv(path)
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if not table["ico_level"].eq(level).all():
            raise ValueError(f"{path} contains an unexpected ico level")
        if not table["hemisphere"].isin(HEMISPHERES).all():
            raise ValueError(f"{path} contains an unexpected hemisphere")
        for hemisphere in HEMISPHERES:
            hemi = table.loc[table["hemisphere"].eq(hemisphere)].copy()
            hemi = hemi.sort_values("ico_node_id", kind="stable").reset_index(drop=True)
            if hemi["ico_node_id"].duplicated().any():
                raise ValueError(f"ico{level} {hemisphere} has duplicate ico node IDs")
            vertex_ids = hemi["fsaverage_vertex_id"].to_numpy(dtype=np.int64)
            expected_mask = template_masks[hemisphere][vertex_ids]
            recorded_mask = hemi["in_analysis_mask"].astype(bool).to_numpy()
            if not np.array_equal(recorded_mask, expected_mask):
                count = int(np.count_nonzero(recorded_mask != expected_mask))
                raise RuntimeError(
                    f"ico{level} {hemisphere}: Figure 5 mask differs from Figure 4 "
                    f"cortex/valid-DK mask at {count} nodes"
                )
            retained = hemi.loc[recorded_mask].reset_index(drop=True)
            if len(retained) < 2:
                raise RuntimeError(f"ico{level} {hemisphere} has fewer than two retained nodes")
            tables[level, hemisphere] = retained
    return tables


def nearest_neighbors(
    graph,
    table: pd.DataFrame,
    batch_size: int,
    distance_limit_mm: float,
) -> pd.DataFrame:
    retained_ids = table["fsaverage_vertex_id"].to_numpy(dtype=np.int64)
    n_nodes = len(retained_ids)
    nearest_distance = np.full(n_nodes, np.inf, dtype=np.float64)
    nearest_position = np.full(n_nodes, -1, dtype=np.int64)
    for start in range(0, n_nodes, batch_size):
        stop = min(start + batch_size, n_nodes)
        distances = dijkstra(
            graph,
            directed=False,
            indices=retained_ids[start:stop],
            limit=distance_limit_mm,
        )
        candidate_distances = distances[:, retained_ids]
        local_rows = np.arange(stop - start)
        candidate_distances[local_rows, np.arange(start, stop)] = np.inf
        local_nearest = np.argmin(candidate_distances, axis=1)
        local_distance = candidate_distances[local_rows, local_nearest]
        nearest_position[start:stop] = local_nearest
        nearest_distance[start:stop] = local_distance
    if not np.isfinite(nearest_distance).all():
        missing = int(np.count_nonzero(~np.isfinite(nearest_distance)))
        raise RuntimeError(
            f"{missing} nodes had no retained neighbor within {distance_limit_mm:g} mm; "
            "increase --distance-limit-mm"
        )
    result = table.copy()
    result = result.rename(
        columns={
            "ico_node_id": "node_ico_id",
            "fsaverage_vertex_id": "node_fsaverage_vertex_id",
        }
    )
    result["nearest_neighbor_ico_id"] = table.iloc[nearest_position][
        "ico_node_id"
    ].to_numpy(dtype=np.int64)
    result["nearest_neighbor_fsaverage_vertex_id"] = table.iloc[nearest_position][
        "fsaverage_vertex_id"
    ].to_numpy(dtype=np.int64)
    result["nearest_neighbor_geodesic_distance_mm"] = nearest_distance
    preferred = [
        "ico_level",
        "hemisphere",
        "node_ico_id",
        "node_fsaverage_vertex_id",
        "nearest_neighbor_ico_id",
        "nearest_neighbor_fsaverage_vertex_id",
        "nearest_neighbor_geodesic_distance_mm",
    ]
    return result.loc[:, preferred]


def summarize(level: int, hemisphere: str, values: np.ndarray) -> dict[str, object]:
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "level": f"ico{level}",
        "hemisphere": hemisphere,
        "n_nodes": len(values),
        "mean_nn_mm": float(np.mean(values)),
        "sd_nn_mm": float(np.std(values, ddof=1)),
        "median_nn_mm": float(quantiles[2]),
        "q1_nn_mm": float(quantiles[1]),
        "q3_nn_mm": float(quantiles[3]),
        "p5_nn_mm": float(quantiles[0]),
        "p95_nn_mm": float(quantiles[4]),
        "min_nn_mm": float(np.min(values)),
        "max_nn_mm": float(np.max(values)),
    }


def validate_against_figure4_cache(
    cache_path: Path,
    graphs: dict[str, object],
    pairs_per_hemisphere: int,
    tolerance_mm: float,
) -> dict[str, object]:
    columns = [
        "hemisphere",
        "anchor_vertex",
        "neighbor_vertex",
        "geodesic_distance_mm",
    ]
    cache = pd.read_parquet(cache_path, columns=columns)
    differences: list[float] = []
    checked_rows: list[dict[str, object]] = []
    for hemisphere in HEMISPHERES:
        hemi = cache.loc[cache["hemisphere"].eq(hemisphere)].reset_index(drop=True)
        positions = np.linspace(
            0, len(hemi) - 1, min(pairs_per_hemisphere, len(hemi)), dtype=int
        )
        selected = hemi.iloc[positions].reset_index(drop=True)
        anchors = selected["anchor_vertex"].to_numpy(dtype=np.int64)
        neighbors = selected["neighbor_vertex"].to_numpy(dtype=np.int64)
        recomputed = dijkstra(
            graphs[hemisphere], directed=False, indices=anchors
        )[np.arange(len(selected)), neighbors]
        cached = selected["geodesic_distance_mm"].to_numpy(dtype=np.float64)
        absolute = np.abs(recomputed - cached)
        differences.extend(absolute.tolist())
        for row_index, difference in enumerate(absolute):
            checked_rows.append(
                {
                    "hemisphere": hemisphere,
                    "anchor_vertex": int(anchors[row_index]),
                    "neighbor_vertex": int(neighbors[row_index]),
                    "cached_distance_mm": float(cached[row_index]),
                    "recomputed_distance_mm": float(recomputed[row_index]),
                    "absolute_discrepancy_mm": float(difference),
                }
            )
    maximum = float(np.max(differences))
    result = {
        "status": "passed" if maximum <= tolerance_mm else "failed",
        "comparison": "recomputed distances versus Figure 4 cached pair distances",
        "figure4_pair_cache": str(cache_path.resolve()),
        "n_pairs_checked": len(differences),
        "maximum_absolute_discrepancy_mm": maximum,
        "mean_absolute_discrepancy_mm": float(np.mean(differences)),
        "tolerance_mm": tolerance_mm,
        "checked_pairs": checked_rows,
    }
    if result["status"] != "passed":
        raise RuntimeError(
            f"Figure 4 distance validation failed: {maximum:.12g} mm > "
            f"{tolerance_mm:.12g} mm"
        )
    return result


def plot_distributions(details: pd.DataFrame, output_path: Path) -> None:
    colors = {"lh": "#4B74B2", "rh": "#FC8C5A"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for axis, level in zip(axes, LEVELS):
        level_table = details.loc[details["ico_level"].eq(level)]
        values = [
            level_table.loc[level_table["hemisphere"].eq(hemisphere),
                            "nearest_neighbor_geodesic_distance_mm"].to_numpy()
            for hemisphere in HEMISPHERES
        ]
        violins = axis.violinplot(
            values, positions=[1, 2], widths=0.72, showextrema=False
        )
        for body, hemisphere in zip(violins["bodies"], HEMISPHERES):
            body.set_facecolor(colors[hemisphere])
            body.set_edgecolor("white")
            body.set_alpha(0.78)
        for position, current in zip([1, 2], values):
            q1, median, q3 = np.quantile(current, [0.25, 0.5, 0.75])
            axis.vlines(position, q1, q3, color="black", linewidth=3.0, zorder=3)
            axis.scatter(position, median, color="white", edgecolor="black", s=25, zorder=4)
        axis.set_title(f"ico{level}")
        axis.set_xticks([1, 2], ["LH", "RH"])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Nearest-neighbor white-surface\ngeodesic distance (mm)")
    fig.suptitle("Masked Figure 5 node spacing on fsaverage", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.distance_limit_mm <= 0:
        raise ValueError("Batch size and distance limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_root = args.subjects_dir / args.template
    tables = load_node_tables(args.vertex_order_dir, template_root)

    graphs: dict[str, object] = {}
    detail_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for hemisphere in HEMISPHERES:
        vertices, faces = fsio.read_geometry(
            str(template_root / "surf" / f"{hemisphere}.white")
        )
        graphs[hemisphere] = build_graph(vertices, faces)
        for level in LEVELS:
            detail = nearest_neighbors(
                graphs[hemisphere],
                tables[level, hemisphere],
                args.batch_size,
                args.distance_limit_mm,
            )
            detail_tables.append(detail)
            values = detail["nearest_neighbor_geodesic_distance_mm"].to_numpy()
            summary_rows.append(summarize(level, hemisphere, values))

    details = pd.concat(detail_tables, ignore_index=True).sort_values(
        ["ico_level", "hemisphere", "node_ico_id"], kind="stable"
    )
    for level in LEVELS:
        pooled = details.loc[
            details["ico_level"].eq(level),
            "nearest_neighbor_geodesic_distance_mm",
        ].to_numpy()
        summary_rows.append(summarize(level, "bilateral_pooled", pooled))
    summary = pd.DataFrame(summary_rows).loc[:, SUMMARY_COLUMNS]
    summary["_hemisphere_order"] = summary["hemisphere"].map(
        {"lh": 0, "rh": 1, "bilateral_pooled": 2}
    )
    summary = summary.sort_values(
        ["level", "_hemisphere_order"], kind="stable"
    ).drop(columns="_hemisphere_order")

    detail_path = args.output_dir / "ico_nearest_neighbor_distances.csv"
    summary_path = args.output_dir / "ico_nearest_neighbor_summary.csv"
    plot_path = args.output_dir / "ico_nearest_neighbor_distributions.png"
    details.to_csv(detail_path, index=False, float_format="%.12g")
    summary.to_csv(summary_path, index=False, float_format="%.12g")
    plot_distributions(details, plot_path)

    validation: dict[str, object]
    if args.figure4_pair_cache is None:
        validation = {"status": "not_run", "reason": "no Figure 4 cache supplied"}
    else:
        validation = validate_against_figure4_cache(
            args.figure4_pair_cache,
            graphs,
            args.validation_pairs_per_hemisphere,
            args.validation_tolerance_mm,
        )
    validation["geometry"] = "fsaverage white mesh"
    validation["geodesic_implementation"] = (
        "Figure 4 build_graph edge weights plus scipy.sparse.csgraph.dijkstra"
    )
    validation["cache_reuse_decision"] = (
        "Figure 4 cache is a sampled sparse pair table, not a complete retained-node "
        "distance matrix; it was used for validation but cannot supply all nearest neighbors."
    )
    validation["vertex_order_dir"] = str(args.vertex_order_dir.resolve())
    validation["template_root"] = str(template_root.resolve())
    (args.output_dir / "ico_nearest_neighbor_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps({key: value for key, value in validation.items() if key != "checked_pairs"}, indent=2))


if __name__ == "__main__":
    main()
