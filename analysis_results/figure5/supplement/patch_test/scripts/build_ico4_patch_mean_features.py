#!/usr/bin/env python3
"""Build template-fixed ico4 geodesic Voronoi patch-mean MD features."""

from __future__ import annotations

import argparse
import heapq
import json
import resource
import sys
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


HEMISPHERES = ("lh", "rh")
N_FULL = 163842
N_DEPTH = 21
REPO_ROOT = Path(__file__).resolve().parents[5]
FIGURE4_SCRIPT_DIR = REPO_ROOT / "analysis_results" / "figure4" / "scripts"
sys.path.insert(0, str(FIGURE4_SCRIPT_DIR))
from build_geodesic_pair_tables import build_graph, valid_template_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--vertex-order-dir", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--spot-check-subjects", type=int, default=3)
    parser.add_argument("--spot-checks-per-subject", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261028)
    return parser.parse_args()


def profile_path(root: Path, subject_id: str, hemisphere: str) -> Path:
    return root / subject_id / "profiles" / f"{subject_id}_{hemisphere}_md_profiles.npz"


def multi_source_voronoi(graph: csr_matrix, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact mesh-edge distance and deterministic nearest-center position."""
    n_vertices = graph.shape[0]
    distance = np.full(n_vertices, np.inf, dtype=np.float64)
    owner = np.full(n_vertices, -1, dtype=np.int32)
    queue: list[tuple[float, int, int]] = []
    for center_position, vertex_id in enumerate(centers):
        distance[vertex_id] = 0.0
        owner[vertex_id] = center_position
        heapq.heappush(queue, (0.0, center_position, int(vertex_id)))
    indptr, indices, weights = graph.indptr, graph.indices, graph.data
    while queue:
        current_distance, current_owner, vertex = heapq.heappop(queue)
        if current_distance != distance[vertex] or current_owner != owner[vertex]:
            continue
        for edge_position in range(indptr[vertex], indptr[vertex + 1]):
            neighbor = int(indices[edge_position])
            candidate_distance = current_distance + float(weights[edge_position])
            old_distance = distance[neighbor]
            if candidate_distance < old_distance or (
                candidate_distance == old_distance and current_owner < owner[neighbor]
            ):
                distance[neighbor] = candidate_distance
                owner[neighbor] = current_owner
                heapq.heappush(queue, (candidate_distance, current_owner, neighbor))
    if np.any(owner < 0) or not np.isfinite(distance).all():
        raise RuntimeError("Template surface contains vertices unreachable from every ico4 center")
    return distance, owner


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                 vertices[faces[:, 2]] - vertices[faces[:, 0]]),
        axis=1,
    )
    result = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        np.add.at(result, faces[:, corner], triangle_area / 3.0)
    return result


def build_membership(
    template_root: Path, node_table: pd.DataFrame
) -> tuple[dict[str, dict[str, np.ndarray | csr_matrix]], pd.DataFrame, pd.DataFrame]:
    per_hemi: dict[str, dict[str, np.ndarray | csr_matrix]] = {}
    membership_rows: list[pd.DataFrame] = []
    patch_rows: list[dict[str, object]] = []
    for hemisphere in HEMISPHERES:
        vertices, faces = fsio.read_geometry(str(template_root / "surf" / f"{hemisphere}.white"))
        if len(vertices) != N_FULL:
            raise RuntimeError(f"{hemisphere}: expected {N_FULL} fsaverage vertices, got {len(vertices)}")
        mask, _ = valid_template_mask(template_root, hemisphere)
        nodes = node_table.loc[node_table["hemisphere"].eq(hemisphere)].reset_index(drop=True)
        centers = nodes["fsaverage_vertex_id"].to_numpy(dtype=np.int64)
        if not mask[centers].all():
            raise RuntimeError(f"{hemisphere}: a retained ico4 center is outside the analysis mask")
        graph = build_graph(vertices, faces)
        distance, owner_all = multi_source_voronoi(graph, centers)
        valid_vertices = np.flatnonzero(mask)
        owners = owner_all[valid_vertices]
        n_patches = len(nodes)
        counts = np.bincount(owners, minlength=n_patches)
        if np.any(counts == 0):
            raise RuntimeError(f"{hemisphere}: geodesic Voronoi construction produced an empty patch")
        areas = vertex_areas(vertices, faces)
        patch_area = np.bincount(owners, weights=areas[valid_vertices], minlength=n_patches)
        membership = csr_matrix(
            (np.ones(len(valid_vertices), dtype=np.float32), (owners, valid_vertices)),
            shape=(n_patches, N_FULL),
        )
        center_check = owner_all[centers]
        if not np.array_equal(center_check, np.arange(n_patches)):
            raise RuntimeError(f"{hemisphere}: at least one center does not own itself")
        per_hemi[hemisphere] = {
            "analysis_mask": mask,
            "centers": centers,
            "valid_vertices": valid_vertices,
            "owners": owners,
            "distance": distance[valid_vertices],
            "membership": membership,
            "patch_counts": counts,
        }
        member = pd.DataFrame(
            {
                "hemisphere": hemisphere,
                "fsaverage_vertex_id": valid_vertices,
                "patch_position_within_hemisphere": owners,
                "ico_node_id": nodes.iloc[owners]["ico_node_id"].to_numpy(dtype=np.int64),
                "center_fsaverage_vertex_id": centers[owners],
                "geodesic_distance_to_center_mm": distance[valid_vertices],
                "vertex_area_mm2": areas[valid_vertices],
            }
        )
        membership_rows.append(member)
        for patch_position, node in enumerate(nodes.itertuples(index=False)):
            selected_distance = distance[valid_vertices[owners == patch_position]]
            patch_rows.append(
                {
                    "hemisphere": hemisphere,
                    "ico_level": 4,
                    "ico_node_id": int(node.ico_node_id),
                    "center_fsaverage_vertex_id": int(node.fsaverage_vertex_id),
                    "dk_label_id_at_center": int(node.dk_label_id),
                    "dk_label_name_at_center": str(node.dk_label_name),
                    "n_full_resolution_vertices": int(counts[patch_position]),
                    "patch_area_mm2": float(patch_area[patch_position]),
                    "median_geodesic_radius_mm": float(np.median(selected_distance)),
                    "maximum_geodesic_radius_mm": float(np.max(selected_distance)),
                }
            )
    return per_hemi, pd.concat(membership_rows, ignore_index=True), pd.DataFrame(patch_rows)


def main() -> None:
    args = parse_args()
    start = time.time()
    matrices_dir = args.output_dir / "matrices"
    metadata_dir = args.output_dir / "feature_metadata"
    patch_dir = args.output_dir / "ico4_patch_mean_qc"
    for directory in (matrices_dir, metadata_dir, patch_dir):
        directory.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_csv(args.base_feature_dir / "subject_order.csv", dtype={"subject_id": str})
    if args.max_subjects is not None:
        if args.max_subjects < 1:
            raise ValueError("--max-subjects must be positive")
        subjects = subjects.iloc[: args.max_subjects].copy()
    if subjects["subject_id"].duplicated().any() or not np.array_equal(
        subjects["row_index"].to_numpy(), np.arange(len(subjects))
    ):
        raise RuntimeError("Feature subject order is duplicated or non-contiguous")
    subjects.to_csv(args.output_dir / "subject_order.csv", index=False)
    node_table = pd.read_csv(args.vertex_order_dir / "ico4_vertex_order.csv")
    node_table = node_table.loc[node_table["in_analysis_mask"].astype(bool)].reset_index(drop=True)
    expected = node_table.sort_values(["hemisphere", "ico_node_id"], kind="stable").reset_index(drop=True)
    if not node_table.equals(expected):
        raise RuntimeError("ico4 node table is not ordered hemisphere then ico_node_id")
    template_root = args.subjects_dir / args.template
    per_hemi, membership_table, patch_summary = build_membership(template_root, node_table)
    if len(patch_summary) != len(node_table):
        raise RuntimeError("Patch count does not match retained ico4 center count")

    membership_table.to_csv(
        patch_dir / "ico4_geodesic_voronoi_membership.csv.gz",
        index=False,
        compression="gzip",
    )
    patch_summary.insert(0, "patch_index", np.arange(len(patch_summary), dtype=np.int64))
    patch_summary.to_csv(patch_dir / "ico4_patch_geometry.csv", index=False)
    np.savez_compressed(
        patch_dir / "ico4_geodesic_voronoi_membership.npz",
        hemisphere=membership_table["hemisphere"].to_numpy(dtype="U2"),
        fsaverage_vertex_id=membership_table["fsaverage_vertex_id"].to_numpy(dtype=np.int32),
        patch_position_within_hemisphere=membership_table[
            "patch_position_within_hemisphere"
        ].to_numpy(dtype=np.int32),
        geodesic_distance_to_center_mm=membership_table[
            "geodesic_distance_to_center_mm"
        ].to_numpy(dtype=np.float32),
    )

    with np.load(profile_path(args.profile_root, subjects.iloc[0]["subject_id"], "lh"), allow_pickle=False) as ref:
        depths = ref["depth_fractions"].astype(np.float64)
        direction = str(np.asarray(ref["depth_direction"]).item())
    if len(depths) != N_DEPTH:
        raise RuntimeError(f"Expected {N_DEPTH} depths, got {len(depths)}")
    feature_rows: list[dict[str, object]] = []
    for patch in patch_summary.itertuples(index=False):
        for depth_index, depth_fraction in enumerate(depths):
            feature_rows.append(
                {
                    "hemisphere": patch.hemisphere,
                    "ico_level": 4,
                    "ico_node_id": patch.ico_node_id,
                    "fsaverage_vertex_id": patch.center_fsaverage_vertex_id,
                    "dk_label_id": patch.dk_label_id_at_center,
                    "dk_label_name": patch.dk_label_name_at_center,
                    "n_full_resolution_vertices": patch.n_full_resolution_vertices,
                    "patch_area_mm2": patch.patch_area_mm2,
                    "median_geodesic_radius_mm": patch.median_geodesic_radius_mm,
                    "maximum_geodesic_radius_mm": patch.maximum_geodesic_radius_mm,
                    "depth_index": depth_index,
                    "depth_fraction": float(depth_fraction),
                    "depth_direction": direction,
                }
            )
    feature_metadata = pd.DataFrame(feature_rows)
    feature_metadata.insert(0, "feature_index", np.arange(len(feature_metadata), dtype=np.int64))
    feature_metadata.to_csv(metadata_dir / "features_ico4_patch_mean_md.csv", index=False)

    shape = (len(subjects), len(feature_metadata))
    matrix = np.lib.format.open_memmap(
        matrices_dir / "X_ico4_patch_mean_md.npy", mode="w+", dtype=np.float32, shape=shape
    )
    matrix[:] = np.nan
    coverage = np.lib.format.open_memmap(
        patch_dir / "ico4_patch_depth_coverage.npy", mode="w+", dtype=np.float32, shape=shape
    )
    coverage[:] = np.nan
    rng = np.random.default_rng(args.seed)
    check_subjects = set(
        rng.choice(len(subjects), size=min(args.spot_check_subjects, len(subjects)), replace=False).tolist()
    )
    spot_rows: list[dict[str, object]] = []
    for subject_index, subject_id in enumerate(subjects["subject_id"]):
        mean_parts: list[np.ndarray] = []
        coverage_parts: list[np.ndarray] = []
        source_by_hemi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for hemisphere in HEMISPHERES:
            with np.load(profile_path(args.profile_root, subject_id, hemisphere), allow_pickle=False) as profile:
                profiles = profile["profiles"].astype(np.float64)
                column_valid = profile["column_valid"].astype(bool)
                analysis_mask = profile["in_analysis_mask"].astype(bool)
                current_depths = profile["depth_fractions"].astype(np.float64)
                current_direction = str(np.asarray(profile["depth_direction"]).item())
            if profiles.shape != (N_FULL, N_DEPTH):
                raise RuntimeError(f"{subject_id} {hemisphere}: unexpected profile shape {profiles.shape}")
            fixed_mask = per_hemi[hemisphere]["analysis_mask"]
            if not np.array_equal(analysis_mask, fixed_mask):
                raise RuntimeError(f"{subject_id} {hemisphere}: analysis mask differs from template")
            if not np.array_equal(current_depths, depths) or current_direction != direction:
                raise RuntimeError(f"{subject_id} {hemisphere}: depth convention differs")
            finite = np.isfinite(profiles) & column_valid[:, None] & analysis_mask[:, None]
            pooling = per_hemi[hemisphere]["membership"]
            assert isinstance(pooling, csr_matrix)
            sums = np.asarray(pooling @ np.where(finite, profiles, 0.0))
            valid_counts = np.asarray(pooling @ finite.astype(np.float32))
            means = np.full(sums.shape, np.nan, dtype=np.float64)
            np.divide(sums, valid_counts, out=means, where=valid_counts > 0)
            denominators = np.asarray(per_hemi[hemisphere]["patch_counts"], dtype=np.float64)[:, None]
            fractions = valid_counts / denominators
            mean_parts.append(means)
            coverage_parts.append(fractions)
            source_by_hemi[hemisphere] = (profiles, finite)
        joined_means = np.concatenate(mean_parts, axis=0)
        joined_coverage = np.concatenate(coverage_parts, axis=0)
        matrix[subject_index] = joined_means.ravel().astype(np.float32)
        coverage[subject_index] = joined_coverage.ravel().astype(np.float32)
        if subject_index in check_subjects:
            feature_indices = rng.choice(
                joined_means.size,
                size=min(args.spot_checks_per_subject, joined_means.size),
                replace=False,
            )
            for feature_index in feature_indices:
                patch_index, depth_index = divmod(int(feature_index), N_DEPTH)
                patch = patch_summary.iloc[patch_index]
                hemisphere = str(patch["hemisphere"])
                hemi_patch_position = int(
                    (patch_summary.iloc[:patch_index]["hemisphere"] == hemisphere).sum()
                )
                info = per_hemi[hemisphere]
                members = np.asarray(info["valid_vertices"])[
                    np.asarray(info["owners"]) == hemi_patch_position
                ]
                profiles, finite = source_by_hemi[hemisphere]
                usable = finite[members, depth_index]
                direct = float(np.mean(profiles[members[usable], depth_index])) if usable.any() else np.nan
                observed = float(matrix[subject_index, feature_index])
                spot_rows.append(
                    {
                        "row_index": subject_index,
                        "feature_index": int(feature_index),
                        "hemisphere": hemisphere,
                        "ico_node_id": int(patch["ico_node_id"]),
                        "depth_index": depth_index,
                        "n_patch_vertices": len(members),
                        "n_valid_vertices": int(usable.sum()),
                        "direct_mean": direct,
                        "matrix_mean": observed,
                        "equal_allowing_nan": bool(np.isclose(direct, observed, rtol=1e-6, equal_nan=True)),
                    }
                )
        matrix.flush()
        coverage.flush()
        if (subject_index + 1) % 10 == 0 or subject_index + 1 == len(subjects):
            print(f"completed_rows={subject_index + 1}/{len(subjects)}", flush=True)

    spot_checks = pd.DataFrame(spot_rows)
    spot_checks.to_csv(patch_dir / "source_value_spot_checks.csv", index=False)
    if spot_checks.empty or not spot_checks["equal_allowing_nan"].all():
        raise RuntimeError("At least one patch-mean source-value spot check failed")
    coverage_array = np.asarray(coverage)
    feature_coverage = feature_metadata[["feature_index"]].copy()
    feature_coverage["minimum_coverage_fraction"] = np.min(coverage_array, axis=0)
    feature_coverage["median_coverage_fraction"] = np.median(coverage_array, axis=0)
    feature_coverage["mean_coverage_fraction"] = np.mean(coverage_array, axis=0)
    feature_coverage["n_zero_coverage_subjects"] = np.sum(coverage_array == 0, axis=0)
    feature_coverage.to_csv(patch_dir / "patch_depth_coverage_by_feature.csv", index=False)
    subject_coverage = subjects[["row_index", "subject_id"]].copy()
    subject_coverage["minimum_patch_depth_coverage_fraction"] = np.min(coverage_array, axis=1)
    subject_coverage["median_patch_depth_coverage_fraction"] = np.median(coverage_array, axis=1)
    subject_coverage["mean_patch_depth_coverage_fraction"] = np.mean(coverage_array, axis=1)
    subject_coverage["n_zero_coverage_patch_depths"] = np.sum(coverage_array == 0, axis=1)
    subject_coverage.to_csv(patch_dir / "patch_depth_coverage_by_subject.csv", index=False)

    hemisphere_summary = patch_summary.groupby("hemisphere", sort=False).agg(
        n_patches=("patch_index", "size"),
        total_vertices=("n_full_resolution_vertices", "sum"),
        minimum_vertices=("n_full_resolution_vertices", "min"),
        median_vertices=("n_full_resolution_vertices", "median"),
        maximum_vertices=("n_full_resolution_vertices", "max"),
        total_area_mm2=("patch_area_mm2", "sum"),
        median_area_mm2=("patch_area_mm2", "median"),
        median_of_patch_median_radius_mm=("median_geodesic_radius_mm", "median"),
        maximum_radius_mm=("maximum_geodesic_radius_mm", "max"),
    ).reset_index()
    hemisphere_summary["n_empty_patches"] = 0
    hemisphere_summary.to_csv(patch_dir / "ico4_patch_geometry_by_hemisphere.csv", index=False)
    report = {
        "status": "complete",
        "representation": "ico4_patch_mean_md",
        "definition": "mean MD at each depth over fixed fsaverage white-surface mesh-edge geodesic Voronoi patches",
        "template": args.template,
        "template_root": str(template_root.resolve()),
        "base_feature_dir": str(args.base_feature_dir.resolve()),
        "membership_uses_outcomes": False,
        "membership_fixed_across_subjects": True,
        "invalid_columns": "excluded in place from each patch/depth mean; zero valid vertices yields NaN",
        "n_subjects": len(subjects),
        "n_patches": len(patch_summary),
        "n_depths": N_DEPTH,
        "matrix_shape": list(shape),
        "n_empty_patches": 0,
        "minimum_patch_depth_coverage_fraction": float(np.min(coverage_array)),
        "median_patch_depth_coverage_fraction": float(np.median(coverage_array)),
        "n_zero_coverage_patch_depth_cells": int(np.sum(coverage_array == 0)),
        "spot_checks": {"n": len(spot_checks), "all_passed": True},
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (patch_dir / "ico4_patch_mean_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
