#!/usr/bin/env python3
"""Validate ico4 patch membership, geometry, coverage, and feature ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_DEPTH = 21
HEMISPHERES = ("lh", "rh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--vertex-order-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = args.feature_dir / "ico4_patch_mean_qc"
    subjects = pd.read_csv(args.feature_dir / "subject_order.csv", dtype={"subject_id": str})
    nodes = pd.read_csv(args.vertex_order_dir / "ico4_vertex_order.csv")
    nodes = nodes.loc[nodes["in_analysis_mask"].astype(bool)].reset_index(drop=True)
    metadata = pd.read_csv(
        args.feature_dir / "feature_metadata" / "features_ico4_patch_mean_md.csv"
    )
    geometry = pd.read_csv(patch_dir / "ico4_patch_geometry.csv")
    membership = pd.read_csv(patch_dir / "ico4_geodesic_voronoi_membership.csv.gz")
    matrix = np.load(args.feature_dir / "matrices" / "X_ico4_patch_mean_md.npy", mmap_mode="r")
    coverage = np.load(patch_dir / "ico4_patch_depth_coverage.npy", mmap_mode="r")

    observed_nodes = metadata.drop_duplicates(["hemisphere", "ico_node_id"])[
        ["hemisphere", "ico_node_id", "fsaverage_vertex_id"]
    ].reset_index(drop=True)
    expected_nodes = nodes[["hemisphere", "ico_node_id", "fsaverage_vertex_id"]].reset_index(drop=True)
    membership_counts = membership.groupby(
        ["hemisphere", "ico_node_id"], sort=True
    ).size().rename("n").reset_index()
    geometry_counts = geometry[
        ["hemisphere", "ico_node_id", "n_full_resolution_vertices"]
    ].rename(columns={"n_full_resolution_vertices": "n"}).sort_values(
        ["hemisphere", "ico_node_id"], kind="stable"
    ).reset_index(drop=True)
    owned_centers = membership.loc[
        membership["fsaverage_vertex_id"].eq(membership["center_fsaverage_vertex_id"]),
        ["hemisphere", "ico_node_id"],
    ].drop_duplicates()
    checks = {
        "matrix_shape_matches_subjects_and_metadata": matrix.shape == (len(subjects), len(metadata)),
        "coverage_shape_matches_matrix": coverage.shape == matrix.shape,
        "feature_index_contiguous": np.array_equal(
            metadata["feature_index"].to_numpy(), np.arange(len(metadata))
        ),
        "exactly_21_depths_per_patch": bool(
            metadata.groupby(["hemisphere", "ico_node_id"], sort=False).size().eq(N_DEPTH).all()
        ),
        "patch_order_matches_ico4": observed_nodes.equals(expected_nodes),
        "patch_count_matches_ico4": len(geometry) == len(nodes),
        "no_empty_patches": bool(geometry["n_full_resolution_vertices"].gt(0).all()),
        "membership_unique_within_hemisphere": not bool(
            membership[["hemisphere", "fsaverage_vertex_id"]].duplicated().any()
        ),
        "membership_counts_match_geometry": membership_counts.equals(geometry_counts),
        "every_center_owns_itself": len(owned_centers) == len(geometry),
        "finite_nonnegative_geometry": bool(
            np.isfinite(
                geometry[
                    ["patch_area_mm2", "median_geodesic_radius_mm", "maximum_geodesic_radius_mm"]
                ]
            ).all().all()
            and (
                geometry[
                    ["patch_area_mm2", "median_geodesic_radius_mm", "maximum_geodesic_radius_mm"]
                ]
                >= 0
            ).all().all()
        ),
        "coverage_in_unit_interval": bool(
            np.isfinite(coverage).all() and np.min(coverage) >= 0 and np.max(coverage) <= 1
        ),
        "matrix_contains_no_infinity": not bool(np.isinf(matrix).any()),
        "matrix_nan_exactly_when_coverage_zero": bool(
            np.array_equal(np.isnan(matrix), np.asarray(coverage) == 0)
        ),
    }
    hemi_rows = []
    for hemisphere in HEMISPHERES:
        current = geometry.loc[geometry["hemisphere"].eq(hemisphere)]
        hemi_rows.append(
            {
                "hemisphere": hemisphere,
                "n_patches": len(current),
                "n_vertices": int(current["n_full_resolution_vertices"].sum()),
                "minimum_patch_vertices": int(current["n_full_resolution_vertices"].min()),
                "median_patch_vertices": float(current["n_full_resolution_vertices"].median()),
                "maximum_patch_vertices": int(current["n_full_resolution_vertices"].max()),
                "median_patch_area_mm2": float(current["patch_area_mm2"].median()),
                "median_patch_radius_mm": float(current["median_geodesic_radius_mm"].median()),
                "maximum_patch_radius_mm": float(current["maximum_geodesic_radius_mm"].max()),
                "n_empty_patches": int(current["n_full_resolution_vertices"].eq(0).sum()),
            }
        )
    pd.DataFrame(hemi_rows).to_csv(args.output_dir / "ico4_patch_qc_by_hemisphere.csv", index=False)
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "representation": "ico4_patch_mean_md",
        "n_subjects": len(subjects),
        "n_patches": len(geometry),
        "n_features": len(metadata),
        "checks": checks,
        "minimum_coverage_fraction": float(np.min(coverage)),
        "median_coverage_fraction": float(np.median(coverage)),
        "maximum_feature_missing_fraction": float(np.mean(~np.isfinite(matrix), axis=0).max()),
    }
    (args.output_dir / "ico4_patch_mean_qc_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    if report["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"ico4 patch-mean QC failed: {failed}")


if __name__ == "__main__":
    main()
