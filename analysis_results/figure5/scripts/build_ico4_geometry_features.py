#!/usr/bin/env python3
"""Build ico4 column-length and subject-specific curvature matrices."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


HEMISPHERES = ("lh", "rh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--native-freesurfer-root", type=Path, required=True)
    parser.add_argument("--vertex-order", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--max-subjects", type=int)
    return parser.parse_args()


def write_metadata(path: Path, table: pd.DataFrame) -> None:
    output = table[
        [
            "hemisphere",
            "ico_level",
            "ico_node_id",
            "fsaverage_vertex_id",
            "dk_label_id",
            "dk_label_name",
        ]
    ].copy()
    output.insert(0, "feature_index", np.arange(len(output), dtype=np.int64))
    output.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    start = time.time()
    matrices_dir = args.feature_dir / "matrices"
    metadata_dir = args.feature_dir / "feature_metadata"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    if args.max_subjects is not None:
        subjects = subjects.iloc[: args.max_subjects].copy()
    if subjects.empty or subjects["subject_id"].duplicated().any():
        raise RuntimeError("Subject order must be nonempty and unique")
    nodes = pd.read_csv(args.vertex_order)
    nodes = nodes.loc[nodes["in_analysis_mask"].astype(bool)].reset_index(drop=True)
    expected = nodes.sort_values(["hemisphere", "ico_node_id"], kind="stable").reset_index(
        drop=True
    )
    if not nodes.equals(expected) or len(nodes) != 4687:
        raise RuntimeError("ico4 fixed node table is not the expected ordered 4,687-node mask")
    write_metadata(metadata_dir / "features_ico4_column_length.csv", nodes)
    write_metadata(metadata_dir / "features_ico4_curvature.csv", nodes)
    n_subjects = len(subjects)
    n_nodes = len(nodes)
    length_matrix = np.lib.format.open_memmap(
        matrices_dir / "X_ico4_column_length.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_subjects, n_nodes),
    )
    curvature_matrix = np.lib.format.open_memmap(
        matrices_dir / "X_ico4_curvature.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_subjects, n_nodes),
    )
    length_matrix[:] = np.nan
    curvature_matrix[:] = np.nan
    qc_rows: list[dict[str, object]] = []
    spot_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20261028)
    spot_subjects = set(
        rng.choice(n_subjects, size=min(3, n_subjects), replace=False).astype(int).tolist()
    )
    for row_index, subject_id in enumerate(subjects["subject_id"]):
        length_parts: list[np.ndarray] = []
        curvature_parts: list[np.ndarray] = []
        for hemisphere in HEMISPHERES:
            hemi_nodes = nodes.loc[nodes["hemisphere"].eq(hemisphere)]
            vertex_ids = hemi_nodes["fsaverage_vertex_id"].to_numpy(dtype=np.int64)
            profile_path = (
                args.profile_root
                / subject_id
                / "profiles"
                / f"{subject_id}_{hemisphere}_md_profiles.npz"
            )
            with np.load(profile_path, allow_pickle=False) as profile:
                full_ids = profile["fsaverage_vertex_id"].astype(np.int64)
                column_length = profile["column_length"].astype(np.float64)
                mapping_valid = profile["mapping_valid"].astype(bool)
                is_zero_length = profile["is_zero_length"].astype(bool)
                triangles = profile["native_triangle_vertex_ids"].astype(np.int64)
                weights = profile["barycentric_weights"].astype(np.float64)
            if not np.array_equal(full_ids, np.arange(163842)):
                raise RuntimeError(f"row {row_index} {hemisphere}: unstable fsaverage vertex order")
            length = column_length[vertex_ids].copy()
            valid_length = mapping_valid[vertex_ids] & ~is_zero_length[vertex_ids]
            valid_length &= np.isfinite(length)
            length[~valid_length] = np.nan
            length_parts.append(length)
            curv_path = args.native_freesurfer_root / subject_id / "surf" / f"{hemisphere}.curv"
            native_curv = fsio.read_morph_data(curv_path).astype(np.float64)
            selected_triangles = triangles[vertex_ids]
            selected_weights = weights[vertex_ids]
            valid_curvature = mapping_valid[vertex_ids]
            valid_curvature &= np.all(selected_triangles >= 0, axis=1)
            valid_curvature &= np.all(selected_triangles < len(native_curv), axis=1)
            valid_curvature &= np.all(np.isfinite(selected_weights), axis=1)
            valid_curvature &= np.isclose(selected_weights.sum(axis=1), 1.0, atol=1e-5)
            curvature = np.full(len(vertex_ids), np.nan, dtype=np.float64)
            curvature[valid_curvature] = np.sum(
                native_curv[selected_triangles[valid_curvature]]
                * selected_weights[valid_curvature],
                axis=1,
            )
            curvature_parts.append(curvature)
            qc_rows.append(
                {
                    "row_index": row_index,
                    "hemisphere": hemisphere,
                    "n_nodes": len(vertex_ids),
                    "length_finite_fraction": float(np.isfinite(length).mean()),
                    "curvature_finite_fraction": float(np.isfinite(curvature).mean()),
                    "maximum_weight_sum_absolute_error": float(
                        np.max(np.abs(selected_weights.sum(axis=1) - 1.0))
                    ),
                }
            )
        length_values = np.concatenate(length_parts)
        curvature_values = np.concatenate(curvature_parts)
        length_matrix[row_index] = length_values.astype(np.float32)
        curvature_matrix[row_index] = curvature_values.astype(np.float32)
        if row_index in spot_subjects:
            for name, source, matrix in (
                ("ico4_column_length", length_values, length_matrix),
                ("ico4_curvature", curvature_values, curvature_matrix),
            ):
                for feature_index in sorted({0, n_nodes // 2, n_nodes - 1}):
                    observed = float(matrix[row_index, feature_index])
                    spot_rows.append(
                        {
                            "row_index": row_index,
                            "representation": name,
                            "feature_index": feature_index,
                            "source_value": float(source[feature_index]),
                            "matrix_value": observed,
                            "equal_allowing_nan": bool(
                                np.isclose(source[feature_index], observed, equal_nan=True)
                            ),
                        }
                    )
        if (row_index + 1) % 25 == 0 or row_index + 1 == n_subjects:
            length_matrix.flush()
            curvature_matrix.flush()
            print(f"completed_rows={row_index + 1}/{n_subjects}", flush=True)
    qc = pd.DataFrame(qc_rows)
    spots = pd.DataFrame(spot_rows)
    qc.to_csv(args.feature_dir / "ico4_geometry_mapping_qc.csv", index=False)
    spots.to_csv(args.feature_dir / "ico4_geometry_source_value_spot_checks.csv", index=False)
    if not spots["equal_allowing_nan"].all():
        raise RuntimeError("At least one ico4 geometry source-value spot check failed")
    report = {
        "status": "complete",
        "n_subjects": n_subjects,
        "n_ico4_nodes": n_nodes,
        "matrix_shapes": {
            "ico4_column_length": list(length_matrix.shape),
            "ico4_curvature": list(curvature_matrix.shape),
        },
        "curvature_method": (
            "subject-native FreeSurfer curvature mapped with stored native triangle vertex IDs "
            "and barycentric weights; native sign and scale preserved"
        ),
        "minimum_length_finite_fraction": float(qc["length_finite_fraction"].min()),
        "minimum_curvature_finite_fraction": float(qc["curvature_finite_fraction"].min()),
        "maximum_weight_sum_absolute_error": float(
            qc["maximum_weight_sum_absolute_error"].max()
        ),
        "source_value_spot_checks": len(spots),
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.feature_dir / "ico4_geometry_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
