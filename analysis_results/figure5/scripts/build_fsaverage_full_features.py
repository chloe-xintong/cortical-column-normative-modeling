#!/usr/bin/env python3
"""Build the full-resolution fsaverage 21-depth MD feature matrix.

Description
-----------
Build the full-resolution fsaverage 21-depth MD feature matrix.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--feature-dir`` command-line input
- ``--profile-root`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/build_fsaverage_full_features.py --help
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd


HEMISPHERES = ("lh", "rh")
N_FULL = 163_842
N_DEPTH = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    return parser.parse_args()


def profile_path(root: Path, subject_id: str, hemisphere: str) -> Path:
    return root / subject_id / "profiles" / f"{subject_id}_{hemisphere}_md_profiles.npz"


def main() -> None:
    args = parse_args()
    started = time.time()
    subject_order = pd.read_csv(
        args.feature_dir / "subject_order.csv", dtype={"subject_id": str}
    )
    if subject_order["subject_id"].duplicated().any() or not np.array_equal(
        subject_order["row_index"].to_numpy(), np.arange(len(subject_order))
    ):
        raise RuntimeError("subject_order.csv is duplicated or non-contiguous")

    reference_id = str(subject_order.iloc[0]["subject_id"])
    reference_masks: dict[str, np.ndarray] = {}
    vertex_rows: list[dict[str, object]] = []
    feature_offset = 0
    depth_reference: np.ndarray | None = None
    direction_reference: str | None = None
    for hemisphere in HEMISPHERES:
        with np.load(profile_path(args.profile_root, reference_id, hemisphere)) as data:
            vertex_ids = data["fsaverage_vertex_id"].astype(np.int64)
            mask = data["in_analysis_mask"].astype(bool)
            labels = data["template_label_id"].astype(np.int64)
            names = data["template_label_name"].astype(str)
            depths = data["depth_fractions"].astype(np.float64)
            direction = str(np.asarray(data["depth_direction"]).item())
        if not np.array_equal(vertex_ids, np.arange(N_FULL)):
            raise RuntimeError(f"{hemisphere}: full fsaverage vertex order is not contiguous")
        if depth_reference is None:
            depth_reference = depths
            direction_reference = direction
        elif not np.array_equal(depth_reference, depths) or direction_reference != direction:
            raise RuntimeError("Reference hemispheres disagree on depth convention")
        reference_masks[hemisphere] = mask
        for vertex_id in np.flatnonzero(mask):
            vertex_rows.append(
                {
                    "vertex_order": len(vertex_rows),
                    "hemisphere": hemisphere,
                    "fsaverage_vertex_id": int(vertex_id),
                    "dk_label_id": int(labels[vertex_id]),
                    "dk_label_name": str(names[vertex_id]),
                    "feature_start_index": feature_offset,
                    "feature_stop_index_exclusive": feature_offset + N_DEPTH,
                }
            )
            feature_offset += N_DEPTH

    if depth_reference is None or len(depth_reference) != N_DEPTH:
        raise RuntimeError("Expected 21 profile depths")
    n_subjects = len(subject_order)
    n_vertices = len(vertex_rows)
    n_features = n_vertices * N_DEPTH
    matrix_dir = args.feature_dir / "matrices"
    metadata_dir = args.feature_dir / "feature_metadata"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    final_path = matrix_dir / "X_fsaverage_full_md.npy"
    building_path = matrix_dir / "X_fsaverage_full_md.building.npy"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing matrix: {final_path}")
    if building_path.exists():
        building_path.unlink()
    matrix = np.lib.format.open_memmap(
        building_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_subjects, n_features),
    )
    matrix[:] = np.nan

    coverage_rows: list[dict[str, object]] = []
    try:
        for row_index, subject_id in enumerate(subject_order["subject_id"].tolist()):
            parts: list[np.ndarray] = []
            for hemisphere in HEMISPHERES:
                with np.load(profile_path(args.profile_root, subject_id, hemisphere)) as data:
                    vertex_ids = data["fsaverage_vertex_id"].astype(np.int64)
                    mask = data["in_analysis_mask"].astype(bool)
                    profiles = data["profiles"].astype(np.float32)
                    valid = data["column_valid"].astype(bool)
                    depths = data["depth_fractions"].astype(np.float64)
                    direction = str(np.asarray(data["depth_direction"]).item())
                if not np.array_equal(vertex_ids, np.arange(N_FULL)):
                    raise RuntimeError(f"{subject_id} {hemisphere}: vertex order changed")
                if not np.array_equal(mask, reference_masks[hemisphere]):
                    raise RuntimeError(f"{subject_id} {hemisphere}: analysis mask changed")
                if profiles.shape != (N_FULL, N_DEPTH):
                    raise RuntimeError(f"{subject_id} {hemisphere}: profile shape changed")
                if not np.array_equal(depths, depth_reference) or direction != direction_reference:
                    raise RuntimeError(f"{subject_id} {hemisphere}: depth convention changed")
                values = profiles[mask].copy()
                retained_valid = valid[mask]
                values[~retained_valid] = np.nan
                parts.append(values)
                coverage_rows.append(
                    {
                        "row_index": row_index,
                        "subject_id": subject_id,
                        "hemisphere": hemisphere,
                        "n_vertices": int(mask.sum()),
                        "n_valid_columns": int(retained_valid.sum()),
                        "coverage_fraction": float(retained_valid.mean()),
                    }
                )
            matrix[row_index] = np.concatenate(parts, axis=0).ravel()
            matrix.flush()
            if (row_index + 1) % 10 == 0 or row_index + 1 == n_subjects:
                print(f"completed_rows={row_index + 1}/{n_subjects}", flush=True)
        del matrix
        building_path.replace(final_path)
    except Exception:
        del matrix
        raise

    vertex_table = pd.DataFrame(vertex_rows)
    vertex_table.to_csv(
        metadata_dir / "features_fsaverage_full_md_vertices.csv", index=False
    )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.feature_dir / "fsaverage_full_coverage.csv", index=False)
    report = {
        "status": "complete",
        "representation": "fsaverage_full_md",
        "n_subjects": n_subjects,
        "n_vertices": n_vertices,
        "n_depths": N_DEPTH,
        "n_features": n_features,
        "matrix_shape": [n_subjects, n_features],
        "matrix_dtype": "float32",
        "vertex_order": "left hemisphere then right hemisphere; ascending fsaverage vertex ID",
        "feature_order": "vertex-major, then pial-to-white depth index",
        "depth_fractions": depth_reference.tolist(),
        "depth_direction": direction_reference,
        "minimum_subject_hemisphere_coverage": float(coverage.coverage_fraction.min()),
        "profile_root": str(args.profile_root.resolve()),
        "subject_order": str((args.feature_dir / "subject_order.csv").resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.feature_dir / "fsaverage_full_feature_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
