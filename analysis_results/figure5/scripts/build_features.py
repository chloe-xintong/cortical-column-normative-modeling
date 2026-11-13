#!/usr/bin/env python3
"""Build fixed-order Figure 5 feature matrices from cleaned Option 3 profiles.

Description
-----------
Build fixed-order Figure 5 feature matrices from cleaned Option 3 profiles.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--cohort`` command-line input
- ``--profile-root`` command-line input
- ``--native-freesurfer-root`` command-line input
- ``--vertex-order-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/build_features.py --help
"""

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
LEVELS = (3, 4, 5)
N_FULL = 163842
N_DEPTH = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--native-freesurfer-root", type=Path, required=True)
    parser.add_argument("--vertex-order-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    return parser.parse_args()


def profile_path(root: Path, subject_id: str, hemisphere: str) -> Path:
    return root / subject_id / "profiles" / f"{subject_id}_{hemisphere}_md_profiles.npz"


def scalar_text(array: np.ndarray) -> str:
    return str(np.asarray(array).item())


def open_matrix(path: Path, shape: tuple[int, int]) -> np.memmap:
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    matrix[:] = np.nan
    return matrix


def write_feature_table(path: Path, rows: list[dict[str, object]]) -> None:
    table = pd.DataFrame(rows)
    table.insert(0, "feature_index", np.arange(len(table), dtype=np.int64))
    table.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = args.output_dir / "matrices"
    metadata_dir = args.output_dir / "feature_metadata"
    matrices_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    cohort = pd.read_csv(args.cohort, dtype={"subject_id": str})
    required = {"subject_id", "age_at_scan"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"Cohort is missing columns: {sorted(required - set(cohort.columns))}")
    cohort = cohort.loc[:, ["subject_id", "age_at_scan"]].copy()
    cohort["age_at_scan"] = pd.to_numeric(cohort["age_at_scan"], errors="raise")
    if args.max_subjects is not None:
        cohort = cohort.iloc[: args.max_subjects].copy()
    if cohort.empty or cohort["subject_id"].duplicated().any() or cohort["age_at_scan"].isna().any():
        raise ValueError("Subject order must be nonempty with unique IDs and complete ages")
    candidate_coverage_rows: list[dict[str, object]] = []
    for candidate_index, subject_id in enumerate(cohort["subject_id"].tolist()):
        n_mask = 0
        n_valid = 0
        for hemisphere in HEMISPHERES:
            path = profile_path(args.profile_root, subject_id, hemisphere)
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as profile:
                mask = profile["in_analysis_mask"].astype(bool)
                valid = profile["column_valid"].astype(bool)
            n_mask += int(mask.sum())
            n_valid += int((mask & valid).sum())
        coverage = n_valid / n_mask
        candidate_coverage_rows.append(
            {
                "candidate_row_index": candidate_index,
                "subject_id": subject_id,
                "coverage_fraction": coverage,
                "passes_minimum_coverage": bool(coverage >= args.minimum_coverage),
            }
        )
    candidate_coverage = pd.DataFrame(candidate_coverage_rows)
    candidate_coverage.to_csv(args.output_dir / "candidate_coverage_screen.csv", index=False)
    keep_ids = set(
        candidate_coverage.loc[candidate_coverage["passes_minimum_coverage"], "subject_id"]
    )
    cohort = cohort.loc[cohort["subject_id"].isin(keep_ids)].reset_index(drop=True)
    if cohort.empty:
        raise RuntimeError("No subjects passed the pre-specified coverage threshold")
    cohort.insert(0, "row_index", np.arange(len(cohort), dtype=np.int64))
    cohort.to_csv(args.output_dir / "subject_order.csv", index=False)
    subject_ids = cohort["subject_id"].tolist()
    n_subjects = len(subject_ids)

    node_tables: dict[int, pd.DataFrame] = {}
    for level in LEVELS:
        table = pd.read_csv(args.vertex_order_dir / f"ico{level}_vertex_order.csv")
        table = table.loc[table["in_analysis_mask"].astype(bool)].reset_index(drop=True)
        expected_order = table.sort_values(["hemisphere", "ico_node_id"], kind="stable").reset_index(drop=True)
        if not table.equals(expected_order):
            raise RuntimeError(f"ico{level} node table is not ordered hemisphere then ico_node_id")
        if table[["hemisphere", "ico_node_id"]].duplicated().any():
            raise RuntimeError(f"ico{level} node table has duplicate node keys")
        node_tables[level] = table

    reference = subject_ids[0]
    reference_data: dict[str, dict[str, np.ndarray]] = {}
    depth_reference: np.ndarray | None = None
    region_rows: list[dict[str, object]] = []
    region_keys: list[tuple[str, int, str]] = []
    for hemisphere in HEMISPHERES:
        path = profile_path(args.profile_root, reference, hemisphere)
        with np.load(path, allow_pickle=False) as profile:
            reference_data[hemisphere] = {
                "vertex_id": profile["fsaverage_vertex_id"].astype(np.int64),
                "labels": profile["template_label_id"].astype(np.int64),
                "names": profile["template_label_name"].astype(str),
                "analysis_mask": profile["in_analysis_mask"].astype(bool),
            }
            depths = profile["depth_fractions"].astype(np.float64)
            direction = scalar_text(profile["depth_direction"])
        if not np.array_equal(reference_data[hemisphere]["vertex_id"], np.arange(N_FULL)):
            raise RuntimeError(f"Reference {hemisphere} vertex IDs are not ordered full fsaverage")
        if depth_reference is None:
            depth_reference = depths
            depth_direction = direction
        elif not np.array_equal(depth_reference, depths) or depth_direction != direction:
            raise RuntimeError("Reference hemispheres disagree on depth convention")
        labels = reference_data[hemisphere]["labels"]
        names = reference_data[hemisphere]["names"]
        mask = reference_data[hemisphere]["analysis_mask"]
        for label_id in sorted(np.unique(labels[mask]).tolist()):
            label_names = np.unique(names[(labels == label_id) & mask])
            if len(label_names) != 1:
                raise RuntimeError(f"Ambiguous label name for {hemisphere} label {label_id}")
            key = (hemisphere, int(label_id), str(label_names[0]))
            region_keys.append(key)
            region_rows.append(
                {"hemisphere": hemisphere, "dk_label_id": int(label_id), "dk_label_name": str(label_names[0])}
            )
    observed_depths = 0 if depth_reference is None else len(depth_reference)
    if len(region_keys) != 68 or depth_reference is None or observed_depths != N_DEPTH:
        raise RuntimeError(
            f"Expected 68 DK regions and 21 depths, got {len(region_keys)} and {observed_depths}"
        )

    regional_profile_rows: list[dict[str, object]] = []
    for hemisphere, label_id, label_name in region_keys:
        for depth_index, depth_fraction in enumerate(depth_reference):
            regional_profile_rows.append(
                {
                    "hemisphere": hemisphere,
                    "dk_label_id": label_id,
                    "dk_label_name": label_name,
                    "depth_index": depth_index,
                    "depth_fraction": float(depth_fraction),
                    "depth_direction": depth_direction,
                }
            )
    write_feature_table(metadata_dir / "features_regional_mean_md.csv", region_rows)
    write_feature_table(metadata_dir / "features_regional_thickness.csv", region_rows)
    write_feature_table(metadata_dir / "features_regional_profile.csv", regional_profile_rows)

    node_feature_rows: dict[int, list[dict[str, object]]] = {}
    for level, table in node_tables.items():
        rows: list[dict[str, object]] = []
        for node in table.itertuples(index=False):
            for depth_index, depth_fraction in enumerate(depth_reference):
                rows.append(
                    {
                        "hemisphere": node.hemisphere,
                        "ico_level": level,
                        "ico_node_id": int(node.ico_node_id),
                        "fsaverage_vertex_id": int(node.fsaverage_vertex_id),
                        "dk_label_id": int(node.dk_label_id),
                        "dk_label_name": str(node.dk_label_name),
                        "depth_index": depth_index,
                        "depth_fraction": float(depth_fraction),
                        "depth_direction": depth_direction,
                    }
                )
        node_feature_rows[level] = rows
        write_feature_table(metadata_dir / f"features_ico{level}_md.csv", rows)
    ico3_scalar_rows = [
        {
            "hemisphere": row.hemisphere,
            "ico_level": 3,
            "ico_node_id": int(row.ico_node_id),
            "fsaverage_vertex_id": int(row.fsaverage_vertex_id),
            "dk_label_id": int(row.dk_label_id),
            "dk_label_name": str(row.dk_label_name),
        }
        for row in node_tables[3].itertuples(index=False)
    ]
    write_feature_table(metadata_dir / "features_ico3_column_length.csv", ico3_scalar_rows)
    write_feature_table(metadata_dir / "features_ico3_curvature.csv", ico3_scalar_rows)

    matrices: dict[str, np.memmap] = {
        "regional_mean_md": open_matrix(matrices_dir / "X_regional_mean_md.npy", (n_subjects, 68)),
        "regional_thickness": open_matrix(matrices_dir / "X_regional_thickness.npy", (n_subjects, 68)),
        "regional_profile": open_matrix(matrices_dir / "X_regional_profile.npy", (n_subjects, 68 * N_DEPTH)),
        "ico3_column_length": open_matrix(matrices_dir / "X_ico3_column_length.npy", (n_subjects, len(node_tables[3]))),
        "ico3_curvature": open_matrix(matrices_dir / "X_ico3_curvature.npy", (n_subjects, len(node_tables[3]))),
    }
    for level in LEVELS:
        matrices[f"ico{level}_md"] = open_matrix(
            matrices_dir / f"X_ico{level}_md.npy", (n_subjects, len(node_tables[level]) * N_DEPTH)
        )

    coverage_rows: list[dict[str, object]] = []
    curvature_qc_rows: list[dict[str, object]] = []
    spot_check_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20261027)
    spot_check_subject_rows = set(
        rng.choice(n_subjects, size=min(3, n_subjects), replace=False).astype(int).tolist()
    )
    for subject_index, subject_id in enumerate(subject_ids):
        regional_mean_values: list[float] = []
        regional_thickness_values: list[float] = []
        regional_profile_values: list[np.ndarray] = []
        per_subject: dict[str, dict[str, np.ndarray]] = {}
        for hemisphere in HEMISPHERES:
            path = profile_path(args.profile_root, subject_id, hemisphere)
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as profile:
                vertex_id = profile["fsaverage_vertex_id"].astype(np.int64)
                labels = profile["template_label_id"].astype(np.int64)
                names = profile["template_label_name"].astype(str)
                analysis_mask = profile["in_analysis_mask"].astype(bool)
                profiles = profile["profiles"].astype(np.float64)
                column_valid = profile["column_valid"].astype(bool)
                column_length = profile["column_length"].astype(np.float64)
                thickness = profile["cortical_thickness"].astype(np.float64)
                depths = profile["depth_fractions"].astype(np.float64)
                direction = scalar_text(profile["depth_direction"])
                triangles = profile["native_triangle_vertex_ids"].astype(np.int64)
                weights = profile["barycentric_weights"].astype(np.float64)
                mapping_valid = profile["mapping_valid"].astype(bool)
                is_zero_length = profile["is_zero_length"].astype(bool)
            reference_hemi = reference_data[hemisphere]
            if not np.array_equal(vertex_id, np.arange(N_FULL)):
                raise RuntimeError(f"row {subject_index} {hemisphere}: unstable full vertex IDs")
            if profiles.shape != (N_FULL, N_DEPTH) or triangles.shape != (N_FULL, 3) or weights.shape != (N_FULL, 3):
                raise RuntimeError(f"row {subject_index} {hemisphere}: unexpected array shape")
            if not np.array_equal(labels, reference_hemi["labels"]) or not np.array_equal(names, reference_hemi["names"]):
                raise RuntimeError(f"row {subject_index} {hemisphere}: DK labels differ from reference")
            if not np.array_equal(analysis_mask, reference_hemi["analysis_mask"]):
                raise RuntimeError(f"row {subject_index} {hemisphere}: analysis mask differs from reference")
            if not np.array_equal(depths, depth_reference) or direction != depth_direction:
                raise RuntimeError(f"row {subject_index} {hemisphere}: depth convention differs")
            per_subject[hemisphere] = {
                "profiles": profiles,
                "column_valid": column_valid,
                "column_length": column_length,
                "triangles": triangles,
                "weights": weights,
                "mapping_valid": mapping_valid,
                "is_zero_length": is_zero_length,
            }
            valid_mask = analysis_mask & column_valid
            coverage_rows.append(
                {
                    "row_index": subject_index,
                    "hemisphere": hemisphere,
                    "n_analysis_vertices": int(analysis_mask.sum()),
                    "n_valid_columns": int(valid_mask.sum()),
                    "coverage_fraction": float(valid_mask.sum() / analysis_mask.sum()),
                    "coverage_ge_0_95": bool(valid_mask.sum() / analysis_mask.sum() >= 0.95),
                }
            )
            for key_hemi, label_id, _ in region_keys:
                if key_hemi != hemisphere:
                    continue
                region_mask = analysis_mask & labels.__eq__(label_id)
                md_mask = region_mask & column_valid
                regional_mean_values.append(float(np.nanmean(profiles[md_mask])) if md_mask.any() else np.nan)
                regional_profile_values.append(
                    np.nanmean(profiles[md_mask], axis=0) if md_mask.any() else np.full(N_DEPTH, np.nan)
                )
                thickness_values = thickness[region_mask & np.isfinite(thickness)]
                regional_thickness_values.append(
                    float(np.mean(thickness_values)) if len(thickness_values) else np.nan
                )

        matrices["regional_mean_md"][subject_index] = np.asarray(regional_mean_values, dtype=np.float32)
        matrices["regional_thickness"][subject_index] = np.asarray(regional_thickness_values, dtype=np.float32)
        matrices["regional_profile"][subject_index] = np.concatenate(regional_profile_values).astype(np.float32)

        for level, table in node_tables.items():
            md_parts: list[np.ndarray] = []
            for hemisphere in HEMISPHERES:
                hemi_nodes = table.loc[table["hemisphere"].eq(hemisphere)]
                vertex_ids = hemi_nodes["fsaverage_vertex_id"].to_numpy(dtype=np.int64)
                values = per_subject[hemisphere]["profiles"][vertex_ids].copy()
                valid = per_subject[hemisphere]["column_valid"][vertex_ids]
                values[~valid] = np.nan
                md_parts.append(values)
            matrices[f"ico{level}_md"][subject_index] = np.concatenate(md_parts, axis=0).ravel().astype(np.float32)

        length_parts: list[np.ndarray] = []
        curvature_parts: list[np.ndarray] = []
        ico3_table = node_tables[3]
        for hemisphere in HEMISPHERES:
            hemi_nodes = ico3_table.loc[ico3_table["hemisphere"].eq(hemisphere)]
            vertex_ids = hemi_nodes["fsaverage_vertex_id"].to_numpy(dtype=np.int64)
            data = per_subject[hemisphere]
            length_values = data["column_length"][vertex_ids].copy()
            valid_length = data["mapping_valid"][vertex_ids]
            valid_length &= ~data["is_zero_length"][vertex_ids]
            valid_length &= np.isfinite(length_values)
            length_values[~valid_length] = np.nan
            length_parts.append(length_values)
            curv_path = args.native_freesurfer_root / subject_id / "surf" / f"{hemisphere}.curv"
            if not curv_path.is_file():
                raise FileNotFoundError(curv_path)
            native_curv = fsio.read_morph_data(curv_path).astype(np.float64)
            triangle_ids = data["triangles"][vertex_ids]
            weights = data["weights"][vertex_ids]
            valid_mapping = data["mapping_valid"][vertex_ids]
            valid_mapping &= np.all(triangle_ids >= 0, axis=1)
            valid_mapping &= np.all(triangle_ids < len(native_curv), axis=1)
            valid_mapping &= np.all(np.isfinite(weights), axis=1)
            valid_mapping &= np.isclose(weights.sum(axis=1), 1.0, atol=1e-5)
            curvature = np.full(len(vertex_ids), np.nan, dtype=np.float64)
            curvature[valid_mapping] = np.sum(
                native_curv[triangle_ids[valid_mapping]] * weights[valid_mapping], axis=1
            )
            curvature_parts.append(curvature)
            curvature_qc_rows.append(
                {
                    "row_index": subject_index,
                    "hemisphere": hemisphere,
                    "n_ico3_nodes": len(vertex_ids),
                    "n_valid_mappings": int(valid_mapping.sum()),
                    "finite_fraction": float(np.isfinite(curvature).mean()),
                    "maximum_weight_sum_absolute_error": float(
                        np.max(np.abs(weights.sum(axis=1) - 1.0))
                    ),
                    "native_curvature_vertex_count": len(native_curv),
                }
            )
        matrices["ico3_column_length"][subject_index] = np.concatenate(length_parts).astype(np.float32)
        matrices["ico3_curvature"][subject_index] = np.concatenate(curvature_parts).astype(np.float32)

        if subject_index in spot_check_subject_rows:
            for level, table in node_tables.items():
                for node_position in sorted({0, len(table) // 2, len(table) - 1}):
                    node = table.iloc[node_position]
                    depth_index = node_position % N_DEPTH
                    direct = per_subject[str(node["hemisphere"])]["profiles"][
                        int(node["fsaverage_vertex_id"]), depth_index
                    ]
                    if not per_subject[str(node["hemisphere"])]["column_valid"][
                        int(node["fsaverage_vertex_id"])
                    ]:
                        direct = np.nan
                    matrix_value = float(
                        matrices[f"ico{level}_md"][subject_index, node_position * N_DEPTH + depth_index]
                    )
                    spot_check_rows.append(
                        {
                            "row_index": subject_index,
                            "representation": f"ico{level}_md",
                            "feature_index": node_position * N_DEPTH + depth_index,
                            "source_value": float(direct),
                            "matrix_value": matrix_value,
                            "equal_allowing_nan": bool(np.isclose(direct, matrix_value, equal_nan=True)),
                        }
                    )
            scalar_sources = {
                "ico3_column_length": np.concatenate(length_parts),
                "ico3_curvature": np.concatenate(curvature_parts),
            }
            for name, source in scalar_sources.items():
                for feature_index in sorted({0, len(source) // 2, len(source) - 1}):
                    matrix_value = float(matrices[name][subject_index, feature_index])
                    spot_check_rows.append(
                        {
                            "row_index": subject_index,
                            "representation": name,
                            "feature_index": feature_index,
                            "source_value": float(source[feature_index]),
                            "matrix_value": matrix_value,
                            "equal_allowing_nan": bool(
                                np.isclose(source[feature_index], matrix_value, equal_nan=True)
                            ),
                        }
                    )
            regional_sources = {
                "regional_mean_md": np.asarray(regional_mean_values),
                "regional_thickness": np.asarray(regional_thickness_values),
                "regional_profile": np.concatenate(regional_profile_values),
            }
            for name, source in regional_sources.items():
                for feature_index in sorted({0, len(source) // 2, len(source) - 1}):
                    matrix_value = float(matrices[name][subject_index, feature_index])
                    spot_check_rows.append(
                        {
                            "row_index": subject_index,
                            "representation": name,
                            "feature_index": feature_index,
                            "source_value": float(source[feature_index]),
                            "matrix_value": matrix_value,
                            "equal_allowing_nan": bool(
                                np.isclose(source[feature_index], matrix_value, equal_nan=True)
                            ),
                        }
                    )

        for matrix in matrices.values():
            matrix.flush()
        if (subject_index + 1) % 10 == 0 or subject_index + 1 == n_subjects:
            print(f"completed_rows={subject_index + 1}/{n_subjects}", flush=True)

    coverage = pd.DataFrame(coverage_rows)
    bilateral = coverage.groupby("row_index", sort=True).agg(
        n_analysis_vertices=("n_analysis_vertices", "sum"),
        n_valid_columns=("n_valid_columns", "sum"),
    )
    bilateral["coverage_fraction"] = bilateral["n_valid_columns"] / bilateral["n_analysis_vertices"]
    bilateral["coverage_ge_0_95"] = bilateral["coverage_fraction"] >= 0.95
    coverage.to_csv(args.output_dir / "subject_coverage_by_hemisphere.csv", index=False)
    bilateral.reset_index().to_csv(args.output_dir / "subject_coverage.csv", index=False)
    curvature_qc = pd.DataFrame(curvature_qc_rows)
    curvature_qc.to_csv(args.output_dir / "curvature_mapping_qc.csv", index=False)
    spot_checks = pd.DataFrame(spot_check_rows)
    spot_checks.to_csv(args.output_dir / "source_value_spot_checks.csv", index=False)
    if not spot_checks["equal_allowing_nan"].all():
        raise RuntimeError("At least one anonymous source-value spot check failed")

    report = {
        "status": "complete",
        "n_subjects": n_subjects,
        "n_regions": len(region_keys),
        "n_depths": N_DEPTH,
        "depth_fractions": depth_reference.tolist(),
        "depth_direction": depth_direction,
        "matrix_shapes": {name: list(matrix.shape) for name, matrix in matrices.items()},
        "ico_nodes_after_analysis_mask": {f"ico{level}": len(node_tables[level]) for level in LEVELS},
        "curvature_method": (
            "subject-native FreeSurfer surf/<hemi>.curv mapped to full fsaverage with the stored "
            "native_triangle_vertex_ids and barycentric_weights; no group template curvature used"
        ),
        "curvature_convention": (
            "native FreeSurfer curv values are read unchanged with nibabel.freesurfer.io.read_morph_data; "
            "the native FreeSurfer sign and scale convention is preserved"
        ),
        "curvature_mapping_qc": {
            "minimum_finite_fraction": float(curvature_qc["finite_fraction"].min()),
            "maximum_weight_sum_absolute_error": float(
                curvature_qc["maximum_weight_sum_absolute_error"].max()
            ),
        },
        "source_value_spot_checks": {
            "n_checks": len(spot_checks),
            "n_anonymous_subject_rows": int(spot_checks["row_index"].nunique()),
            "all_passed": bool(spot_checks["equal_allowing_nan"].all()),
        },
        "curvature_source_root": str(args.native_freesurfer_root.resolve()),
        "profile_source_root": str(args.profile_root.resolve()),
        "cohort_source": str(args.cohort.resolve()),
        "coverage_threshold_recorded": args.minimum_coverage,
        "n_candidate_subjects": len(candidate_coverage),
        "n_excluded_below_coverage_threshold": int(
            (~candidate_coverage["passes_minimum_coverage"]).sum()
        ),
        "n_below_coverage_threshold": int((~bilateral["coverage_ge_0_95"]).sum()),
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.output_dir / "feature_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "n_subjects": n_subjects, "matrix_shapes": report["matrix_shapes"]}))


if __name__ == "__main__":
    main()
