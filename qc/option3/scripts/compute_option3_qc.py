#!/usr/bin/env python3
"""Compute cached O3 mapping, geometry, sampling, label, and registration QC.

Description
-----------
Compute cached O3 mapping, geometry, sampling, label, and registration QC.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--output-root`` command-line input
- ``--subjects-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/option3/scripts/compute_option3_qc.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from collections import deque
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy import stats


DEPTHS = np.linspace(0.0, 1.0, 21)
HEMISPHERES = ("lh", "rh")
EXCLUDED_LABELS = {
    "unknown",
    "corpuscallosum",
    "medial_wall",
    "medialwall",
    "background",
    "???",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    parser.add_argument("--zero-length-tolerance-mm", type=float, default=1e-6)
    parser.add_argument("--near-zero-md-threshold", type=float, default=1e-12)
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = data[key].item()
    return value.decode() if isinstance(value, bytes) else str(value)


def normalize_label(name: str | bytes) -> str:
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return str(name).lower().replace("-", "_").replace(" ", "_")


def select_subject(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.index is not None:
        if args.index < 0 or args.index >= len(manifest):
            raise IndexError(
                f"Manifest index {args.index} is outside 0..{len(manifest)-1}"
            )
        return manifest.iloc[args.index]
    matches = manifest.loc[manifest["subject_id"].eq(args.subject_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for {args.subject_id}")
    return matches.iloc[0]


def template_boundary_hops(
    subjects_dir: Path,
    template: str,
    hemisphere: str,
) -> np.ndarray:
    surface_path = subjects_dir / template / "surf" / f"{hemisphere}.white"
    annot_path = subjects_dir / template / "label" / f"{hemisphere}.aparc.annot"
    _, faces = fsio.read_geometry(str(surface_path))
    labels, _, _ = fsio.read_annot(str(annot_path))
    n_vertices = len(labels)
    adjacency = [[] for _ in range(n_vertices)]
    boundary = np.zeros(n_vertices, dtype=bool)
    for face in faces:
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            left = int(left)
            right = int(right)
            adjacency[left].append(right)
            adjacency[right].append(left)
            if labels[left] != labels[right]:
                boundary[left] = True
                boundary[right] = True

    hops = np.full(n_vertices, -1, dtype=np.int32)
    queue = deque(np.flatnonzero(boundary).tolist())
    hops[boundary] = 0
    while queue:
        vertex = queue.popleft()
        next_hop = hops[vertex] + 1
        for neighbor in adjacency[vertex]:
            if hops[neighbor] < 0:
                hops[neighbor] = next_hop
                queue.append(neighbor)
    return hops


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(x[valid], y[valid]).statistic)
    return float(stats.spearmanr(x[valid], y[valid]).statistic)


def regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0:
        return np.nan, np.nan
    result = stats.linregress(x[valid], y[valid])
    return float(result.slope), float(result.intercept)


def read_mincost(path: Path) -> dict[str, float]:
    values = np.asarray(path.read_text(encoding="utf-8").split(), dtype=float)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError(f"Invalid bbregister mincost file: {path}")
    return {
        "bbregister_mincost": float(values[0]),
        "mean_white_signal": float(values[1]),
        "mean_gray_signal": float(values[2]),
        "percent_contrast": float(values[3]),
    }


def process_hemisphere(
    row: pd.Series,
    hemisphere: str,
    args: argparse.Namespace,
    subject_output: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_id = str(row["subject_id"])
    column_path = Path(row[f"column_{hemisphere}"])
    profile_path = Path(row[f"profile_{hemisphere}"])
    fs_subject = Path(row["freesurfer_subject_dir"])

    with np.load(column_path, allow_pickle=True) as column_data:
        columns = {key: column_data[key] for key in column_data.files}
    with np.load(profile_path, allow_pickle=True) as profile_data:
        profiles = {key: profile_data[key] for key in profile_data.files}

    target_ids = np.asarray(columns["target_vertex_id"], dtype=np.int64)
    column_points = np.asarray(columns["column_points"], dtype=float)
    source_ids = np.asarray(columns["source_vertex_ids"], dtype=np.int64)
    weights = np.asarray(columns["barycentric_weights"], dtype=float)
    mapping_valid = np.asarray(columns["mapping_valid"], dtype=bool)
    analysis_mask = np.asarray(columns["in_analysis_mask"], dtype=bool)
    pial = np.asarray(columns["pial_endpoint"], dtype=float)
    white = np.asarray(columns["white_endpoint"], dtype=float)
    length = np.asarray(columns["column_length"], dtype=float)
    thickness = np.asarray(columns["cortical_thickness"], dtype=float)
    mapping_error = np.asarray(columns["mapping_error"], dtype=float)
    depth_fractions = np.asarray(columns["depth_fractions"], dtype=float)
    md = np.asarray(profiles["profiles"], dtype=float)
    in_bounds = np.asarray(profiles["coordinate_in_bounds_mask"], dtype=bool)
    sampled_finite = np.asarray(profiles["sampled_finite_mask"], dtype=bool)

    if scalar_from_mapping(columns, "depth_direction") != "pial_to_white":
        raise ValueError(
            f"{subject_id} {hemisphere}: depth direction is not pial_to_white"
        )
    expected_shapes = {
        "column_points": (len(target_ids), 21, 3),
        "profiles": (len(target_ids), 21),
        "in_bounds": (len(target_ids), 21),
        "weights": (len(target_ids), 3),
        "source_ids": (len(target_ids), 3),
    }
    actual_shapes = {
        "column_points": column_points.shape,
        "profiles": md.shape,
        "in_bounds": in_bounds.shape,
        "weights": weights.shape,
        "source_ids": source_ids.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(
                f"{subject_id} {hemisphere}: {name} shape {actual_shapes[name]}"
            )
    if not np.array_equal(target_ids, np.arange(len(target_ids))):
        raise ValueError(
            f"{subject_id} {hemisphere}: target IDs are not unique full indexing"
        )
    if not np.allclose(depth_fractions, DEPTHS, atol=1e-8, rtol=0.0):
        raise ValueError(f"{subject_id} {hemisphere}: noncanonical depth fractions")

    native_white, _ = fsio.read_geometry(
        str(fs_subject / "surf" / f"{hemisphere}.white")
    )
    native_pial, _ = fsio.read_geometry(str(fs_subject / "surf" / f"{hemisphere}.pial"))
    native_labels, _, native_names = fsio.read_annot(
        str(fs_subject / "label" / f"{hemisphere}.aparc.annot")
    )
    if not (len(native_white) == len(native_pial) == len(native_labels)):
        raise ValueError(
            f"{subject_id} {hemisphere}: native surface/label counts differ"
        )

    source_in_range = np.all(
        (source_ids >= 0) & (source_ids < len(native_white)),
        axis=1,
    )
    finite_weights = np.isfinite(weights).all(axis=1)
    weight_sum_error = np.abs(weights.sum(axis=1) - 1.0)
    out_of_range_weights = np.any(
        (weights < -1e-6) | (weights > 1.0 + 1e-6),
        axis=1,
    )
    endpoints_finite = np.isfinite(pial).all(axis=1) & np.isfinite(white).all(axis=1)
    zero_length = np.isfinite(length) & (length < args.zero_length_tolerance_mm)
    expected_points = (
        pial[:, None, :] * (1.0 - DEPTHS[None, :, None])
        + white[:, None, :] * DEPTHS[None, :, None]
    )
    coordinate_formula_error = np.nanmax(
        np.abs(column_points - expected_points),
        axis=(1, 2),
    )
    endpoint_error_pial = np.linalg.norm(column_points[:, 0] - pial, axis=1)
    endpoint_error_white = np.linalg.norm(column_points[:, -1] - white, axis=1)

    md_finite = np.isfinite(md)
    exact_zero = md_finite & (md == 0)
    near_zero = md_finite & (np.abs(md) <= args.near_zero_md_threshold)
    complete_in_bounds = in_bounds.all(axis=1)
    complete_finite = sampled_finite.all(axis=1) & md_finite.all(axis=1)
    complete_valid = (
        mapping_valid
        & endpoints_finite
        & ~zero_length
        & complete_in_bounds
        & complete_finite
        & ~exact_zero.any(axis=1)
    )

    triangle_labels = np.full(source_ids.shape, -1, dtype=np.int64)
    triangle_labels[source_in_range] = native_labels[source_ids[source_in_range]]
    normalized_native_names = np.asarray(
        [normalize_label(name) for name in native_names],
        dtype=str,
    )
    valid_native_label = triangle_labels >= 0
    for label_id, name in enumerate(normalized_native_names):
        if name in EXCLUDED_LABELS:
            valid_native_label &= triangle_labels != label_id
    all_labels_valid = valid_native_label.all(axis=1)
    n_unique_labels = np.full(len(target_ids), 0, dtype=np.int8)
    for index in np.flatnonzero(source_in_range):
        n_unique_labels[index] = len(set(triangle_labels[index].tolist()))
    mixed_label = all_labels_valid & (n_unique_labels > 1)
    noncortical_involved = source_in_range & ~all_labels_valid

    target_label_names = np.asarray(columns["template_label_name"]).astype(str)
    source_label_names = np.full(source_ids.shape, "invalid", dtype=object)
    valid_entries = triangle_labels >= 0
    source_label_names[valid_entries] = normalized_native_names[
        triangle_labels[valid_entries]
    ]
    normalized_target_names = np.asarray(
        [normalize_label(name) for name in target_label_names],
        dtype=str,
    )
    target_support = np.sum(
        weights * (source_label_names == normalized_target_names[:, None]),
        axis=1,
    )
    target_support[~finite_weights] = np.nan
    boundary_hops = template_boundary_hops(
        args.subjects_dir,
        args.template,
        hemisphere,
    )

    valid_geometry = mapping_valid & endpoints_finite & ~zero_length
    finite_length_thickness = (
        analysis_mask & valid_geometry & np.isfinite(length) & np.isfinite(thickness)
    )
    differences = length[finite_length_thickness] - thickness[finite_length_thickness]
    means = (length[finite_length_thickness] + thickness[finite_length_thickness]) / 2.0
    slope, intercept = regression(
        thickness[finite_length_thickness],
        length[finite_length_thickness],
    )

    full_denominator = len(target_ids)
    mask_denominator = int(analysis_mask.sum())
    summary: dict[str, object] = {
        "subject_id": subject_id,
        "hemisphere": hemisphere,
        "n_template_vertices": full_denominator,
        "n_analysis_mask_vertices": mask_denominator,
        "n_native_white_vertices": len(native_white),
        "n_native_pial_vertices": len(native_pial),
        "n_native_aparc_labels": len(native_labels),
        "native_index_correspondence_valid": (
            len(native_white) == len(native_pial) == len(native_labels)
        ),
        "n_triangle_lookup_success": int(
            (np.asarray(columns["source_face_id"]) >= 0).sum()
        ),
        "n_mapping_valid": int(mapping_valid.sum()),
        "mapping_valid_percentage": 100.0 * mapping_valid.mean(),
        "n_fallback": 0,
        "n_invalid_source_vertex_ids": int((mapping_valid & ~source_in_range).sum()),
        "weight_min": float(np.nanmin(weights)),
        "weight_max": float(np.nanmax(weights)),
        "n_nonfinite_weights": int((mapping_valid & ~finite_weights).sum()),
        "n_out_of_range_weights": int((mapping_valid & out_of_range_weights).sum()),
        "weight_sum_error_median": float(np.nanmedian(weight_sum_error[mapping_valid])),
        "weight_sum_error_max": float(np.nanmax(weight_sum_error[mapping_valid])),
        "mapping_error_median": float(np.nanmedian(mapping_error[mapping_valid])),
        "mapping_error_q25": float(np.nanquantile(mapping_error[mapping_valid], 0.25)),
        "mapping_error_q75": float(np.nanquantile(mapping_error[mapping_valid], 0.75)),
        "mapping_error_p95": float(np.nanquantile(mapping_error[mapping_valid], 0.95)),
        "mapping_error_max": float(np.nanmax(mapping_error[mapping_valid])),
        "n_finite_endpoints": int(endpoints_finite.sum()),
        "n_nonzero_length": int((endpoints_finite & ~zero_length).sum()),
        "n_complete_in_bounds": int(complete_in_bounds.sum()),
        "n_complete_finite_md": int(complete_finite.sum()),
        "n_complete_valid_profile": int(complete_valid.sum()),
        "complete_valid_profile_pct_full_mesh": 100.0
        * complete_valid.sum()
        / full_denominator,
        "complete_valid_profile_pct_analysis_mask": 100.0
        * (complete_valid & analysis_mask).sum()
        / mask_denominator,
        "zero_length_count_full_mesh": int(zero_length.sum()),
        "zero_length_pct_full_mesh": 100.0 * zero_length.mean(),
        "zero_length_count_analysis_mask": int((zero_length & analysis_mask).sum()),
        "zero_length_pct_analysis_mask": 100.0
        * (zero_length & analysis_mask).sum()
        / mask_denominator,
        "max_column_formula_error_mm": float(np.nanmax(coordinate_formula_error)),
        "max_pial_endpoint_error_mm": float(np.nanmax(endpoint_error_pial)),
        "max_white_endpoint_error_mm": float(np.nanmax(endpoint_error_white)),
        "point_oob_percentage": 100.0 * (~in_bounds).mean(),
        "column_any_oob_percentage": 100.0 * (~complete_in_bounds).mean(),
        "complete_in_bounds_percentage": 100.0 * complete_in_bounds.mean(),
        "point_nonfinite_md_percentage": 100.0 * (~md_finite).mean(),
        "point_zero_md_percentage": 100.0 * exact_zero.mean(),
        "point_near_zero_md_percentage": 100.0 * near_zero.mean(),
        "n_all_three_labels_valid": int(all_labels_valid.sum()),
        "valid_label_triangle_percentage": 100.0 * all_labels_valid.mean(),
        "mixed_label_triangle_percentage": 100.0 * mixed_label.mean(),
        "noncortical_triangle_percentage": 100.0 * noncortical_involved.mean(),
        "target_label_support_median": float(np.nanmedian(target_support)),
        "mixed_boundary_hops_median": float(np.nanmedian(boundary_hops[mixed_label]))
        if mixed_label.any()
        else np.nan,
        "mixed_within_one_boundary_hop_percentage": 100.0
        * np.mean(boundary_hops[mixed_label] <= 1)
        if mixed_label.any()
        else np.nan,
        "length_thickness_pearson_r": safe_correlation(length, thickness, "pearson"),
        "length_thickness_spearman_rho": safe_correlation(
            length, thickness, "spearman"
        ),
        "length_thickness_mae_mm": float(np.mean(np.abs(differences))),
        "length_thickness_rmse_mm": float(np.sqrt(np.mean(differences**2))),
        "length_minus_thickness_median_mm": float(np.median(differences)),
        "length_thickness_relative_difference_median": float(
            np.median(differences / means)
        ),
        "length_thickness_regression_slope": slope,
        "length_thickness_regression_intercept": intercept,
    }

    depth_rows = []
    for depth_index, depth in enumerate(DEPTHS):
        depth_rows.append(
            {
                "subject_id": subject_id,
                "hemisphere": hemisphere,
                "depth_index": depth_index,
                "depth_fraction_pial_to_white": depth,
                "point_oob_percentage": 100.0 * (~in_bounds[:, depth_index]).mean(),
                "point_nonfinite_md_percentage": 100.0
                * (~md_finite[:, depth_index]).mean(),
                "point_zero_md_percentage": 100.0 * exact_zero[:, depth_index].mean(),
                "point_near_zero_md_percentage": 100.0
                * near_zero[:, depth_index].mean(),
            }
        )

    label_table = pd.DataFrame(
        {
            "subject_id": subject_id,
            "hemisphere": hemisphere,
            "target_vertex_id": target_ids,
            "target_label_name": target_label_names,
            "in_analysis_mask": analysis_mask,
            "mapping_valid": mapping_valid,
            "zero_length": zero_length,
            "n_unique_triangle_labels": n_unique_labels,
            "all_triangle_labels_valid": all_labels_valid,
            "mixed_label": mixed_label,
            "noncortical_label_involved": noncortical_involved,
            "target_label_support": target_support,
            "distance_to_dk_boundary_hops": boundary_hops,
            "complete_valid_profile": complete_valid,
        }
    )
    mapping_table = pd.DataFrame(
        {
            "subject_id": subject_id,
            "hemisphere": hemisphere,
            "target_vertex_id": target_ids,
            "mapping_valid": mapping_valid,
            "mapping_error": mapping_error,
            "weight_sum_error": weight_sum_error,
            "weights_out_of_range": out_of_range_weights,
            "source_vertex_ids_valid": source_in_range,
            "coordinate_formula_error_mm": coordinate_formula_error,
            "column_length_mm": length,
            "freesurfer_thickness_mm": thickness,
        }
    )
    return summary, pd.DataFrame(depth_rows), label_table, mapping_table


def scalar_from_mapping(mapping: dict[str, np.ndarray], key: str) -> str:
    value = np.asarray(mapping[key]).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    subject_output = args.output_root / "subjects" / subject_id
    subject_output.mkdir(parents=True, exist_ok=True)
    failure_path = subject_output / "failure.json"
    completion_path = subject_output / "completion.json"
    started = time.time()

    try:
        if not bool(row["input_complete"]):
            raise FileNotFoundError(
                f"Incomplete manifest inputs: {row['missing_inputs']}"
            )
        summaries = []
        depth_tables = []
        label_tables = []
        mapping_tables = []
        for hemisphere in HEMISPHERES:
            logging.info("[%s] Starting %s QC", subject_id, hemisphere)
            summary, depth, labels, mapping = process_hemisphere(
                row,
                hemisphere,
                args,
                subject_output,
            )
            summaries.append(summary)
            depth_tables.append(depth)
            label_tables.append(labels)
            mapping_tables.append(mapping)
            logging.info("[%s] Completed %s QC", subject_id, hemisphere)

        registration = {
            "subject_id": subject_id,
            "transform_path": row["transform_path"],
            "mincost_path": row["mincost_path"],
            "moving_image_path": row["md_image"],
            "freesurfer_subject_dir": row["freesurfer_subject_dir"],
            **read_mincost(Path(row["mincost_path"])),
        }
        pd.DataFrame(summaries).to_csv(
            subject_output / "option3_mapping_qc_subject_hemi.csv",
            index=False,
        )
        pd.concat(depth_tables, ignore_index=True).to_csv(
            subject_output / "option3_invalid_md_by_depth.csv",
            index=False,
        )
        pd.concat(label_tables, ignore_index=True).to_parquet(
            subject_output / "option3_label_qc_vertex.parquet",
            index=False,
        )
        label_table = pd.concat(label_tables, ignore_index=True)
        (
            label_table.groupby(
                ["subject_id", "hemisphere", "target_label_name"],
                observed=True,
            )
            .agg(
                n_vertices=("target_vertex_id", "size"),
                n_zero_length=("zero_length", "sum"),
                n_mixed_label=("mixed_label", "sum"),
                n_noncortical_triangle=("noncortical_label_involved", "sum"),
                median_target_label_support=("target_label_support", "median"),
            )
            .reset_index()
            .to_csv(subject_output / "option3_label_qc_region.csv", index=False)
        )
        pd.concat(mapping_tables, ignore_index=True).to_parquet(
            subject_output / "option3_mapping_error_vertex.parquet",
            index=False,
        )
        pd.DataFrame([registration]).to_csv(
            subject_output / "registration_cost.csv",
            index=False,
        )
        runtime = time.time() - started
        completion_path.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "complete",
                    "runtime_seconds": runtime,
                    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        failure_path.unlink(missing_ok=True)
        logging.info("[%s] Complete in %.1f seconds", subject_id, runtime)
    except Exception as error:
        failure_path.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        completion_path.unlink(missing_ok=True)
        logging.exception("[%s] QC failed", subject_id)
        raise


if __name__ == "__main__":
    main()
