"""Reusable calculations for the D0086 fsaverage O3 QC notebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_processing.transform_trajectories import (
    load_dwi_geometry,
    load_transform_matrix,
)


def _as_scalar(value):
    """Convert a zero-dimensional NPZ value to a Python scalar."""

    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for constant or short inputs."""

    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(pearsonr(left, right).statistic)


def _write_figure(figure, output_dir: Path, name: str) -> Path:
    """Save a 300-dpi PNG and close the figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _load_annotation(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    """Load annotation codes and a code-to-name map."""

    labels, color_table, names = nib.freesurfer.io.read_annot(str(path))
    del color_table
    names_by_code = {}
    for label_index, name in enumerate(names):
        label_name = name.decode() if isinstance(name, bytes) else str(name)
        names_by_code[label_index] = label_name
    return labels.astype(np.int64), names_by_code


def _load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a FreeSurfer surface geometry."""

    coordinates, faces = nib.freesurfer.io.read_geometry(str(path))
    return coordinates.astype(np.float64), faces.astype(np.int64)


def _triangle_vertex_areas(
    coordinates: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Assign one third of each triangle area to each vertex."""

    triangles = coordinates[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    triangle_area = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.zeros(coordinates.shape[0], dtype=float)
    np.add.at(areas, faces.ravel(), np.repeat(triangle_area / 3.0, 3))
    return areas


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    """Return sorted unique edges from triangular faces."""

    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _profile_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Calculate row-wise profile correlations without Python vertex loops."""

    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
        right_centered,
        axis=1,
    )
    return np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )


def _status(value: float, threshold: float, higher_is_better: bool = True) -> str:
    """Assign PASS or FAIL for a numeric QC check."""

    if not np.isfinite(value):
        return "FAIL"
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def _load_bundle(config: dict, hemisphere: str) -> dict:
    """Load one actual O3 fsaverage NPZ and its supporting files."""

    subject = config["subject_id"]
    output_root = Path(config["o3_output_root"])
    o3_path = (
        output_root
        / subject
        / "fsaverage"
        / "o3_coordinate"
        / f"{subject}_{hemisphere}_profiles.npz"
    )
    qc_path = o3_path.with_suffix(".qc.json")
    native_root = Path(config["native_subjects_root"]) / subject
    fsaverage_root = Path(config["freesurfer_subjects_dir"]) / "fsaverage"
    native_surface_root = native_root / "surf"
    fsaverage_surface_root = fsaverage_root / "surf"
    native_label_path = native_root / "label" / f"{hemisphere}.aparc.annot"
    template_label_path = fsaverage_root / "label" / f"{hemisphere}.aparc.annot"
    thickness_path = native_surface_root / f"{hemisphere}.thickness"
    white_path = fsaverage_surface_root / f"{hemisphere}.white"
    md_path = Path(config["md_volume_path"])
    registration_path = Path(config["registration_path"])

    with np.load(o3_path, allow_pickle=True) as data:
        bundle = {key: data[key] for key in data.files}
    bundle.update(
        {
            "o3_path": o3_path,
            "qc_path": qc_path,
            "native_root": native_root,
            "native_surface_root": native_surface_root,
            "fsaverage_surface_root": fsaverage_surface_root,
            "native_label_path": native_label_path,
            "template_label_path": template_label_path,
            "thickness_path": thickness_path,
            "white_path": white_path,
            "md_path": md_path,
            "registration_path": registration_path,
        }
    )
    return bundle


def _validate_bundle(bundle: dict, config: dict, hemisphere: str) -> dict:
    """Validate dimensions, IDs, mapping, and field availability."""

    expected_vertices = config["expected_fsaverage_vertices"]
    depths = np.asarray(bundle["depths"], dtype=float)
    target_ids = np.asarray(bundle["target_vertex_id"], dtype=np.int64)
    profiles = np.asarray(bundle["md_profiles"], dtype=float)
    required = {
        "target_vertex_id",
        "depths",
        "md_profiles",
        "native_coordinates",
        "dwi_voxel_coordinates",
        "white_anchor",
        "pial_anchor",
        "source_face_id",
        "source_vertex_ids",
        "barycentric_weights",
        "mapping_valid",
        "coordinate_in_bounds_mask",
        "finite_mask",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise KeyError(f"Missing essential O3 fields for {hemisphere}: {missing}")
    if profiles.shape != (expected_vertices, 21):
        raise ValueError(
            f"Expected ({expected_vertices}, 21) profiles for {hemisphere}; "
            f"found {profiles.shape}"
        )
    if depths.shape != (21,) or not np.allclose(depths, np.linspace(0.0, 1.0, 21)):
        raise ValueError(f"Depth array is not the expected 21-point grid: {depths}")
    if target_ids.shape != (expected_vertices,):
        raise ValueError(f"Unexpected target ID shape: {target_ids.shape}")
    unique_ids = np.unique(target_ids)
    missing_ids = np.setdiff1d(np.arange(expected_vertices), unique_ids)
    duplicate_ids = target_ids[np.unique(target_ids, return_counts=True)[1] > 1]
    if missing_ids.size or duplicate_ids.size:
        raise ValueError(
            f"Template IDs are not unique and complete for {hemisphere}: "
            f"missing={missing_ids.size}, duplicates={duplicate_ids.size}"
        )
    dimensions = {
        "expected_vertices": expected_vertices,
        "loaded_vertices": int(target_ids.size),
        "unique_vertices": int(unique_ids.size),
        "missing_vertices": int(missing_ids.size),
        "duplicate_vertices": int(duplicate_ids.size),
        "depth_samples": int(depths.size),
    }
    return {
        "dimensions": dimensions,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
    }


def _mapping_qc(bundle: dict, config: dict, hemisphere: str) -> dict:
    """Calculate barycentric mapping QC and save invalid vertex lists."""

    native_sphere, native_faces = _load_surface(
        bundle["native_surface_root"] / f"{hemisphere}.sphere.reg"
    )
    source_face_id = np.asarray(bundle["source_face_id"], dtype=np.int64)
    source_vertex_ids = np.asarray(bundle["source_vertex_ids"], dtype=np.int64)
    weights = np.asarray(bundle["barycentric_weights"], dtype=float)
    mapping_valid = np.asarray(bundle["mapping_valid"], dtype=bool)
    target_ids = np.asarray(bundle["target_vertex_id"], dtype=np.int64)
    valid_face = (source_face_id >= 0) & (source_face_id < len(native_faces))
    valid_vertices = np.all(
        (source_vertex_ids >= 0) & (source_vertex_ids < len(native_sphere)),
        axis=1,
    )
    finite_weights = np.isfinite(weights).all(axis=1)
    weight_sum_error = np.abs(weights.sum(axis=1) - 1.0)
    low_weight = np.any(weights < -config["weight_tol"], axis=1)
    high_weight = np.any(weights > 1.0 + config["weight_tol"], axis=1)
    face_match = np.zeros(len(target_ids), dtype=bool)
    valid_face_rows = valid_face & valid_vertices
    face_match[valid_face_rows] = np.all(
        native_faces[source_face_id[valid_face_rows]]
        == source_vertex_ids[valid_face_rows],
        axis=1,
    )
    mapping_success = mapping_valid & valid_face & valid_vertices & finite_weights
    invalid_ids = target_ids[~mapping_success]
    qc = {
        "hemisphere": hemisphere,
        "mapping_success_pct": float(mapping_valid.mean() * 100),
        "invalid_face_count": int((~valid_face).sum()),
        "invalid_native_vertex_count": int((~valid_vertices).sum()),
        "face_vertex_mismatch_count": int((valid_face_rows & ~face_match).sum()),
        "nonfinite_weight_count": int((~finite_weights).sum()),
        "max_abs_weight_sum_error": float(np.nanmax(weight_sum_error)),
        "weight_sum_error_gt_tol_pct": float(
            (weight_sum_error > config["weight_tol"]).mean() * 100
        ),
        "weights_below_tolerance_pct": float(low_weight.mean() * 100),
        "weights_above_tolerance_pct": float(high_weight.mean() * 100),
        "max_barycentric_weight_median": float(np.nanmedian(weights.max(axis=1))),
        "max_barycentric_weight_p99": float(np.nanpercentile(weights.max(axis=1), 99)),
        "max_barycentric_weight_max": float(np.nanmax(weights.max(axis=1))),
        "mapping_valid_count": int(mapping_success.sum()),
        "mapping_error_median": float(
            np.nanmedian(np.asarray(bundle["mapping_error"])[mapping_success])
        ),
        "mapping_error_max": float(
            np.nanmax(np.asarray(bundle["mapping_error"])[mapping_success])
        ),
        "target_direction_error_median": float(
            np.nanmedian(np.asarray(bundle["target_direction_error"])[mapping_success])
        ),
        "fallback_count": int(np.asarray(bundle["fallback_used"], dtype=bool).sum()),
    }
    invalid_table = pd.DataFrame({"target_vertex_id": invalid_ids})
    return {
        "qc": qc,
        "mapping_success": mapping_success,
        "weight_sum_error": weight_sum_error,
        "invalid_table": invalid_table,
        "native_faces": native_faces,
    }


def _geometry_qc(
    bundle: dict,
    mapping: dict,
    config: dict,
    hemisphere: str,
) -> dict:
    """Check straight-line column geometry and thickness agreement."""

    depths = np.asarray(bundle["depths"], dtype=float)
    white_anchor = np.asarray(bundle["white_anchor"], dtype=float)
    pial_anchor = np.asarray(bundle["pial_anchor"], dtype=float)
    saved = np.asarray(bundle["native_coordinates"], dtype=float)
    valid = mapping["mapping_success"] & np.isfinite(saved).all(axis=(1, 2))
    expected = (1.0 - depths)[None, :, None] * white_anchor[:, None, :] + depths[
        None, :, None
    ] * pial_anchor[:, None, :]
    reconstruction_error = np.linalg.norm(saved - expected, axis=2)
    line_length = np.linalg.norm(pial_anchor - white_anchor, axis=1)
    step_lengths = np.linalg.norm(np.diff(saved, axis=1), axis=2)
    expected_step = line_length[:, None] / (len(depths) - 1)
    step_error = np.abs(step_lengths - expected_step)
    zero_length = line_length <= config["coord_tol_mm"]
    finite_geometry = np.isfinite(expected).all(axis=(1, 2))
    valid_lengths = line_length[valid & np.isfinite(line_length)]
    length_median = float(np.median(valid_lengths))
    length_mad = float(np.median(np.abs(valid_lengths - length_median)))
    robust_cutoff = length_median + 5.0 * 1.4826 * length_mad
    length_outlier = valid & (line_length > robust_cutoff)

    thickness, _ = (
        nib.freesurfer.io.read_morph_data(str(bundle["thickness_path"])),
        None,
    )
    source_ids = np.asarray(bundle["source_vertex_ids"], dtype=np.int64)
    weights = np.asarray(bundle["barycentric_weights"], dtype=float)
    thickness_anchor = np.full(line_length.shape, np.nan)
    if thickness.size > source_ids.max():
        thickness_anchor[valid] = np.einsum(
            "ij,ij->i",
            weights[valid],
            thickness[source_ids[valid]],
        )
    thickness_keep = np.isfinite(thickness_anchor) & np.isfinite(line_length)
    thickness_corr = _safe_corr(
        line_length[thickness_keep], thickness_anchor[thickness_keep]
    )
    qc = {
        "coordinate_reconstruction_max_mm": float(np.nanmax(reconstruction_error)),
        "coordinate_reconstruction_median_mm": float(
            np.nanmedian(reconstruction_error)
        ),
        "white_endpoint_max_mm": float(
            np.nanmax(np.linalg.norm(saved[:, 0] - white_anchor, axis=1))
        ),
        "pial_endpoint_max_mm": float(
            np.nanmax(np.linalg.norm(saved[:, -1] - pial_anchor, axis=1))
        ),
        "finite_geometry_pct": float(finite_geometry.mean() * 100),
        "zero_length_column_count": int(zero_length.sum()),
        "step_error_max_mm": float(np.nanmax(step_error)),
        "step_error_median_mm": float(np.nanmedian(step_error)),
        "column_length_median_mm": length_median,
        "column_length_iqr_mm": float(
            np.percentile(valid_lengths, 75) - np.percentile(valid_lengths, 25)
        ),
        "column_length_min_mm": float(np.min(valid_lengths)),
        "column_length_max_mm": float(np.max(valid_lengths)),
        "column_length_outlier_count": int(length_outlier.sum()),
        "freesurfer_thickness_correlation": thickness_corr,
        "line_minus_thickness_median_mm": float(
            np.nanmedian(line_length[thickness_keep] - thickness_anchor[thickness_keep])
        ),
    }
    outlier_table = pd.DataFrame(
        {
            "target_vertex_id": bundle["target_vertex_id"][length_outlier],
            "column_length_mm": line_length[length_outlier],
            "white_x": white_anchor[length_outlier, 0],
            "white_y": white_anchor[length_outlier, 1],
            "white_z": white_anchor[length_outlier, 2],
            "pial_x": pial_anchor[length_outlier, 0],
            "pial_y": pial_anchor[length_outlier, 1],
            "pial_z": pial_anchor[length_outlier, 2],
        }
    ).sort_values("column_length_mm", ascending=False)
    zero_table = pd.DataFrame(
        {
            "target_vertex_id": bundle["target_vertex_id"][zero_length],
            "column_length_mm": line_length[zero_length],
        }
    )
    return {
        "qc": qc,
        "expected": expected,
        "line_length": line_length,
        "step_error": step_error,
        "length_outlier": length_outlier,
        "zero_table": zero_table,
        "thickness_anchor": thickness_anchor,
        "outlier_table": outlier_table,
    }


def _coordinate_qc(bundle: dict, config: dict, geometry: dict) -> dict:
    """Check DWI coordinates, bounds, and deterministic transform round trip."""

    md_image = nib.load(str(bundle["md_path"]))
    shape = np.asarray(md_image.shape[:3], dtype=float)
    dwi_coordinates = np.asarray(bundle["dwi_voxel_coordinates"], dtype=float)
    native_coordinates = np.asarray(bundle["native_coordinates"], dtype=float)
    finite_t1 = np.isfinite(native_coordinates).all(axis=2)
    finite_dwi = np.isfinite(dwi_coordinates).all(axis=2)
    recomputed_in_bounds = finite_dwi & np.all(
        (dwi_coordinates >= 0.0) & (dwi_coordinates <= shape[None, None, :] - 1),
        axis=2,
    )
    saved_in_bounds = np.asarray(bundle["coordinate_in_bounds_mask"], dtype=bool)
    transform = load_transform_matrix(bundle["registration_path"])
    _, _, t_mov = load_dwi_geometry(bundle["md_path"])
    valid_points = np.flatnonzero(finite_t1.ravel() & finite_dwi.ravel())
    sample_size = min(config["round_trip_sample_size"], valid_points.size)
    rng = np.random.default_rng(config["random_seed"])
    selected = rng.choice(valid_points, size=sample_size, replace=False)
    native_flat = native_coordinates.reshape(-1, 3)
    dwi_flat = dwi_coordinates.reshape(-1, 3)
    dwi_one_based = np.column_stack((dwi_flat[selected] + 1.0, np.ones(sample_size)))
    reconstructed_t1 = (np.linalg.inv(transform) @ t_mov @ dwi_one_based.T).T[:, :3]
    round_trip_error = np.linalg.norm(
        reconstructed_t1 - native_flat[selected],
        axis=1,
    )
    depth_rows = []
    for depth_index, depth in enumerate(np.asarray(bundle["depths"], dtype=float)):
        depth_rows.append(
            {
                "depth": float(depth),
                "finite_t1_pct": float(finite_t1[:, depth_index].mean() * 100),
                "finite_dwi_pct": float(finite_dwi[:, depth_index].mean() * 100),
                "in_bounds_pct": float(
                    recomputed_in_bounds[:, depth_index].mean() * 100
                ),
                "out_of_bounds_pct": float(
                    (~recomputed_in_bounds[:, depth_index]).mean() * 100
                ),
            }
        )
    qc = {
        "finite_t1_pct": float(finite_t1.mean() * 100),
        "finite_dwi_pct": float(finite_dwi.mean() * 100),
        "in_bounds_pct": float(recomputed_in_bounds.mean() * 100),
        "saved_vs_recomputed_bounds_mismatch": int(
            np.sum(saved_in_bounds != recomputed_in_bounds)
        ),
        "round_trip_sample_size": int(sample_size),
        "round_trip_median_mm": float(np.median(round_trip_error)),
        "round_trip_95th_mm": float(np.percentile(round_trip_error, 95)),
        "round_trip_max_mm": float(np.max(round_trip_error)),
        "coordinate_system": "FreeSurfer surface RAS to 0-based DWI voxel coordinates",
        "md_volume_shape": [int(value) for value in shape],
    }
    affected_ids = np.unique(
        bundle["target_vertex_id"][np.any(~recomputed_in_bounds, axis=1)]
    )
    return {
        "qc": qc,
        "depth_table": pd.DataFrame(depth_rows),
        "recomputed_in_bounds": recomputed_in_bounds,
        "affected_ids": affected_ids,
        "round_trip_error": round_trip_error,
    }


def _md_qc(bundle: dict, config: dict) -> dict:
    """Calculate depth-wise and profile-wise MD QC."""

    profiles = np.asarray(bundle["md_profiles"], dtype=float)
    finite = np.isfinite(profiles)
    zeros = profiles == 0.0
    complete = finite.all(axis=1)
    any_zero = zeros.any(axis=1)
    any_nan = (~finite).any(axis=1)
    completely_zero = zeros.all(axis=1)
    constant = complete & (np.ptp(profiles, axis=1) == 0.0)
    adjacent_change = np.full(profiles.shape[0], np.nan)
    adjacent_change[complete] = np.max(
        np.abs(np.diff(profiles[complete], axis=1)),
        axis=1,
    )
    depth_rows = []
    for depth_index, depth in enumerate(np.asarray(bundle["depths"], dtype=float)):
        values = profiles[:, depth_index]
        valid_values = values[np.isfinite(values)]
        depth_rows.append(
            {
                "depth": float(depth),
                "finite_pct": float(np.isfinite(values).mean() * 100),
                "nan_pct": float(np.isnan(values).mean() * 100),
                "zero_pct": float((values == 0).mean() * 100),
                "median": float(np.median(valid_values)),
                "q25": float(np.percentile(valid_values, 25)),
                "q75": float(np.percentile(valid_values, 75)),
                "iqr": float(
                    np.percentile(valid_values, 75) - np.percentile(valid_values, 25)
                ),
                "p01": float(np.percentile(valid_values, 1)),
                "p99": float(np.percentile(valid_values, 99)),
            }
        )
    outlier_ids = np.argsort(np.nan_to_num(adjacent_change, nan=-np.inf))[::-1][:100]
    outlier_table = pd.DataFrame(
        {
            "target_vertex_id": bundle["target_vertex_id"][outlier_ids],
            "max_absolute_adjacent_depth_change": adjacent_change[outlier_ids],
        }
    )
    return {
        "qc": {
            "complete_finite_profile_pct": float(complete.mean() * 100),
            "profiles_with_zero_pct": float(any_zero.mean() * 100),
            "profiles_with_nan_pct": float(any_nan.mean() * 100),
            "completely_zero_profile_pct": float(completely_zero.mean() * 100),
            "constant_profile_pct": float(constant.mean() * 100),
            "max_adjacent_change_p99": float(np.nanpercentile(adjacent_change, 99)),
            "max_adjacent_change_max": float(np.nanmax(adjacent_change)),
        },
        "depth_table": pd.DataFrame(depth_rows),
        "complete": complete,
        "adjacent_change": adjacent_change,
        "outlier_table": outlier_table,
    }


def _spatial_qc(
    bundle: dict,
    mapping: dict,
    geometry: dict,
    hemisphere: str,
) -> dict:
    """Calculate vectorized neighboring-vertex continuity metrics."""

    _, faces = _load_surface(bundle["white_path"])
    edges = _unique_edges(faces)
    profiles = np.asarray(bundle["md_profiles"], dtype=float)
    complete = np.asarray(mapping["mapping_success"], dtype=bool) & np.isfinite(
        profiles
    ).all(axis=1)
    keep = complete[edges[:, 0]] & complete[edges[:, 1]]
    edges = edges[keep]
    white = np.asarray(bundle["white_anchor"], dtype=float)
    pial = np.asarray(bundle["pial_anchor"], dtype=float)
    mid = 0.5 * (white + pial)
    directions = pial - white
    norms = np.linalg.norm(directions, axis=1)
    unit = directions / np.maximum(norms[:, None], 1e-12)
    left, right = edges[:, 0], edges[:, 1]
    profile_left = profiles[left]
    profile_right = profiles[right]
    correlations = _profile_correlations(profile_left, profile_right)
    profile_mae = np.mean(np.abs(profile_left - profile_right), axis=1)
    angle = np.degrees(
        np.arccos(np.clip(np.sum(unit[left] * unit[right], axis=1), -1.0, 1.0))
    )
    table = pd.DataFrame(
        {
            "vertex_i": left,
            "vertex_j": right,
            "white_distance_mm": np.linalg.norm(white[left] - white[right], axis=1),
            "pial_distance_mm": np.linalg.norm(pial[left] - pial[right], axis=1),
            "mid_depth_distance_mm": np.linalg.norm(mid[left] - mid[right], axis=1),
            "column_direction_angle_deg": angle,
            "profile_correlation": correlations,
            "profile_mae": profile_mae,
        }
    )
    outlier_table = table.sort_values(
        ["profile_mae", "column_direction_angle_deg"],
        ascending=False,
    ).head(100)
    return {
        "summary": {
            "n_unique_edges": int(len(edges)),
            "mid_depth_distance_median_mm": float(table.mid_depth_distance_mm.median()),
            "direction_angle_median_deg": float(
                table.column_direction_angle_deg.median()
            ),
            "profile_correlation_median": float(table.profile_correlation.median()),
            "profile_mae_median": float(table.profile_mae.median()),
        },
        "table": table,
        "outlier_table": outlier_table,
    }


def _region_qc(bundle: dict, mapping: dict, hemisphere: str) -> dict:
    """Compare template annotation and the three reconstructed native labels."""

    template_labels, template_names = _load_annotation(bundle["template_label_path"])
    native_labels, native_names = _load_annotation(bundle["native_label_path"])
    source_ids = np.asarray(bundle["source_vertex_ids"], dtype=np.int64)
    valid = mapping["mapping_success"]
    source_label_values = np.full(source_ids.shape, -1, dtype=np.int64)
    source_label_values[valid] = native_labels[source_ids[valid]]
    mixed = valid & (
        (source_label_values[:, 0] != source_label_values[:, 1])
        | (source_label_values[:, 1] != source_label_values[:, 2])
    )
    complete_profile = valid & np.isfinite(bundle["md_profiles"]).all(axis=1)
    template_codes = template_labels[bundle["target_vertex_id"]]
    table = pd.DataFrame(
        {
            "target_vertex_id": bundle["target_vertex_id"],
            "template_region_code": template_codes,
            "template_region_name": [
                template_names.get(int(code), "unknown") for code in template_codes
            ],
            "native_label_1": source_label_values[:, 0],
            "native_label_2": source_label_values[:, 1],
            "native_label_3": source_label_values[:, 2],
            "mixed_label_boundary": mixed,
            "mapping_valid": valid,
            "complete_profile": complete_profile,
        }
    )
    rows = []
    for boundary, group in table.groupby("mixed_label_boundary", dropna=False):
        rows.append(
            {
                "hemisphere": hemisphere,
                "boundary": bool(boundary),
                "n_vertices": int(len(group)),
                "complete_profile_pct": float(group.complete_profile.mean() * 100),
            }
        )
    return {
        "qc": {
            "mixed_label_triangle_pct": float(mixed.mean() * 100),
            "native_label_names": len(native_names),
            "template_label_names": len(template_names),
        },
        "table": table,
        "summary_table": pd.DataFrame(rows),
    }


def _make_figures(
    bundle: dict,
    mapping: dict,
    geometry: dict,
    coordinates: dict,
    md: dict,
    spatial: dict,
    region: dict,
    figure_root: Path,
    hemisphere: str,
) -> list[Path]:
    """Create the requested QC figures for one hemisphere."""

    paths = []
    mapping_figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(mapping["weight_sum_error"], bins=60, color="#0072B2")
    axes[0].set_xlabel("Absolute barycentric weight-sum error")
    axes[0].set_ylabel("Target vertices")
    axes[0].set_title(f"{hemisphere}: weight-sum error")
    axes[1].hist(
        np.asarray(bundle["barycentric_weights"]).max(axis=1),
        bins=60,
        color="#009E73",
    )
    axes[1].set_xlabel("Maximum barycentric weight")
    axes[1].set_ylabel("Target vertices")
    axes[1].set_title(f"{hemisphere}: maximum weight")
    paths.append(
        _write_figure(mapping_figure, figure_root, f"{hemisphere}_01_mapping_qc")
    )

    geometry_figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(geometry["line_length"], bins=70, color="#56B4E9")
    axes[0].set_xlabel("O3 white-to-pial line length (mm)")
    axes[0].set_ylabel("Target vertices")
    axes[0].set_title(f"{hemisphere}: column length")
    axes[1].scatter(
        geometry["line_length"],
        geometry["thickness_anchor"],
        s=1,
        alpha=0.12,
        color="#D55E00",
        linewidths=0,
    )
    axes[1].set_xlabel("O3 line length (mm)")
    axes[1].set_ylabel("Native FreeSurfer thickness (mm)")
    axes[1].set_title(f"{hemisphere}: geometry vs thickness")
    paths.append(
        _write_figure(geometry_figure, figure_root, f"{hemisphere}_02_column_geometry")
    )

    spacing_figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(geometry["step_error"].ravel(), bins=70, color="#CC79A7")
    axes[0].set_xlabel("Absolute depth-step spacing error (mm)")
    axes[0].set_ylabel("Depth intervals")
    axes[0].set_title(f"{hemisphere}: depth spacing")
    axes[1].hist(md["adjacent_change"], bins=70, color="#E69F00")
    axes[1].set_xlabel("Maximum adjacent-depth MD change")
    axes[1].set_ylabel("Target vertices")
    axes[1].set_title(f"{hemisphere}: profile change")
    paths.append(
        _write_figure(
            spacing_figure, figure_root, f"{hemisphere}_03_spacing_and_profile_change"
        )
    )

    coordinate_figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    depth_table = coordinates["depth_table"]
    axes[0].plot(depth_table.depth, depth_table.in_bounds_pct, color="#0072B2")
    axes[0].set_ylim(0, 100.5)
    axes[0].set_xlabel("Cortical depth")
    axes[0].set_ylabel("In-bounds coordinates (%)")
    axes[0].set_title(f"{hemisphere}: DWI coordinate validity")
    axes[1].hist(coordinates["round_trip_error"], bins=60, color="#CC79A7")
    axes[1].set_xlabel("T1 → DWI → T1 error (mm)")
    axes[1].set_ylabel("Sampled points")
    axes[1].set_title(f"{hemisphere}: transform round trip")
    paths.append(
        _write_figure(coordinate_figure, figure_root, f"{hemisphere}_04_coordinate_qc")
    )

    md_figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    md_depth = md["depth_table"]
    axes[0].plot(
        md_depth["depth"],
        md_depth["median"],
        color="#0072B2",
        label="Median",
    )
    axes[0].fill_between(
        md_depth["depth"],
        md_depth["q25"],
        md_depth["q75"],
        color="#0072B2",
        alpha=0.18,
        label="IQR approximation",
    )
    axes[0].set_xlabel("Cortical depth")
    axes[0].set_ylabel("MD")
    axes[0].set_title(f"{hemisphere}: MD profile summary")
    axes[0].legend(frameon=False, fontsize=8)
    valid_ids = np.flatnonzero(md["complete"])
    sample_ids = valid_ids[
        np.linspace(0, len(valid_ids) - 1, min(100, len(valid_ids))).astype(int)
    ]
    axes[1].plot(
        bundle["depths"],
        bundle["md_profiles"][sample_ids].T,
        color="#56B4E9",
        alpha=0.15,
        linewidth=0.5,
    )
    axes[1].set_xlabel("Cortical depth")
    axes[1].set_ylabel("MD")
    axes[1].set_title(f"{hemisphere}: deterministic profile sample")
    paths.append(
        _write_figure(md_figure, figure_root, f"{hemisphere}_05_md_profile_qc")
    )

    spatial_figure, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    spatial_table = spatial["table"]
    axes[0].hist(spatial_table.mid_depth_distance_mm, bins=70, color="#0072B2")
    axes[0].set_xlabel("Neighbor mid-depth distance (mm)")
    axes[0].set_ylabel("Edges")
    axes[1].hist(spatial_table.column_direction_angle_deg, bins=70, color="#009E73")
    axes[1].set_xlabel("Column direction angle (degrees)")
    axes[1].set_ylabel("Edges")
    axes[2].hist(spatial_table.profile_correlation.dropna(), bins=70, color="#D55E00")
    axes[2].set_xlabel("Neighbor profile correlation")
    axes[2].set_ylabel("Edges")
    spatial_figure.suptitle(f"{hemisphere}: spatial continuity")
    paths.append(
        _write_figure(
            spatial_figure, figure_root, f"{hemisphere}_06_spatial_continuity"
        )
    )

    return paths


def _save_overlays(
    bundle: dict,
    mapping: dict,
    geometry: dict,
    coordinates: dict,
    md: dict,
    region: dict,
    overlay_root: Path,
    hemisphere: str,
) -> None:
    """Save compact FreeSurfer morph overlays for visual QC."""

    overlay_root.mkdir(parents=True, exist_ok=True)
    target_count = len(bundle["target_vertex_id"])
    values = {
        "column_length_mm": geometry["line_length"],
        "invalid_profile": (~md["complete"]).astype(float),
        "out_of_bounds": np.any(~coordinates["recomputed_in_bounds"], axis=1).astype(
            float
        ),
        "mixed_label_boundary": region["table"].mixed_label_boundary.to_numpy(
            dtype=float
        ),
    }
    for name, array in values.items():
        if len(array) != target_count:
            raise ValueError(f"Overlay length mismatch for {name}")
        nib.freesurfer.write_morph_data(
            str(overlay_root / f"D0086_{hemisphere}_{name}.mgh"),
            np.asarray(array, dtype=np.float32),
        )


def run_qc(config: dict) -> dict:
    """Run the complete O3 fsaverage QC and write all requested outputs."""

    output_root = Path(config["qc_output_root"]) / config["subject_id"]
    figure_root = output_root / "figures"
    table_root = output_root / "tables"
    overlay_root = output_root / "overlays"
    table_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    all_summary = []
    principal = {}
    figure_paths = []
    for hemisphere in config["hemispheres"]:
        bundle = _load_bundle(config, hemisphere)
        validation = _validate_bundle(bundle, config, hemisphere)
        mapping = _mapping_qc(bundle, config, hemisphere)
        geometry = _geometry_qc(bundle, mapping, config, hemisphere)
        coordinates = _coordinate_qc(bundle, config, geometry)
        md = _md_qc(bundle, config)
        spatial = _spatial_qc(bundle, mapping, geometry, hemisphere)
        region = _region_qc(bundle, mapping, hemisphere)

        mapping["invalid_table"].to_csv(
            table_root / f"{config['subject_id']}_{hemisphere}_invalid_vertices.csv",
            index=False,
        )
        geometry["outlier_table"].to_csv(
            table_root / f"{config['subject_id']}_{hemisphere}_geometry_outliers.csv",
            index=False,
        )
        geometry["zero_table"].to_csv(
            table_root
            / f"{config['subject_id']}_{hemisphere}_zero_length_vertices.csv",
            index=False,
        )
        coordinates["depth_table"].to_csv(
            table_root / f"{config['subject_id']}_{hemisphere}_coordinate_depth_qc.csv",
            index=False,
        )
        pd.DataFrame(coordinates["affected_ids"], columns=["target_vertex_id"]).to_csv(
            table_root
            / f"{config['subject_id']}_{hemisphere}_out_of_bounds_vertices.csv",
            index=False,
        )
        md["depth_table"].to_csv(
            table_root / f"{config['subject_id']}_{hemisphere}_md_depth_summary.csv",
            index=False,
        )
        md["outlier_table"].to_csv(
            table_root / f"{config['subject_id']}_{hemisphere}_md_profile_outliers.csv",
            index=False,
        )
        spatial["outlier_table"].to_csv(
            table_root
            / f"{config['subject_id']}_{hemisphere}_spatial_discontinuity_outliers.csv",
            index=False,
        )
        region["table"].to_csv(
            table_root
            / f"{config['subject_id']}_{hemisphere}_region_boundary_flags.csv",
            index=False,
        )
        region["summary_table"].to_csv(
            table_root / f"{config['subject_id']}_{hemisphere}_boundary_summary.csv",
            index=False,
        )
        figure_paths.extend(
            _make_figures(
                bundle,
                mapping,
                geometry,
                coordinates,
                md,
                spatial,
                region,
                figure_root,
                hemisphere,
            )
        )
        _save_overlays(
            bundle,
            mapping,
            geometry,
            coordinates,
            md,
            region,
            overlay_root,
            hemisphere,
        )

        profile_fraction = md["qc"]["complete_finite_profile_pct"] / 100.0
        in_bounds_fraction = coordinates["qc"]["in_bounds_pct"] / 100.0
        summary_rows = [
            {
                "category": "target_vertex_completeness",
                "metric": "loaded_vertices",
                "value": validation["dimensions"]["loaded_vertices"],
                "threshold": config["expected_fsaverage_vertices"],
                "status": "PASS"
                if validation["dimensions"]["loaded_vertices"]
                == config["expected_fsaverage_vertices"]
                else "FAIL",
            },
            {
                "category": "target_vertex_completeness",
                "metric": "duplicate_vertex_ids",
                "value": validation["dimensions"]["duplicate_vertices"],
                "threshold": 0,
                "status": "PASS"
                if validation["dimensions"]["duplicate_vertices"] == 0
                else "FAIL",
            },
            {
                "category": "mapping",
                "metric": "mapping_success_pct",
                "value": mapping["qc"]["mapping_success_pct"],
                "threshold": 99.0,
                "status": _status(mapping["qc"]["mapping_success_pct"], 99.0),
            },
            {
                "category": "barycentric_weights",
                "metric": "max_abs_weight_sum_error",
                "value": mapping["qc"]["max_abs_weight_sum_error"],
                "threshold": config["weight_tol"],
                "status": _status(
                    mapping["qc"]["max_abs_weight_sum_error"],
                    config["weight_tol"],
                    False,
                ),
            },
            {
                "category": "geometry",
                "metric": "coordinate_reconstruction_max_mm",
                "value": geometry["qc"]["coordinate_reconstruction_max_mm"],
                "threshold": config["coord_tol_mm"],
                "status": _status(
                    geometry["qc"]["coordinate_reconstruction_max_mm"],
                    config["coord_tol_mm"],
                    False,
                ),
            },
            {
                "category": "geometry",
                "metric": "zero_length_column_count",
                "value": geometry["qc"]["zero_length_column_count"],
                "threshold": 0,
                "status": "PASS"
                if geometry["qc"]["zero_length_column_count"] == 0
                else "FAIL",
            },
            {
                "category": "coordinates",
                "metric": "finite_t1_pct",
                "value": coordinates["qc"]["finite_t1_pct"],
                "threshold": 99.0,
                "status": _status(coordinates["qc"]["finite_t1_pct"], 99.0),
            },
            {
                "category": "coordinates",
                "metric": "dwi_in_bounds_pct",
                "value": coordinates["qc"]["in_bounds_pct"],
                "threshold": config["min_in_bounds_fraction"] * 100,
                "status": _status(
                    coordinates["qc"]["in_bounds_pct"],
                    config["min_in_bounds_fraction"] * 100,
                ),
            },
            {
                "category": "md_profile",
                "metric": "complete_finite_profile_pct",
                "value": md["qc"]["complete_finite_profile_pct"],
                "threshold": config["min_valid_profile_fraction"] * 100,
                "status": _status(
                    md["qc"]["complete_finite_profile_pct"],
                    config["min_valid_profile_fraction"] * 100,
                ),
            },
            {
                "category": "region_boundary",
                "metric": "mixed_label_triangle_pct",
                "value": region["qc"]["mixed_label_triangle_pct"],
                "threshold": "descriptive",
                "status": "WARNING"
                if region["qc"]["mixed_label_triangle_pct"] > 0
                else "PASS",
            },
        ]
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(
            table_root / f"{config['subject_id']}_{hemisphere}_o3_qc_summary.csv",
            index=False,
        )
        all_summary.append(summary.assign(hemisphere=hemisphere))
        principal[hemisphere] = {
            "validation": validation["dimensions"],
            "mapping": mapping["qc"],
            "geometry": geometry["qc"],
            "coordinates": coordinates["qc"],
            "md": md["qc"],
            "spatial": spatial["summary"],
            "region": region["qc"],
            "profile_fraction": profile_fraction,
            "in_bounds_fraction": in_bounds_fraction,
        }

    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_summary.to_csv(
        output_root / f"{config['subject_id']}_o3_qc_summary_all_hemispheres.csv",
        index=False,
    )
    metadata = {
        "config": {key: str(value) for key, value in config.items()},
        "source_o3_files": {
            hemisphere: str(
                Path(config["o3_output_root"])
                / config["subject_id"]
                / "fsaverage"
                / "o3_coordinate"
                / f"{config['subject_id']}_{hemisphere}_profiles.npz"
            )
            for hemisphere in config["hemispheres"]
        },
        "coordinate_conventions": {
            "white_anchor": "native FreeSurfer surface RAS in subject T1 space",
            "pial_anchor": "native FreeSurfer surface RAS in subject T1 space",
            "native_coordinates": (
                "straight white-to-pial coordinates in FreeSurfer surface RAS"
            ),
            "dwi_voxel_coordinates": (
                "0-based DWI voxel coordinates after subtracting 1 from "
                "transform output"
            ),
            "md_profiles": (
                "MD values sampled from the MD volume at saved DWI voxel coordinates"
            ),
        },
        "principal_metrics": principal,
        "figure_paths": [str(path) for path in figure_paths],
    }
    (output_root / f"{config['subject_id']}_o3_qc_metrics.json").write_text(
        json.dumps(metadata, indent=2, default=lambda value: value.tolist())
    )
    return {
        "output_root": output_root,
        "summary": combined_summary,
        "principal": principal,
        "figure_paths": figure_paths,
        "metadata": metadata,
    }
