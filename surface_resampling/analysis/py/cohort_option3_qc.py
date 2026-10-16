#!/usr/bin/env python3
"""Build cohort-level QC and analysis masks for Option 3 fsaverage profiles."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


DEPTHS = np.linspace(0.0, 1.0, 21)
EXPECTED_VERTICES = 163842
EXPECTED_DEPTHS = 21
WEIGHT_TOL = 1e-6
COORD_TOL_MM = 1e-5
VALIDITY_WARN_PCT = 99.0
VALIDITY_FAIL_PCT = 95.0
COHORT_RELIABILITY_THRESHOLD = 0.95
EXCLUDED_LABELS = {"unknown", "corpuscallosum", "medial_wall"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--o3-root", type=Path, required=True)
    parser.add_argument("--freesurfer-subjects-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage", choices=["fsaverage"])
    parser.add_argument("--cohort-reliability", type=float, default=0.95)
    parser.add_argument("--subject-id", action="append")
    return parser.parse_args()


def decode_names(names: list[bytes | str]) -> list[str]:
    """Decode annotation names."""

    return [name.decode() if isinstance(name, bytes) else str(name) for name in names]


def load_template_info(
    subjects_dir: Path,
    template: str,
    hemisphere: str,
) -> dict[str, np.ndarray]:
    """Load template labels, cortex mask, surface faces, and white coordinates."""

    template_root = subjects_dir / template
    annot_path = template_root / "label" / f"{hemisphere}.aparc.annot"
    cortex_path = template_root / "label" / f"{hemisphere}.cortex.label"
    white_path = template_root / "surf" / f"{hemisphere}.white"
    inflated_path = template_root / "surf" / f"{hemisphere}.inflated"
    labels, _, names = nib.freesurfer.io.read_annot(str(annot_path))
    names = decode_names(list(names))
    valid_dk = np.zeros(len(labels), dtype=bool)
    for code, name in enumerate(names):
        if name not in EXCLUDED_LABELS:
            valid_dk |= labels == code
    cortex_vertices = np.asarray(nib.freesurfer.io.read_label(str(cortex_path)))
    if cortex_vertices.ndim > 1:
        cortex_vertices = cortex_vertices[:, 0]
    cortex_mask = np.zeros(len(labels), dtype=bool)
    cortex_mask[cortex_vertices.astype(int)] = True
    coordinates, faces = nib.freesurfer.io.read_geometry(str(white_path))
    display_coordinates, _ = nib.freesurfer.io.read_geometry(
        str(inflated_path)
    )
    return {
        "labels": labels.astype(np.int64),
        "cortex_mask": cortex_mask,
        "analysis_mask": cortex_mask & valid_dk,
        "coordinates": coordinates.astype(float),
        "display_coordinates": display_coordinates.astype(float),
        "faces": faces.astype(np.int64),
    }


def load_profile_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load one Option 3 profile bundle into memory."""

    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def ordered_array(bundle: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Return an array ordered by the original global target vertex ID."""

    target_ids = np.asarray(bundle["target_vertex_id"], dtype=np.int64)
    order = np.argsort(target_ids)
    expected = np.arange(EXPECTED_VERTICES, dtype=np.int64)
    if target_ids.shape != expected.shape or not np.array_equal(target_ids[order], expected):
        raise ValueError("target_vertex_id is not a complete global vertex index")
    value = np.asarray(bundle[key])
    return value[order]


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation or NaN for constant vectors."""

    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def unique_edges(faces: np.ndarray) -> np.ndarray:
    """Return unique undirected mesh edges."""

    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def compute_dk_boundary_hops(
    faces: np.ndarray,
    labels: np.ndarray,
    analysis_mask: np.ndarray,
) -> np.ndarray:
    """Compute mesh-hop distance from each vertex to the DK label boundary."""

    edges = unique_edges(faces)
    edge_left = edges[:, 0]
    edge_right = edges[:, 1]
    left_in_mask = analysis_mask[edge_left]
    right_in_mask = analysis_mask[edge_right]
    different_labels = labels[edge_left] != labels[edge_right]
    boundary = np.zeros(len(labels), dtype=bool)
    boundary[edge_left[left_in_mask & (~right_in_mask)]] = True
    boundary[edge_right[right_in_mask & (~left_in_mask)]] = True
    boundary[edge_left[left_in_mask & right_in_mask & different_labels]] = True
    boundary[edge_right[left_in_mask & right_in_mask & different_labels]] = True

    adjacency = csr_matrix(
        (
            np.ones(len(edges) * 2, dtype=float),
            (
                np.concatenate([edge_left, edge_right]),
                np.concatenate([edge_right, edge_left]),
            ),
        ),
        shape=(len(labels), len(labels)),
    )
    distances = dijkstra(
        adjacency,
        directed=False,
        indices=np.flatnonzero(boundary),
        min_only=True,
    )
    distances[~analysis_mask] = np.nan
    return distances


def classify_status(row: dict[str, object]) -> tuple[str, str]:
    """Assign a reproducible PASS, WARN, or FAIL status."""

    hard_fail = (
        row["dimension_valid"] is False
        or row["mapping_invalid_count"] > 0
        or row["invalid_weight_count"] > 0
        or row["face_vertex_mismatch_count"] > 0
        or row["mapping_error_max_mm"] > 1e-6
        or row["coordinate_reconstruction_max_mm"] > 1e-6
    )
    if hard_fail:
        return "FAIL", "dimension_or_mapping_validation"
    coverage = float(row["valid_profile_pct"])
    if coverage < VALIDITY_FAIL_PCT:
        return "FAIL", "analysis_mask_coverage_below_95_percent"
    warnings = []
    if coverage < VALIDITY_WARN_PCT:
        warnings.append("analysis_mask_coverage_below_99_percent")
    if row["zero_length_in_mask_pct"] > 0:
        warnings.append("zero_length_in_analysis_mask")
    if row["oob_vertex_in_mask_pct"] > 0:
        warnings.append("out_of_bounds_profile_in_analysis_mask")
    if row["md_zero_profile_in_mask_pct"] > 0:
        warnings.append("zero_md_in_analysis_mask")
    if row["md_nonfinite_profile_in_mask_pct"] > 0:
        warnings.append("nonfinite_md_in_analysis_mask")
    if row["length_outlier_in_mask_pct"] > 0:
        warnings.append("column_length_outliers_in_analysis_mask")
    if warnings:
        return "WARN", ";".join(warnings)
    return "PASS", "all_primary_checks_passed"


def compute_subject_hemisphere(
    subject: str,
    hemisphere: str,
    manifest_row: pd.Series,
    o3_root: Path,
    template_info: dict[str, np.ndarray],
    template: str,
    output_root: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Calculate vertex flags and summary metrics for one subject and hemisphere."""

    profile_path = (
        o3_root
        / subject
        / template
        / "o3_coordinate"
        / f"{subject}_{hemisphere}_profiles.npz"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    bundle = load_profile_bundle(profile_path)
    target_ids = ordered_array(bundle, "target_vertex_id")
    profiles = ordered_array(bundle, "md_profiles").astype(float)
    depths = np.asarray(bundle["depths"], dtype=float)
    if profiles.shape != (EXPECTED_VERTICES, EXPECTED_DEPTHS):
        raise ValueError(f"Unexpected profile shape: {profiles.shape}")
    if depths.shape != (EXPECTED_DEPTHS,) or not np.allclose(depths, DEPTHS):
        raise ValueError("Unexpected depth grid")

    source_face_id = ordered_array(bundle, "source_face_id").astype(np.int64)
    source_vertex_ids = ordered_array(bundle, "source_vertex_ids").astype(np.int64)
    weights = ordered_array(bundle, "barycentric_weights").astype(float)
    mapping_valid = ordered_array(bundle, "mapping_valid").astype(bool)
    native_root = Path(manifest_row["subjects_dir"]) / subject
    native_surface = native_root / "surf" / f"{hemisphere}.sphere.reg"
    native_annot = native_root / "label" / f"{hemisphere}.aparc.annot"
    native_sphere, native_faces = nib.freesurfer.io.read_geometry(str(native_surface))
    native_labels, _, _ = nib.freesurfer.io.read_annot(str(native_annot))
    source_ids_valid = np.all(
        (source_vertex_ids >= 0) & (source_vertex_ids < len(native_sphere)),
        axis=1,
    )
    face_valid = (source_face_id >= 0) & (source_face_id < len(native_faces))
    finite_weights = np.isfinite(weights).all(axis=1)
    weight_sum_error = np.abs(weights.sum(axis=1) - 1.0)
    weight_range_valid = (
        np.all(weights >= -WEIGHT_TOL, axis=1)
        & np.all(weights <= 1.0 + WEIGHT_TOL, axis=1)
    )
    face_vertex_match = np.zeros(EXPECTED_VERTICES, dtype=bool)
    rows = face_valid & source_ids_valid
    face_vertex_match[rows] = np.all(
        native_faces[source_face_id[rows]] == source_vertex_ids[rows],
        axis=1,
    )
    strict_mapping = (
        mapping_valid
        & face_valid
        & source_ids_valid
        & finite_weights
        & weight_range_valid
        & (weight_sum_error <= WEIGHT_TOL)
        & face_vertex_match
    )

    native_label_values = np.full(source_vertex_ids.shape, -1, dtype=np.int64)
    native_label_values[source_ids_valid] = native_labels[source_vertex_ids[source_ids_valid]]
    mixed_label = strict_mapping & (
        (native_label_values[:, 0] != native_label_values[:, 1])
        | (native_label_values[:, 1] != native_label_values[:, 2])
    )
    white_anchor = ordered_array(bundle, "white_anchor").astype(float)
    pial_anchor = ordered_array(bundle, "pial_anchor").astype(float)
    native_coordinates = ordered_array(bundle, "native_coordinates").astype(float)
    line_length = np.linalg.norm(pial_anchor - white_anchor, axis=1)
    zero_length = line_length <= COORD_TOL_MM
    finite_geometry = np.isfinite(white_anchor).all(axis=1) & np.isfinite(pial_anchor).all(axis=1)
    expected_coordinates = (
        (1.0 - depths[None, :, None]) * white_anchor[:, None, :]
        + depths[None, :, None] * pial_anchor[:, None, :]
    )
    coordinate_reconstruction_error = np.linalg.norm(
        native_coordinates - expected_coordinates,
        axis=2,
    )
    coordinate_mask = ordered_array(bundle, "coordinate_in_bounds_mask").astype(bool)
    dwi_voxel_coordinates = ordered_array(
        bundle,
        "dwi_voxel_coordinates",
    ).astype(float)
    metric_image_path = Path(np.asarray(bundle["metric_image"]).item())
    metric_shape = np.asarray(nib.load(str(metric_image_path)).shape[:3])
    finite_dwi = np.isfinite(dwi_voxel_coordinates).all(axis=2)
    raw_in_bounds = finite_dwi & np.all(
        (dwi_voxel_coordinates >= 0.0)
        & (dwi_voxel_coordinates <= metric_shape[None, None, :] - 1),
        axis=2,
    )
    oob_point = strict_mapping[:, None] & ~raw_in_bounds
    saved_bounds_mismatch_count = int(
        np.sum(coordinate_mask != (mapping_valid[:, None] & raw_in_bounds))
    )
    finite_md_point = np.isfinite(profiles)
    zero_md_point = profiles == 0.0
    nonfinite_md_point = ~finite_md_point
    zero_md_profile = zero_md_point.any(axis=1)
    nonfinite_md_profile = nonfinite_md_point.any(axis=1)
    invalid_md_profile = zero_md_profile | nonfinite_md_profile
    finite_profile = finite_md_point.all(axis=1)
    oob_vertex = oob_point.any(axis=1)
    final_valid = (
        template_info["analysis_mask"]
        & strict_mapping
        & finite_geometry
        & ~zero_length
        & ~oob_vertex
        & ~invalid_md_profile
    )
    analysis_mask = template_info["analysis_mask"]

    valid_length = analysis_mask & strict_mapping & finite_geometry
    length_values = line_length[valid_length]
    length_median = float(np.median(length_values)) if length_values.size else np.nan
    length_mad = (
        float(np.median(np.abs(length_values - length_median)))
        if length_values.size
        else np.nan
    )
    cutoff = length_median + 5.0 * 1.4826 * length_mad
    length_outlier = valid_length & (line_length > cutoff)

    thickness_path = native_root / "surf" / f"{hemisphere}.thickness"
    thickness = nib.freesurfer.io.read_morph_data(str(thickness_path))
    thickness_anchor = np.full(EXPECTED_VERTICES, np.nan, dtype=float)
    thickness_anchor[strict_mapping] = np.einsum(
        "ij,ij->i",
        weights[strict_mapping],
        thickness[source_vertex_ids[strict_mapping]],
    )
    thickness_valid = valid_length & np.isfinite(thickness_anchor)
    thickness_r = safe_corr(line_length[thickness_valid], thickness_anchor[thickness_valid])

    boundary_hops = template_info["dk_boundary_hops"]

    edges = unique_edges(template_info["faces"])
    edge_keep = final_valid[edges[:, 0]] & final_valid[edges[:, 1]]
    kept_edges = edges[edge_keep]
    if len(kept_edges):
        left = profiles[kept_edges[:, 0]]
        right = profiles[kept_edges[:, 1]]
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
            right_centered,
            axis=1,
        )
        edge_correlations = np.divide(
            np.sum(left_centered * right_centered, axis=1),
            denominator,
            out=np.full(len(kept_edges), np.nan),
            where=denominator > 0,
        )
        edge_mae = np.mean(np.abs(left - right), axis=1)
        edge_median = float(np.nanmedian(edge_mae))
        edge_mad = float(np.nanmedian(np.abs(edge_mae - edge_median)))
        discontinuity_cutoff = edge_median + 5.0 * 1.4826 * edge_mad
        discontinuity_pct = float((edge_mae > discontinuity_cutoff).mean() * 100)
        neighbor_r_median = float(np.nanmedian(edge_correlations))
    else:
        discontinuity_pct = np.nan
        neighbor_r_median = np.nan

    row: dict[str, object] = {
        "subject": subject,
        "hemisphere": hemisphere,
        "template": template,
        "n_vertices_total": EXPECTED_VERTICES,
        "n_analysis_mask": int(analysis_mask.sum()),
        "dimension_valid": True,
        "mapping_success_pct": float(mapping_valid.mean() * 100),
        "mapping_invalid_count": int((~strict_mapping).sum()),
        "invalid_weight_count": int((~weight_range_valid | ~finite_weights).sum()),
        "face_vertex_mismatch_count": int((rows & ~face_vertex_match).sum()),
        "mapping_error_max_mm": float(
            np.nanmax(np.asarray(bundle["mapping_error"])[strict_mapping])
        ),
        "coordinate_reconstruction_max_mm": float(
            np.nanmax(coordinate_reconstruction_error)
        ),
        "zero_length_all": int(zero_length.sum()),
        "zero_length_in_mask": int((zero_length & analysis_mask).sum()),
        "zero_length_in_mask_pct": float(
            (zero_length & analysis_mask).sum() / analysis_mask.sum() * 100
        ),
        "oob_point_pct": float(oob_point[analysis_mask].mean() * 100),
        "saved_bounds_mismatch_count": saved_bounds_mismatch_count,
        "oob_vertex_in_mask": int((oob_vertex & analysis_mask).sum()),
        "oob_vertex_in_mask_pct": float(
            (oob_vertex & analysis_mask).sum() / analysis_mask.sum() * 100
        ),
        "valid_profile_pct": float(final_valid[analysis_mask].mean() * 100),
        "mixed_label_pct": float(mixed_label[analysis_mask].mean() * 100),
        "md_zero_point_pct": float(
            zero_md_point[analysis_mask].mean() * 100
        ),
        "md_nonfinite_point_pct": float(
            nonfinite_md_point[analysis_mask].mean() * 100
        ),
        "md_zero_profile_in_mask": int(
            (zero_md_profile & analysis_mask).sum()
        ),
        "md_zero_profile_in_mask_pct": float(
            (zero_md_profile & analysis_mask).sum()
            / analysis_mask.sum()
            * 100
        ),
        "md_nonfinite_profile_in_mask": int(
            (nonfinite_md_profile & analysis_mask).sum()
        ),
        "md_nonfinite_profile_in_mask_pct": float(
            (nonfinite_md_profile & analysis_mask).sum()
            / analysis_mask.sum()
            * 100
        ),
        "invalid_md_profile_in_mask": int(
            (invalid_md_profile & analysis_mask).sum()
        ),
        "invalid_md_profile_in_mask_pct": float(
            (invalid_md_profile & analysis_mask).sum()
            / analysis_mask.sum()
            * 100
        ),
        "invalid_profile_in_mask": int((~final_valid & analysis_mask).sum()),
        "invalid_profile_in_mask_pct": float(
            (~final_valid & analysis_mask).sum() / analysis_mask.sum() * 100
        ),
        "mixed_label_boundary_hops_median": float(
            np.nanmedian(boundary_hops[mixed_label & analysis_mask])
        ),
        "length_median_mm": float(np.nanmedian(line_length[valid_length])),
        "length_iqr_mm": float(
            np.nanpercentile(line_length[valid_length], 75)
            - np.nanpercentile(line_length[valid_length], 25)
        ),
        "length_p01_mm": float(np.nanpercentile(line_length[valid_length], 1)),
        "length_p99_mm": float(np.nanpercentile(line_length[valid_length], 99)),
        "length_outlier_in_mask": int((length_outlier & analysis_mask).sum()),
        "length_outlier_in_mask_pct": float(
            (length_outlier & analysis_mask).sum() / analysis_mask.sum() * 100
        ),
        "thickness_r": thickness_r,
        "neighbor_r_median": neighbor_r_median,
        "discontinuity_pct": discontinuity_pct,
        "profile_zero_pct": float((profiles[analysis_mask] == 0).mean() * 100),
        "profile_nan_pct": float((~np.isfinite(profiles[analysis_mask])).mean() * 100),
        "depth_oob_pct": oob_point[analysis_mask].mean(axis=0) * 100,
        "depth_zero_md_pct": zero_md_point[analysis_mask].mean(axis=0) * 100,
        "depth_nonfinite_md_pct": nonfinite_md_point[analysis_mask].mean(axis=0) * 100,
        "depth_invalid_md_pct": (
            (zero_md_point | nonfinite_md_point)[analysis_mask].mean(axis=0) * 100
        ),
    }
    row["status"], row["reason"] = classify_status(row)

    flags = {
        "target_vertex_id": target_ids,
        "subject_id": np.asarray(subject),
        "in_analysis_mask": analysis_mask,
        "dk_label_code": template_info["labels"],
        "mapping_valid": strict_mapping,
        "zero_length": zero_length,
        "out_of_bounds": oob_vertex,
        "profile_finite": finite_profile,
        "md_zero": zero_md_profile,
        "md_nonfinite": nonfinite_md_profile,
        "invalid_md_profile": invalid_md_profile,
        "invalid_profile": ~final_valid,
        "mixed_label": mixed_label,
        "length_outlier": length_outlier,
        "line_length_mm": line_length,
        "thickness_anchor_mm": thickness_anchor,
        "dk_boundary_hops": boundary_hops,
        "final_valid": final_valid,
    }
    subject_output = output_root / "vertex_flags"
    subject_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subject_output / f"{subject}_{hemisphere}_flags.npz",
        **flags,
    )
    flagged = analysis_mask & (
        zero_length
        | oob_vertex
        | invalid_md_profile
        | mixed_label
        | length_outlier
    )
    pd.DataFrame(
        {
            "subject": subject,
            "hemisphere": hemisphere,
            "vertex_id": target_ids[flagged],
            "in_analysis_mask": analysis_mask[flagged],
            "dk_label_code": template_info["labels"][flagged],
            "zero_length": zero_length[flagged],
            "out_of_bounds": oob_vertex[flagged],
            "profile_finite": finite_profile[flagged],
            "md_zero": zero_md_profile[flagged],
            "md_nonfinite": nonfinite_md_profile[flagged],
            "invalid_md_profile": invalid_md_profile[flagged],
            "invalid_profile": (~final_valid)[flagged],
            "mixed_label": mixed_label[flagged],
            "length_outlier": length_outlier[flagged],
            "dk_boundary_hops": boundary_hops[flagged],
            "final_valid": final_valid[flagged],
        }
    ).to_csv(
        subject_output / f"{subject}_{hemisphere}_flagged_vertices.csv",
        index=False,
    )
    return row, flags


def save_frequency_outputs(
    flags_by_hemi: dict[str, list[dict[str, np.ndarray]]],
    template_info: dict[str, dict[str, np.ndarray]],
    output_root: Path,
    subjects: list[str],
    reliability_threshold: float,
) -> None:
    """Save vertex-level cohort frequencies, masks, and FreeSurfer overlays."""

    table_rows = []
    overlay_root = output_root / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    mask_rows = []
    for hemisphere, flag_list in flags_by_hemi.items():
        stack = {key: np.stack([flags[key] for flags in flag_list]) for key in flag_list[0]}
        n_subjects = len(flag_list)
        analysis_mask = template_info[hemisphere]["analysis_mask"]
        valid_count = stack["final_valid"].sum(axis=0)
        zero_count = stack["zero_length"].sum(axis=0)
        oob_count = stack["out_of_bounds"].sum(axis=0)
        md_zero_count = stack["md_zero"].sum(axis=0)
        md_nonfinite_count = stack["md_nonfinite"].sum(axis=0)
        invalid_md_count = stack["invalid_md_profile"].sum(axis=0)
        invalid_profile_count = stack["invalid_profile"].sum(axis=0)
        mixed_count = stack["mixed_label"].sum(axis=0)
        boundary_hops = stack["dk_boundary_hops"][0]
        frequency = pd.DataFrame(
            {
                "hemisphere": hemisphere,
                "vertex_id": np.arange(EXPECTED_VERTICES),
                "in_analysis_mask": analysis_mask,
                "n_subjects": n_subjects,
                "valid_profile_n": valid_count,
                "valid_profile_pct": valid_count / n_subjects * 100,
                "zero_length_n": zero_count,
                "zero_length_pct": zero_count / n_subjects * 100,
                "out_of_bounds_n": oob_count,
                "out_of_bounds_pct": oob_count / n_subjects * 100,
                "md_zero_n": md_zero_count,
                "md_zero_pct": md_zero_count / n_subjects * 100,
                "md_nonfinite_n": md_nonfinite_count,
                "md_nonfinite_pct": md_nonfinite_count / n_subjects * 100,
                "invalid_md_profile_n": invalid_md_count,
                "invalid_md_profile_pct": invalid_md_count / n_subjects * 100,
                "invalid_profile_n": invalid_profile_count,
                "invalid_profile_pct": invalid_profile_count / n_subjects * 100,
                "mixed_label_n": mixed_count,
                "mixed_label_pct": mixed_count / n_subjects * 100,
                "dk_boundary_hops": boundary_hops,
            }
        )
        table_rows.append(frequency)
        reliability = analysis_mask & (
            valid_count / n_subjects >= reliability_threshold
        )
        structural_path = overlay_root / f"fsaverage_{hemisphere}_structural_mask.mgh"
        reliability_path = overlay_root / f"fsaverage_{hemisphere}_cohort_reliability_mask.mgh"
        nib.freesurfer.write_morph_data(
            str(structural_path), analysis_mask.astype(np.float32)
        )
        nib.freesurfer.write_morph_data(
            str(reliability_path), reliability.astype(np.float32)
        )
        def mask_overlay(values: np.ndarray) -> np.ndarray:
            """Set values outside the structural analysis mask to NaN."""

            masked = np.full(EXPECTED_VERTICES, np.nan, dtype=float)
            masked[analysis_mask] = values[analysis_mask]
            return masked

        for name, values in {
            "zero_length_frequency": zero_count / n_subjects * 100,
            "out_of_bounds_frequency": oob_count / n_subjects * 100,
            "md_zero_frequency": md_zero_count / n_subjects * 100,
            "md_nonfinite_frequency": md_nonfinite_count / n_subjects * 100,
            "invalid_md_profile_frequency": invalid_md_count / n_subjects * 100,
            "invalid_profile_frequency": invalid_profile_count / n_subjects * 100,
            "mixed_label_frequency": mixed_count / n_subjects * 100,
            "valid_profile_frequency": valid_count / n_subjects * 100,
        }.items():
            nib.freesurfer.write_morph_data(
                str(overlay_root / f"fsaverage_{hemisphere}_{name}.mgh"),
                mask_overlay(values).astype(np.float32),
            )
        nib.freesurfer.write_morph_data(
            str(overlay_root / f"fsaverage_{hemisphere}_dk_boundary_hops.mgh"),
            mask_overlay(boundary_hops).astype(np.float32),
        )
        mask_rows.append(
            {
                "hemisphere": hemisphere,
                "n_vertices_total": EXPECTED_VERTICES,
                "n_structural_mask": int(analysis_mask.sum()),
                "n_cohort_reliability_mask": int(reliability.sum()),
                "cohort_reliability_threshold": reliability_threshold,
            }
        )
    pd.concat(table_rows, ignore_index=True).to_csv(
        output_root / "tables" / "cohort_vertex_frequency.csv",
        index=False,
    )
    pd.DataFrame(mask_rows).to_csv(
        output_root / "tables" / "analysis_mask_summary.csv",
        index=False,
    )


def make_spatial_frequency_maps(
    vertex_frequency: pd.DataFrame,
    template_info: dict[str, dict[str, np.ndarray]],
    output_root: Path,
) -> str:
    """Plot cohort frequency maps for mixed-label and invalid locations."""

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mixed_label_pct", "Mixed-label frequency"),
        ("zero_length_pct", "Zero-length frequency"),
        ("out_of_bounds_pct", "Out-of-bounds frequency"),
        ("invalid_profile_pct", "Invalid-profile frequency"),
    ]
    hemispheres = ["lh", "rh"]
    fig, axes = plt.subplots(
        len(metrics),
        len(hemispheres),
        figsize=(10, 13),
        squeeze=False,
    )

    for row_index, (metric, title) in enumerate(metrics):
        row_values = vertex_frequency.loc[
            vertex_frequency.in_analysis_mask,
            metric,
        ].to_numpy(dtype=float)
        row_values = row_values[np.isfinite(row_values)]
        row_vmax = max(1.0, float(np.nanmax(row_values)))
        row_surface = None
        for column_index, hemisphere in enumerate(hemispheres):
            axis = axes[row_index, column_index]
            group = vertex_frequency[
                vertex_frequency.hemisphere.eq(hemisphere)
            ].sort_values("vertex_id")
            coordinates = template_info[hemisphere]["display_coordinates"]
            faces = template_info[hemisphere]["faces"]
            values = group[metric].to_numpy(dtype=float)
            values[~group.in_analysis_mask.to_numpy(dtype=bool)] = np.nan
            triangle_values = values[faces]
            triangle_valid = np.all(np.isfinite(triangle_values), axis=1)
            triangulation = mtri.Triangulation(
                coordinates[:, 0],
                coordinates[:, 1],
                triangles=faces,
            )
            triangulation.set_mask(~triangle_valid)
            surface = axis.tripcolor(
                triangulation,
                values,
                cmap="magma",
                vmin=0.0,
                vmax=row_vmax,
                shading="gouraud",
            )
            row_surface = surface
            axis.set_axis_off()
            axis.set_aspect("equal")
            axis.set_title(hemisphere.upper(), fontsize=10)
            if column_index == 0:
                axis.text(
                    -0.08,
                    0.5,
                    title,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                )

        colorbar = fig.colorbar(
            row_surface,
            ax=axes[row_index, :].tolist(),
            fraction=0.025,
            pad=0.04,
        )
        colorbar.set_label("Frequency (%)")
    fig.suptitle("Spatial frequency of mixed-label and invalid locations")
    path = figure_root / "09_spatial_frequency_maps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_figures(
    subject_qc: pd.DataFrame,
    vertex_frequency: pd.DataFrame,
    output_root: Path,
) -> list[str]:
    """Create cohort-level QC figures."""

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = subject_qc.sort_values(["hemisphere", "subject"])
    colors = ordered.hemisphere.map({"lh": "#0072B2", "rh": "#D55E00"})
    ax.scatter(np.arange(len(ordered)), ordered.valid_profile_pct, c=colors, s=24)
    ax.axhline(99, color="black", linestyle="--", linewidth=0.8, label="99%")
    ax.axhline(95, color="#777777", linestyle=":", linewidth=0.8, label="95%")
    ax.set_ylabel("Valid complete profiles in analysis mask (%)")
    ax.set_xlabel("Subject × hemisphere")
    ax.set_title("Cohort profile coverage")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_root / "01_valid_profile_coverage.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    zero_data = []
    for _, row in subject_qc.iterrows():
        zero_data.extend(
            [
                {"hemisphere": row.hemisphere, "scope": "Full mesh", "pct": row.zero_length_all / row.n_vertices_total * 100},
                {"hemisphere": row.hemisphere, "scope": "Analysis mask", "pct": row.zero_length_in_mask_pct},
            ]
        )
    zero_df = pd.DataFrame(zero_data)
    positions = {("lh", "Full mesh"): 0, ("lh", "Analysis mask"): 1, ("rh", "Full mesh"): 3, ("rh", "Analysis mask"): 4}
    for key, group in zero_df.groupby(["hemisphere", "scope"]):
        ax.scatter(np.full(len(group), positions[key]), group.pct, alpha=0.75, s=18, label=f"{key[0]} {key[1]}")
    ax.set_xticks([0.5, 3.5], ["Left hemisphere", "Right hemisphere"])
    ax.set_ylabel("Zero-length columns (%)")
    ax.set_title("Zero-length columns: full mesh versus analysis mask")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_root / "02_zero_length_full_mesh_vs_mask.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for hemisphere, group in subject_qc.groupby("hemisphere"):
        axes[0].plot(
            group.sort_values("subject").subject,
            group.sort_values("subject").oob_vertex_in_mask_pct,
            marker="o",
            label=hemisphere,
        )
    axes[0].set_ylabel("Vertices with any out-of-bounds depth (%)")
    axes[0].set_xlabel("Subject")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].legend(frameon=False)
    depth_rows = []
    for _, row in subject_qc.iterrows():
        for depth in DEPTHS:
            value = row[f"depth_{depth:.2f}_oob_pct"]
            depth_rows.append({"hemisphere": row.hemisphere, "depth": depth, "pct": value})
    depth_df = pd.DataFrame(depth_rows)
    for hemisphere, group in depth_df.groupby("hemisphere"):
        summary = group.groupby("depth").pct.mean()
        axes[1].plot(summary.index, summary.values, marker="o", label=hemisphere)
    axes[1].set_xlabel("Cortical depth")
    axes[1].set_ylabel("Out-of-bounds points (%)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Out-of-bounds coordinate QC")
    fig.tight_layout()
    path = figure_root / "03_out_of_bounds_vertex_and_depth.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for hemisphere, group in subject_qc.groupby("hemisphere"):
        axes[0].scatter(group.length_median_mm, group.thickness_r, label=hemisphere, s=28)
        axes[1].scatter(group.length_iqr_mm, group.thickness_r, label=hemisphere, s=28)
    axes[0].set_xlabel("Median column length (mm)")
    axes[1].set_xlabel("Column length IQR (mm)")
    for ax in axes:
        ax.set_ylabel("Correlation with FreeSurfer thickness")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Column length and FreeSurfer thickness")
    fig.tight_layout()
    path = figure_root / "04_column_length_and_thickness.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=False)
    axes = axes.ravel()
    for ax, metric, title in zip(
        axes,
        [
            "zero_length_pct",
            "out_of_bounds_pct",
            "invalid_md_profile_pct",
            "mixed_label_pct",
        ],
        [
            "Zero-length",
            "Out-of-bounds",
            "Invalid MD profile",
            "Mixed-label",
        ],
    ):
        for hemisphere, group in vertex_frequency.groupby("hemisphere"):
            values = group.loc[group.in_analysis_mask, metric]
            ax.hist(values, bins=30, alpha=0.5, label=hemisphere)
        ax.set_title(title)
        ax.set_xlabel("Frequency across subjects (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Vertices")
    axes[0].legend(frameon=False)
    fig.suptitle("Cohort flag frequencies within the analysis mask")
    fig.tight_layout()
    path = figure_root / "05_flag_frequency_distributions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    for hemisphere, group in vertex_frequency.groupby("hemisphere"):
        values = group.loc[
            group.in_analysis_mask & (group.mixed_label_pct > 0),
            "dk_boundary_hops",
        ].dropna()
        ax.hist(values, bins=30, alpha=0.5, label=hemisphere)
    ax.set_xlabel("Distance to nearest DK boundary (surface hops)")
    ax.set_ylabel("Mixed-label vertices")
    ax.set_title("Mixed-label locations relative to DK boundaries")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_root / "07_mixed_label_distance_to_dk_boundary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    metrics = [
        "valid_profile_pct",
        "zero_length_in_mask_pct",
        "oob_vertex_in_mask_pct",
        "invalid_md_profile_in_mask_pct",
        "invalid_profile_in_mask_pct",
        "mixed_label_pct",
        "length_median_mm",
        "neighbor_r_median",
    ]
    matrix = subject_qc.set_index(["subject", "hemisphere"])[metrics]
    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0)
    fig, ax = plt.subplots(figsize=(8, 10))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(matrix)), [f"{a} {b}" for a, b in matrix.index])
    ax.set_xticks(np.arange(len(metrics)), metrics, rotation=45, ha="right")
    ax.set_title("Subject × QC metric standardized heatmap")
    fig.colorbar(image, ax=ax, label="Within-metric z-score")
    fig.tight_layout()
    path = figure_root / "06_subject_qc_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def build_merged_subject_qc(
    subject_qc: pd.DataFrame,
    flags_by_hemi: dict[str, list[dict[str, np.ndarray]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build hemisphere-pooled subject and depth-level QC summaries."""

    subject_rows = []
    for subject, group in subject_qc.groupby("subject", sort=True):
        total_vertices = group.n_vertices_total.sum()
        total_mask = group.n_analysis_mask.sum()
        valid_vertices = (
            group.valid_profile_pct / 100.0 * group.n_analysis_mask
        ).sum()
        subject_rows.append(
            {
                "subject": subject,
                "coverage_pct": valid_vertices / total_mask * 100.0,
                "zero_length_full_mesh_pct": (
                    group.zero_length_all.sum() / total_vertices * 100.0
                ),
                "zero_length_analysis_mask_pct": (
                    group.zero_length_in_mask.sum() / total_mask * 100.0
                ),
                "oob_points_pct": (
                    (
                        group.oob_point_pct / 100.0 * group.n_analysis_mask
                    ).sum()
                    / total_mask
                    * 100.0
                ),
                "zero_md_pct": (
                    (
                        group.md_zero_point_pct / 100.0 * group.n_analysis_mask
                    ).sum()
                    / total_mask
                    * 100.0
                ),
                "nonfinite_md_pct": (
                    (
                        group.md_nonfinite_point_pct
                        / 100.0
                        * group.n_analysis_mask
                    ).sum()
                    / total_mask
                    * 100.0
                ),
                "mixed_label_pct": (
                    (
                        group.mixed_label_pct / 100.0 * group.n_analysis_mask
                    ).sum()
                    / total_mask
                    * 100.0
                ),
                "n_analysis_mask": int(total_mask),
            }
        )
    merged_subject = pd.DataFrame(subject_rows)

    depth_rows = []
    depth_metric_names = {
        "oob": "oob",
        "zero_md": "zero_md",
        "nonfinite_md": "nonfinite_md",
    }
    for depth in DEPTHS:
        row = {"depth": depth}
        denominator = subject_qc.n_analysis_mask.sum()
        for output_name, suffix in depth_metric_names.items():
            column = f"depth_{depth:.2f}_{suffix}_pct"
            numerator = (
                subject_qc[column] / 100.0 * subject_qc.n_analysis_mask
            ).sum()
            row[f"{output_name}_pct"] = numerator / denominator * 100.0
        depth_rows.append(row)
    depth_summary = pd.DataFrame(depth_rows)

    correlation_rows = []
    for hemisphere, flag_list in flags_by_hemi.items():
        for flags in flag_list:
            subject = str(np.asarray(flags["subject_id"]).item())
            keep = (
                flags["final_valid"]
                & np.isfinite(flags["line_length_mm"])
                & np.isfinite(flags["thickness_anchor_mm"])
            )
            correlation_rows.append(
                {
                    "subject": subject,
                    "hemisphere": hemisphere,
                    "line_length_mm": flags["line_length_mm"][keep],
                    "thickness_anchor_mm": flags["thickness_anchor_mm"][keep],
                }
            )

    pooled_rows = []
    for subject in sorted(merged_subject.subject):
        length_values = []
        thickness_values = []
        for row in correlation_rows:
            if row["subject"] == subject:
                length_values.append(row["line_length_mm"])
                thickness_values.append(row["thickness_anchor_mm"])
        if length_values:
            length_values = np.concatenate(length_values)
            thickness_values = np.concatenate(thickness_values)
            correlation = safe_corr(length_values, thickness_values)
        else:
            correlation = np.nan
        pooled_rows.append(
            {
                "subject": subject,
                "length_thickness_r": correlation,
            }
        )
    merged_subject = merged_subject.merge(
        pd.DataFrame(pooled_rows),
        on="subject",
        how="left",
    )

    boundary_rows = []
    for subject in sorted(merged_subject.subject):
        mixed_count = 0
        boundary_mixed_count = 0
        for flag_list in flags_by_hemi.values():
            for flags in flag_list:
                flag_subject = str(np.asarray(flags["subject_id"]).item())
                if flag_subject != subject:
                    continue
                mixed = flags["mixed_label"] & flags["in_analysis_mask"]
                boundary_mixed = mixed & (flags["dk_boundary_hops"] <= 2)
                mixed_count += int(mixed.sum())
                boundary_mixed_count += int(boundary_mixed.sum())
        boundary_rows.append(
            {
                "subject": subject,
                "mixed_label_count": mixed_count,
                "boundary_mixed_count": boundary_mixed_count,
                "mixed_boundary_associated_pct": (
                    boundary_mixed_count / mixed_count * 100.0
                    if mixed_count
                    else np.nan
                ),
            }
        )
    merged_subject = merged_subject.merge(
        pd.DataFrame(boundary_rows),
        on="subject",
        how="left",
    )
    return merged_subject, depth_summary


def make_merged_summary_figure(
    merged_subject: pd.DataFrame,
    depth_summary: pd.DataFrame,
    output_root: Path,
) -> str:
    """Create the hemisphere-pooled cohort-level QC summary figure."""

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    coverage = merged_subject.sort_values("coverage_pct").reset_index(drop=True)
    axes[0].scatter(
        np.arange(len(coverage)),
        coverage.coverage_pct,
        color="#0072B2",
        s=24,
    )
    axes[0].axhline(99.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].axhline(95.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("Subject")
    axes[0].set_ylabel("Weighted valid complete profiles (%)")
    axes[0].set_title("A. Hemisphere-pooled profile coverage")
    axes[0].set_xticks(np.arange(len(coverage)))
    axes[0].set_xticklabels(coverage.subject, rotation=90, fontsize=6)

    paired = merged_subject[
        ["zero_length_full_mesh_pct", "zero_length_analysis_mask_pct"]
    ].to_numpy()
    for values in paired:
        axes[1].plot([0, 1], values, color="#999999", alpha=0.45, linewidth=0.8)
    axes[1].boxplot(
        [paired[:, 0], paired[:, 1]],
        positions=[0, 1],
        widths=0.45,
        showfliers=False,
    )
    axes[1].set_xticks([0, 1], ["Full mesh", "Analysis mask"])
    axes[1].set_ylabel("Zero-length vertices (%)")
    axes[1].set_title("B. Zero-length geometry before and after masking")

    for column, label, color in [
        ("oob_pct", "Out-of-bounds", "#D55E00"),
        ("zero_md_pct", "Zero MD", "#009E73"),
        ("nonfinite_md_pct", "NaN/Inf", "#CC79A7"),
    ]:
        axes[2].plot(
            depth_summary.depth,
            depth_summary[column],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=label,
            color=color,
        )
    axes[2].set_xlabel("Cortical depth")
    axes[2].set_ylabel("Invalid sampling points (%)")
    axes[2].set_title("C. Invalid MD sampling by depth")
    axes[2].legend(frameon=False, fontsize=8)

    correlation = merged_subject.length_thickness_r.dropna().sort_values()
    axes[3].boxplot([correlation], positions=[0], widths=0.45, showfliers=False)
    axes[3].scatter(
        np.full(len(correlation), 0.0),
        correlation,
        color="#0072B2",
        s=22,
        alpha=0.8,
    )
    axes[3].set_xticks([0], ["30 subjects"])
    axes[3].set_ylabel("Pooled length–thickness Pearson r")
    axes[3].set_title("D. Column length and FreeSurfer thickness")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cohort-level QC summary", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = figure_root / "08_cohort_level_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_merged_summary_panels(
    merged_subject: pd.DataFrame,
    depth_summary: pd.DataFrame,
    output_root: Path,
) -> list[str]:
    """Create separate figures for the four pooled QC summaries."""

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []

    coverage = merged_subject.sort_values("coverage_pct").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(
        np.arange(len(coverage)),
        coverage.coverage_pct,
        color="#0072B2",
        s=28,
    )
    ax.axhline(99.0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(95.0, color="#777777", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Weighted valid complete profiles (%)")
    ax.set_title("Hemisphere-pooled profile coverage")
    ax.set_xticks(np.arange(len(coverage)))
    ax.set_xticklabels(coverage.subject, rotation=90, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_root / "08a_hemisphere_pooled_profile_coverage.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    paired = merged_subject[
        ["zero_length_full_mesh_pct", "zero_length_analysis_mask_pct"]
    ].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    for values in paired:
        ax.plot([0, 1], values, color="#999999", alpha=0.45, linewidth=0.8)
    ax.boxplot(
        [paired[:, 0], paired[:, 1]],
        positions=[0, 1],
        widths=0.45,
        showfliers=False,
    )
    ax.set_xticks([0, 1], ["Full mesh", "Analysis mask"])
    ax.set_ylabel("Zero-length vertices (%)")
    ax.set_title("Zero-length geometry before and after masking")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_root / "08b_zero_length_full_mesh_vs_mask.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    for column, label, color in [
        ("oob_pct", "Out-of-bounds", "#D55E00"),
        ("zero_md_pct", "Zero MD", "#009E73"),
        ("nonfinite_md_pct", "NaN/Inf", "#CC79A7"),
    ]:
        ax.plot(
            depth_summary.depth,
            depth_summary[column],
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=label,
            color=color,
        )
    ax.set_xlabel("Cortical depth")
    ax.set_ylabel("Invalid sampling points (%)")
    ax.set_title("Invalid MD sampling by depth")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_root / "08c_invalid_md_sampling_by_depth.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    correlation = merged_subject.length_thickness_r.dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.boxplot([correlation], positions=[0], widths=0.45, showfliers=False)
    ax.scatter(
        np.full(len(correlation), 0.0),
        correlation,
        color="#0072B2",
        s=24,
        alpha=0.8,
    )
    ax.set_xticks([0], ["30 subjects"])
    ax.set_ylabel("Pooled length-thickness Pearson r")
    ax.set_title("Column length and FreeSurfer thickness")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_root / "08d_length_thickness_correlation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    return paths


def save_merged_summary_tables(
    merged_subject: pd.DataFrame,
    depth_summary: pd.DataFrame,
    output_root: Path,
) -> None:
    """Save hemisphere-pooled subject and depth summaries."""

    table_root = output_root / "tables"
    merged_subject.to_csv(table_root / "cohort_subject_merged_qc.csv", index=False)
    depth_summary.to_csv(table_root / "cohort_depth_invalid_summary.csv", index=False)
    summary_rows = []
    summary_definitions = [
        ("Valid complete-profile coverage", "coverage_pct", ["median", "IQR", "minimum"]),
        (
            "Zero-length: full mesh",
            "zero_length_full_mesh_pct",
            ["median", "range"],
        ),
        (
            "Zero-length: analysis mask",
            "zero_length_analysis_mask_pct",
            ["median", "range"],
        ),
        (
            "Length-thickness correlation",
            "length_thickness_r",
            ["median", "IQR", "range"],
        ),
    ]
    for metric, column, statistics in summary_definitions:
        values = merged_subject[column].dropna()
        statistic_values = {
            "median": values.median(),
            "IQR": values.quantile(0.75) - values.quantile(0.25),
            "minimum": values.min(),
            "range": values.max() - values.min(),
        }
        for statistic in statistics:
            summary_rows.append(
                {
                    "metric": metric,
                    "statistic": statistic,
                    "value": statistic_values[statistic],
                }
            )

    total_subject_mask = merged_subject.n_analysis_mask.sum()
    for metric, column in [
        ("Out-of-bounds points", "oob_points_pct"),
        ("Zero-MD", "zero_md_pct"),
        ("NaN/Inf", "nonfinite_md_pct"),
        ("Mixed-label", "mixed_label_pct"),
    ]:
        overall = (
            merged_subject[column] * merged_subject.n_analysis_mask
        ).sum() / total_subject_mask
        summary_rows.append(
            {
                "metric": metric,
                "statistic": "overall percentage",
                "value": overall,
            }
        )
    summary_rows.append(
        {
            "metric": "Out-of-bounds points",
            "statistic": "maximum subject",
            "value": merged_subject.oob_points_pct.max(),
        }
    )
    total_mixed = merged_subject.mixed_label_count.sum()
    summary_rows.append(
        {
            "metric": "Mixed-label",
            "statistic": "boundary-associated percentage (<=2 hops)",
            "value": (
                merged_subject.boundary_mixed_count.sum() / total_mixed * 100.0
                if total_mixed
                else np.nan
            ),
        }
    )
    pd.DataFrame(summary_rows).to_csv(
        table_root / "cohort_qc_summary_table.csv",
        index=False,
    )


def main() -> None:
    """Run the cohort-level Option 3 QC."""

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = pd.read_csv(args.manifest, dtype=str)
    if args.subject_id:
        manifest = manifest[manifest.subject_id.isin(args.subject_id)].copy()
    manifest = manifest.sort_values("subject_id").reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No subjects remain in the manifest")

    output_root = args.output_root
    table_root = output_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    template_info = {
        hemisphere: load_template_info(
            args.freesurfer_subjects_dir,
            args.template,
            hemisphere,
        )
        for hemisphere in ["lh", "rh"]
    }
    for hemisphere in ["lh", "rh"]:
        template_info[hemisphere]["dk_boundary_hops"] = compute_dk_boundary_hops(
            template_info[hemisphere]["faces"],
            template_info[hemisphere]["labels"],
            template_info[hemisphere]["analysis_mask"],
        )
    rows = []
    flags_by_hemi = {"lh": [], "rh": []}
    failures = []
    for index, manifest_row in manifest.iterrows():
        subject = manifest_row.subject_id
        logging.info("Progress %d/%d subject=%s", index + 1, len(manifest), subject)
        for hemisphere in ["lh", "rh"]:
            try:
                row, flags = compute_subject_hemisphere(
                    subject,
                    hemisphere,
                    manifest_row,
                    args.o3_root,
                    template_info[hemisphere],
                    args.template,
                    output_root,
                )
                rows.append(row)
                flags_by_hemi[hemisphere].append(flags)
                logging.info(
                    "Completed subject=%s hemisphere=%s status=%s coverage=%.3f%%",
                    subject,
                    hemisphere,
                    row["status"],
                    row["valid_profile_pct"],
                )
            except Exception as exc:
                logging.exception("Failed subject=%s hemisphere=%s", subject, hemisphere)
                failures.append(
                    {
                        "subject": subject,
                        "hemisphere": hemisphere,
                        "stage": "cohort_qc",
                        "error": repr(exc),
                    }
                )

    subject_qc = pd.DataFrame(rows)
    if subject_qc.empty:
        raise RuntimeError("No subject-by-hemisphere QC rows were generated")
    depth_metric_names = {
        "depth_oob_pct": "oob",
        "depth_zero_md_pct": "zero_md",
        "depth_nonfinite_md_pct": "nonfinite_md",
        "depth_invalid_md_pct": "invalid_md",
    }
    depth_columns_data = {}
    for source_name, output_name in depth_metric_names.items():
        depth_columns = [f"depth_{depth:.2f}_{output_name}_pct" for depth in DEPTHS]
        depth_values = np.vstack(subject_qc.pop(source_name).to_numpy())
        for column, values in zip(depth_columns, depth_values.T):
            depth_columns_data[column] = values
    subject_qc = pd.concat(
        [subject_qc, pd.DataFrame(depth_columns_data)],
        axis=1,
    )
    subject_qc.to_csv(table_root / "cohort_subject_hemisphere_qc.csv", index=False)
    pd.DataFrame(
        failures,
        columns=["subject", "hemisphere", "stage", "error"],
    ).to_csv(table_root / "failure_summary.csv", index=False)

    save_frequency_outputs(
        flags_by_hemi,
        template_info,
        output_root,
        manifest.subject_id.tolist(),
        args.cohort_reliability,
    )
    vertex_frequency = pd.read_csv(table_root / "cohort_vertex_frequency.csv")
    figure_paths = make_figures(subject_qc, vertex_frequency, output_root)
    figure_paths.append(
        make_spatial_frequency_maps(
            vertex_frequency,
            template_info,
            output_root,
        )
    )
    merged_subject, depth_summary = build_merged_subject_qc(
        subject_qc,
        flags_by_hemi,
    )
    save_merged_summary_tables(merged_subject, depth_summary, output_root)
    figure_paths.append(
        make_merged_summary_figure(
            merged_subject,
            depth_summary,
            output_root,
        )
    )
    figure_paths.extend(
        make_merged_summary_panels(
            merged_subject,
            depth_summary,
            output_root,
        )
    )

    metadata = {
        "manifest": str(args.manifest),
        "subjects": manifest.subject_id.tolist(),
        "n_subjects": int(manifest.subject_id.nunique()),
        "template": args.template,
        "hemispheres": ["lh", "rh"],
        "depths": DEPTHS.tolist(),
        "analysis_mask_definition": "fsaverage cortex.label intersected with non-excluded aparc labels",
        "final_valid_definition": "analysis mask and strict mapping and finite 21-point profile with no zero MD values and all 21 DWI points in bounds and nonzero white-pial length",
        "oob_policy": "any out-of-bounds depth invalidates the vertex profile",
        "md_zero_policy": "any zero MD value at any of the 21 depths invalidates the vertex profile",
        "md_nonfinite_policy": "any NaN or Inf MD value at any of the 21 depths invalidates the vertex profile",
        "boundary_distance_definition": "minimum number of fsaverage white-surface mesh edges to an analysis-mask DK-label boundary",
        "cohort_reliability_threshold": args.cohort_reliability,
        "validity_warn_pct": VALIDITY_WARN_PCT,
        "validity_fail_pct": VALIDITY_FAIL_PCT,
        "figures": figure_paths,
        "n_failures": len(failures),
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    logging.info("Finished cohort QC. Output root: %s", output_root)


if __name__ == "__main__":
    main()
