#!/usr/bin/env python3
"""Build fsaverage-driven cortical columns in native T1 space.

For every vertex on the target template sphere, the script finds the closest
point on the subject's registered spherical mesh with
``vtkStaticCellLocator``. The source triangle and barycentric weights are then
applied to the subject's native white and pial surfaces. Each target vertex
therefore receives one subject-specific column with 21 points sampled from
pial to white, including both endpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


POINTS_NUM = 21
WEIGHT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=POINTS_NUM)
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    return parser.parse_args()


def read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Surface file not found: {path}")
    return fsio.read_geometry(str(path))


def sphere_metadata(points: np.ndarray) -> dict[str, object]:
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    radius = float(np.median(radii))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Sphere radius must be finite and positive")
    return {
        "center": center,
        "radius_median": radius,
        "radius_min": float(np.min(radii)),
        "radius_max": float(np.max(radii)),
    }


def normalize_sphere(
    points: np.ndarray,
    metadata: dict[str, object],
) -> np.ndarray:
    center = np.asarray(metadata["center"], dtype=float)
    return (points - center) / float(metadata["radius_median"])


def build_static_cell_locator(points: np.ndarray, faces: np.ndarray):
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError as error:
        raise RuntimeError(
            "The new template-column pipeline requires vtk with "
            "vtkStaticCellLocator. Load the VTK-enabled analysis environment."
        ) from error

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points.astype(np.float64), deep=True))
    vtk_cells = vtk.vtkCellArray()
    for face in faces:
        triangle = vtk.vtkTriangle()
        for local_index, vertex_id in enumerate(face):
            triangle.GetPointIds().SetId(local_index, int(vertex_id))
        vtk_cells.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    return locator


def barycentric_weights(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    offset = point - triangle[0]
    d00 = np.dot(edge_1, edge_1)
    d01 = np.dot(edge_1, edge_2)
    d11 = np.dot(edge_2, edge_2)
    d20 = np.dot(offset, edge_1)
    d21 = np.dot(offset, edge_2)
    denominator = d00 * d11 - d01 * d01
    if denominator <= np.finfo(float).eps:
        raise ValueError("Encountered a degenerate source sphere triangle")
    weight_2 = (d11 * d20 - d01 * d21) / denominator
    weight_3 = (d00 * d21 - d01 * d20) / denominator
    return np.array([1.0 - weight_2 - weight_3, weight_2, weight_3])


def locate_target_vertices(
    target_points: np.ndarray,
    source_points: np.ndarray,
    source_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import vtk

    locator = build_static_cell_locator(source_points, source_faces)
    n_target = len(target_points)
    face_ids = np.full(n_target, -1, dtype=np.int64)
    source_vertex_ids = np.full((n_target, 3), -1, dtype=np.int64)
    weights = np.full((n_target, 3), np.nan, dtype=float)
    closest_points = np.full((n_target, 3), np.nan, dtype=float)
    mapping_error = np.full(n_target, np.nan, dtype=float)

    for target_index, target_point in enumerate(target_points):
        if target_index == 0 or (target_index + 1) % 5000 == 0:
            logging.info(
                "Column mapping progress: %d/%d target vertices",
                target_index + 1,
                n_target,
            )
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.mutable(-1)
        sub_id = vtk.mutable(-1)
        distance_squared = vtk.mutable(0.0)
        locator.FindClosestPoint(
            target_point,
            closest,
            cell_id,
            sub_id,
            distance_squared,
        )
        face_id = int(cell_id)
        if face_id < 0 or face_id >= len(source_faces):
            continue
        triangle_ids = source_faces[face_id].astype(np.int64)
        triangle = source_points[triangle_ids]
        closest_array = np.asarray(closest, dtype=float)
        current_weights = barycentric_weights(closest_array, triangle)
        if (
            not np.isfinite(current_weights).all()
            or abs(current_weights.sum() - 1.0) > WEIGHT_TOLERANCE
            or (current_weights < -WEIGHT_TOLERANCE).any()
            or (current_weights > 1.0 + WEIGHT_TOLERANCE).any()
        ):
            continue
        reconstructed = current_weights @ triangle
        face_ids[target_index] = face_id
        source_vertex_ids[target_index] = triangle_ids
        weights[target_index] = current_weights
        closest_points[target_index] = closest_array
        mapping_error[target_index] = np.linalg.norm(reconstructed - closest_array)

    return (
        face_ids,
        source_vertex_ids,
        weights,
        closest_points,
        mapping_error,
    )


def process_hemisphere(
    subject_id: str,
    subjects_dir: Path,
    template: str,
    hemisphere: str,
    output_dir: Path,
    points_num: int,
) -> None:
    subject_root = subjects_dir / subject_id
    template_root = subjects_dir / template
    source_sphere, source_faces = read_surface(
        subject_root / "surf" / f"{hemisphere}.sphere.reg"
    )
    white, _ = read_surface(subject_root / "surf" / f"{hemisphere}.white")
    pial, _ = read_surface(subject_root / "surf" / f"{hemisphere}.pial")
    thickness_path = subject_root / "surf" / f"{hemisphere}.thickness"
    if not thickness_path.is_file():
        raise FileNotFoundError(f"Thickness file not found: {thickness_path}")
    native_thickness = np.asarray(
        fsio.read_morph_data(str(thickness_path)),
        dtype=float,
    )
    if len(native_thickness) != len(white):
        raise ValueError(
            f"Thickness and white surface vertex counts differ for {hemisphere}: "
            f"{len(native_thickness)} versus {len(white)}"
        )
    target_sphere_path = template_root / "surf" / f"{hemisphere}.sphere.reg"
    if not target_sphere_path.is_file():
        target_sphere_path = template_root / "surf" / f"{hemisphere}.sphere"
    target_sphere, _ = read_surface(target_sphere_path)
    annotation_path = template_root / "label" / f"{hemisphere}.aparc.annot"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Template annotation not found: {annotation_path}")
    target_labels, _, annotation_names = fsio.read_annot(str(annotation_path))
    target_labels = np.asarray(target_labels, dtype=np.int64)
    target_label_names = np.asarray(
        [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in annotation_names
        ],
        dtype=str,
    )
    if (target_labels >= len(target_label_names)).any():
        raise ValueError(
            f"Template annotation contains invalid label IDs: {annotation_path}"
        )
    normalized_label_names = np.asarray(
        [
            name.lower().replace("-", "_").replace(" ", "_")
            for name in target_label_names
        ],
        dtype=str,
    )
    unknown_indices = np.flatnonzero(normalized_label_names == "unknown")
    if np.any(target_labels < 0) and len(unknown_indices) == 0:
        raise ValueError(
            "Template annotation contains negative labels but no 'unknown' "
            f"entry is available: {annotation_path}"
        )
    safe_target_labels = target_labels.copy()
    if len(unknown_indices):
        safe_target_labels[safe_target_labels < 0] = int(unknown_indices[0])
    excluded_labels = {
        "unknown",
        "corpuscallosum",
        "medial_wall",
        "medialwall",
        "background+freesurfer_defined_medial_wall",
    }
    in_analysis_mask = ~np.isin(
        normalized_label_names[safe_target_labels],
        list(excluded_labels),
    )

    source_qc = sphere_metadata(source_sphere)
    target_qc = sphere_metadata(target_sphere)
    source_unit = normalize_sphere(source_sphere, source_qc)
    target_unit = normalize_sphere(target_sphere, target_qc)
    (
        face_ids,
        source_vertex_ids,
        weights,
        closest_points,
        mapping_error,
    ) = locate_target_vertices(target_unit, source_unit, source_faces)
    mapping_valid = face_ids >= 0

    white_endpoints = np.full((len(target_sphere), 3), np.nan, dtype=float)
    pial_endpoints = np.full((len(target_sphere), 3), np.nan, dtype=float)
    white_endpoints[mapping_valid] = np.einsum(
        "ij,ijk->ik",
        weights[mapping_valid],
        white[source_vertex_ids[mapping_valid]],
    )
    pial_endpoints[mapping_valid] = np.einsum(
        "ij,ijk->ik",
        weights[mapping_valid],
        pial[source_vertex_ids[mapping_valid]],
    )
    interpolated_thickness = np.full(len(target_sphere), np.nan, dtype=float)
    interpolated_thickness[mapping_valid] = np.einsum(
        "ij,ij->i",
        weights[mapping_valid],
        native_thickness[source_vertex_ids[mapping_valid]],
    )
    column_length = np.linalg.norm(
        pial_endpoints - white_endpoints,
        axis=1,
    )
    is_zero_length = np.isfinite(column_length) & (column_length < 1e-6)
    column_valid = mapping_valid & in_analysis_mask & ~is_zero_length
    depths = np.linspace(0.0, 1.0, points_num, dtype=float)
    column_points = (
        pial_endpoints[:, None, :] * (1.0 - depths[None, :, None])
        + white_endpoints[:, None, :] * depths[None, :, None]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_{hemisphere}_columns.npz"
    np.savez_compressed(
        output_path,
        subject_id=subject_id,
        hemisphere=hemisphere,
        template=template,
        fsaverage_vertex_id=np.arange(len(target_sphere), dtype=np.int64),
        target_vertex_id=np.arange(len(target_sphere), dtype=np.int64),
        template_label_id=target_labels.astype(np.int64),
        template_label_name=target_label_names[safe_target_labels],
        fsaverage_label=target_label_names[safe_target_labels],
        template_label_index=safe_target_labels.astype(np.int64),
        template_annotation_path=str(annotation_path),
        in_analysis_mask=in_analysis_mask,
        source_face_id=face_ids,
        native_triangle_id=face_ids,
        source_vertex_ids=source_vertex_ids,
        native_triangle_vertex_ids=source_vertex_ids,
        barycentric_weights=weights,
        source_sphere_center=np.asarray(source_qc["center"]),
        source_sphere_radius=float(source_qc["radius_median"]),
        target_sphere_center=np.asarray(target_qc["center"]),
        target_sphere_radius=float(target_qc["radius_median"]),
        target_sphere_path=str(target_sphere_path),
        closest_point_on_source_sphere=closest_points,
        mapping_error=mapping_error,
        mapping_valid=mapping_valid,
        white_endpoint=white_endpoints,
        pial_endpoint=pial_endpoints,
        white_coordinates_t1=white_endpoints,
        pial_coordinates_t1=pial_endpoints,
        column_points=column_points,
        column_length=column_length,
        cortical_thickness=interpolated_thickness,
        is_zero_length=is_zero_length,
        column_valid=column_valid,
        depth_fractions=depths,
        depth_direction="pial_to_white",
        mapping_method="vtkStaticCellLocator_closest_point_barycentric",
    )
    qc = {
        "subject_id": subject_id,
        "hemisphere": hemisphere,
        "template": template,
        "n_target_vertices": int(len(target_sphere)),
        "n_valid_mapping": int(mapping_valid.sum()),
        "n_failed_mapping": int((~mapping_valid).sum()),
        "valid_mapping_percentage": float(mapping_valid.mean() * 100),
        "n_in_analysis_mask": int(in_analysis_mask.sum()),
        "analysis_mask_percentage": float(in_analysis_mask.mean() * 100),
        "n_negative_annotation_labels_mapped_to_unknown": int(
            (target_labels < 0).sum()
        ),
        "n_zero_length": int(is_zero_length.sum()),
        "n_zero_length_in_analysis_mask": int(
            (is_zero_length & in_analysis_mask).sum()
        ),
        "column_length_zero_tolerance_mm": 1e-6,
        "column_length_median_mm": float(np.nanmedian(column_length)),
        "column_length_max_mm": float(np.nanmax(column_length)),
        "n_out_of_range_weights": int(
            np.any(
                (weights[mapping_valid] < -WEIGHT_TOLERANCE)
                | (weights[mapping_valid] > 1.0 + WEIGHT_TOLERANCE),
                axis=1,
            ).sum()
        ),
        "mapping_error_median": float(np.nanmedian(mapping_error)),
        "mapping_error_max": float(np.nanmax(mapping_error)),
        "source_sphere_center": np.asarray(source_qc["center"]).tolist(),
        "source_sphere_radius": float(source_qc["radius_median"]),
        "target_sphere_center": np.asarray(target_qc["center"]).tolist(),
        "target_sphere_radius": float(target_qc["radius_median"]),
        "depth_fractions": depths.tolist(),
        "depth_direction": "pial_to_white",
        "mapping_method": "vtkStaticCellLocator_closest_point_barycentric",
    }
    output_path.with_suffix(".qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    logging.info("Saved %s", output_path)


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise ValueError("--points must be at least 2")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    started = time.time()
    for hemisphere in args.hemispheres:
        process_hemisphere(
            subject_id=args.subject_id,
            subjects_dir=args.subjects_dir,
            template=args.template,
            hemisphere=hemisphere,
            output_dir=args.output_dir,
            points_num=args.points,
        )
    logging.info(
        "Template-column generation completed in %.1f seconds", time.time() - started
    )


if __name__ == "__main__":
    main()
