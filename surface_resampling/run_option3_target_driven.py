#!/usr/bin/env python3
"""Generate target-driven native-column MD profiles with barycentric mapping."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_processing.transform_trajectories import (
    load_dwi_geometry,
    load_transform_matrix,
    transform_points_to_dwi,
)


METHOD_NAME = "O3-Coordinate"
SAMPLING_ORDER = 1
SAMPLING_MODE = "nearest"


def normalize_sphere(points: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Center a FreeSurfer sphere and return unit directions and QC metadata."""

    center = np.median(points, axis=0)
    centered = points - center
    radii = np.linalg.norm(centered, axis=1)
    radius = float(np.median(radii))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Sphere radius is not finite and positive")

    unit_points = centered / radii[:, None]
    metadata = {
        "center": center,
        "radius_median": radius,
        "radius_min": float(np.min(radii)),
        "radius_max": float(np.max(radii)),
        "radius_relative_spread": float((np.max(radii) - np.min(radii)) / radius),
    }
    return unit_points, metadata


def ray_triangle_intersection(
    direction: np.ndarray,
    triangle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Intersect a ray from the origin with a planar triangle."""

    origin = np.zeros(3, dtype=float)
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    cross = np.cross(direction, edge_2)
    determinant = np.dot(edge_1, cross)
    if abs(determinant) < 1e-12:
        return None
    inverse = 1.0 / determinant
    distance = origin - triangle[0]
    weight_1 = inverse * np.dot(distance, cross)
    if weight_1 < -1e-8 or weight_1 > 1.0 + 1e-8:
        return None
    cross_2 = np.cross(distance, edge_1)
    weight_2 = inverse * np.dot(direction, cross_2)
    if weight_2 < -1e-8 or weight_1 + weight_2 > 1.0 + 1e-8:
        return None
    distance_along_ray = inverse * np.dot(edge_2, cross_2)
    if distance_along_ray <= 0:
        return None
    weights = np.array([1.0 - weight_1 - weight_2, weight_1, weight_2])
    intersection = distance_along_ray * direction
    return intersection, weights, float(distance_along_ray)


def map_target_to_source(
    target_unit: np.ndarray,
    source_unit: np.ndarray,
    faces: np.ndarray,
):
    tree = cKDTree(source_unit)
    incident_faces = [[] for _ in range(len(source_unit))]
    for face_index, face in enumerate(faces):
        for vertex in face:
            incident_faces[int(vertex)].append(face_index)
    face_ids = np.full(len(target_unit), -1, dtype=np.int64)
    source_vertex_ids = np.full((len(target_unit), 3), -1, dtype=np.int64)
    weights = np.full((len(target_unit), 3), np.nan, dtype=float)
    residuals = np.full(len(target_unit), np.nan, dtype=float)
    direction_errors = np.full(len(target_unit), np.nan, dtype=float)
    intersections = np.full((len(target_unit), 3), np.nan, dtype=float)
    methods = np.full(len(target_unit), "failed", dtype=object)
    fallback = np.zeros(len(target_unit), dtype=bool)

    for index, direction in enumerate(target_unit):
        if index == 0 or (index + 1) % 5000 == 0:
            logging.info(
                "Ray mapping progress: %d/%d target vertices",
                index + 1,
                len(target_unit),
            )
        best = None
        method = "local_ray_intersection"
        for neighbors in (8, 32, 128, 512):
            _, nearest = tree.query(direction, k=min(neighbors, len(source_unit)))
            candidate_ids = sorted(
                {
                    face
                    for vertex in np.atleast_1d(nearest)
                    for face in incident_faces[int(vertex)]
                }
            )
            for face_id in candidate_ids:
                result = ray_triangle_intersection(
                    direction, source_unit[faces[face_id]]
                )
                if result is not None:
                    if best is None or result[2] < best[3]:
                        best = (face_id, result[0], result[1], result[2])
                    method = (
                        "local_ray_intersection"
                        if neighbors == 8
                        else "expanded_ray_intersection"
                    )
            if best is not None:
                break

        if best is None:
            for face_id, face in enumerate(faces):
                result = ray_triangle_intersection(direction, source_unit[face])
                if result is not None and (best is None or result[2] < best[3]):
                    best = (face_id, result[0], result[1], result[2])
            if best is not None:
                method = "full_mesh_ray_intersection"
                fallback[index] = True

        if best is None:
            continue

        face_id, intersection, current_weights, _ = best
        face_ids[index] = face_id
        source_vertex_ids[index] = faces[face_id]
        weights[index] = current_weights
        intersections[index] = intersection
        reconstructed = current_weights @ source_unit[faces[face_id]]
        residuals[index] = np.linalg.norm(reconstructed - intersection)
        reconstructed_direction = reconstructed / np.linalg.norm(reconstructed)
        direction_errors[index] = np.linalg.norm(reconstructed_direction - direction)
        methods[index] = method

    return (
        face_ids,
        source_vertex_ids,
        weights,
        residuals,
        direction_errors,
        intersections,
        methods,
        fallback,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--metric-image", type=Path, required=True)
    parser.add_argument("--reg", type=Path, required=True)
    parser.add_argument(
        "--template", choices=["fsaverage", "fsaverage6"], required=True
    )
    parser.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.time()
    depths = np.asarray([float(x) for x in args.depths.split(",")])
    native_dir = args.subjects_dir / args.subject_id / "surf"
    template_dir = args.subjects_dir / args.template / "surf"
    hemi = args.hemisphere
    native_sphere, native_faces = nib.freesurfer.io.read_geometry(
        str(native_dir / f"{hemi}.sphere.reg")
    )
    white, _ = nib.freesurfer.io.read_geometry(str(native_dir / f"{hemi}.white"))
    pial, _ = nib.freesurfer.io.read_geometry(str(native_dir / f"{hemi}.pial"))
    target_sphere_path = template_dir / f"{hemi}.sphere.reg"
    if not target_sphere_path.exists():
        target_sphere_path = template_dir / f"{hemi}.sphere"
    target_sphere, _ = nib.freesurfer.io.read_geometry(str(target_sphere_path))
    native_unit, native_sphere_qc = normalize_sphere(native_sphere)
    target_unit, target_sphere_qc = normalize_sphere(target_sphere)
    target_vertices = np.arange(len(target_sphere), dtype=np.int64)
    (
        face_ids,
        source_vertex_ids,
        weights,
        residuals,
        direction_errors,
        intersections_unit,
        mapping_methods,
        fallback_used,
    ) = map_target_to_source(
        target_unit,
        native_unit,
        native_faces,
    )
    valid_mapping = face_ids >= 0
    intersections = (
        intersections_unit * native_sphere_qc["radius_median"]
        + native_sphere_qc["center"]
    )
    intersections[~valid_mapping] = np.nan
    white_anchor = np.full((len(target_vertices), 3), np.nan)
    pial_anchor = np.full((len(target_vertices), 3), np.nan)
    white_anchor[valid_mapping] = np.einsum(
        "ij,ijk->ik", weights[valid_mapping], white[source_vertex_ids[valid_mapping]]
    )
    pial_anchor[valid_mapping] = np.einsum(
        "ij,ijk->ik", weights[valid_mapping], pial[source_vertex_ids[valid_mapping]]
    )
    native_coordinates = (
        white_anchor[:, None, :] * (1.0 - depths[None, :, None])
        + pial_anchor[:, None, :] * depths[None, :, None]
    )
    transform = load_transform_matrix(args.reg)
    shape, _, t_mov = load_dwi_geometry(args.metric_image)
    dwi_voxel = (
        transform_points_to_dwi(native_coordinates.reshape(-1, 3), transform, t_mov)
        .T[:, :3]
        .reshape(native_coordinates.shape)
    )
    # The legacy native-column sampler stores transformed DWI coordinates in
    # 1-based indexing and converts them to 0-based coordinates immediately
    # before scipy.ndimage.map_coordinates.  Reproduce that convention here.
    dwi_voxel -= 1.0
    coordinate_in_bounds = valid_mapping[:, None] & np.all(
        (dwi_voxel >= 0) & (dwi_voxel <= np.asarray(shape[:3]) - 1), axis=2
    )
    image = nib.load(str(args.metric_image)).get_fdata(dtype=np.float64)
    if image.ndim == 4:
        if image.shape[3] != 1:
            raise ValueError(f"Expected a 3D MD image, found shape {image.shape}")
        image = image[..., 0]
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D MD image, found shape {image.shape}")
    profiles = np.full((len(target_vertices), len(depths)), np.nan, dtype=float)
    for depth_index in range(len(depths)):
        coordinates = dwi_voxel[:, depth_index, :]
        valid = valid_mapping
        profiles[valid, depth_index] = map_coordinates(
            image,
            coordinates[valid].T,
            order=SAMPLING_ORDER,
            mode=SAMPLING_MODE,
        )
    finite_mask = np.isfinite(profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target_vertex_id=target_vertices,
        depths=depths,
        md_profiles=profiles,
        native_coordinates=native_coordinates,
        dwi_voxel_coordinates=dwi_voxel,
        white_anchor=white_anchor,
        pial_anchor=pial_anchor,
        source_face_id=face_ids,
        source_vertex_ids=source_vertex_ids,
        barycentric_weights=weights,
        ray_intersection_point=intersections,
        ray_intersection_point_unit=intersections_unit,
        mapping_valid=valid_mapping,
        mapping_method=mapping_methods,
        fallback_used=fallback_used,
        mapping_error=residuals,
        target_direction_error=direction_errors,
        coordinate_in_bounds_mask=coordinate_in_bounds,
        raw_in_bounds_mask=coordinate_in_bounds,
        finite_mask=finite_mask,
        sampled_finite_mask=finite_mask,
        source_sphere_center=native_sphere_qc["center"],
        source_sphere_radius=native_sphere_qc["radius_median"],
        target_sphere_center=target_sphere_qc["center"],
        target_sphere_radius=target_sphere_qc["radius_median"],
        target_sphere_path=str(target_sphere_path),
        metric_image=str(args.metric_image),
        registration=str(args.reg),
        depth_fractions=depths,
        sampling_order=SAMPLING_ORDER,
        sampling_mode=SAMPLING_MODE,
        coordinate_indexing="0-based after subtracting 1 from transform output",
        method=METHOD_NAME,
        target_name=args.template,
        hemisphere=hemi,
        subject_id=args.subject_id,
    )
    valid_weights = weights[valid_mapping]
    valid_residuals = residuals[valid_mapping]
    valid_direction_errors = direction_errors[valid_mapping]
    qc = {
        "subject_id": args.subject_id,
        "template": args.template,
        "hemisphere": hemi,
        "method": METHOD_NAME,
        "target_sphere_path": str(target_sphere_path),
        "metric_image": str(args.metric_image),
        "registration": str(args.reg),
        "depth_fractions": depths.tolist(),
        "n_target_vertices": len(target_vertices),
        "n_valid_mapping": int(valid_mapping.sum()),
        "n_failed_mapping": int((~valid_mapping).sum()),
        "n_out_of_range_weights": int(
            np.any(
                (valid_weights < -1e-6) | (valid_weights > 1 + 1e-6),
                axis=1,
            ).sum()
        ),
        "n_fallback": int(fallback_used.sum()),
        "valid_mapping_percentage": float(valid_mapping.mean() * 100),
        "fallback_percentage": float(fallback_used.mean() * 100),
        "source_sphere_center": native_sphere_qc["center"].tolist(),
        "source_sphere_radius_median": native_sphere_qc["radius_median"],
        "source_sphere_radius_min": native_sphere_qc["radius_min"],
        "source_sphere_radius_max": native_sphere_qc["radius_max"],
        "target_sphere_center": target_sphere_qc["center"].tolist(),
        "target_sphere_radius_median": target_sphere_qc["radius_median"],
        "target_sphere_radius_min": target_sphere_qc["radius_min"],
        "target_sphere_radius_max": target_sphere_qc["radius_max"],
        "barycentric_sum_error_median": float(
            np.median(np.abs(valid_weights.sum(axis=1) - 1.0))
        ),
        "barycentric_sum_error_max": float(
            np.max(np.abs(valid_weights.sum(axis=1) - 1.0))
        ),
        "mapping_error_median": float(np.median(valid_residuals)),
        "mapping_error_95th_percentile": float(np.percentile(valid_residuals, 95)),
        "mapping_error_max": float(np.max(valid_residuals)),
        "target_direction_error_median": float(np.median(valid_direction_errors)),
        "target_direction_error_max": float(np.max(valid_direction_errors)),
        "column_length_median_mm": float(
            np.nanmedian(np.linalg.norm(pial_anchor - white_anchor, axis=1))
        ),
        "raw_in_bounds_percentage": float(coordinate_in_bounds.mean() * 100),
        "sampled_finite_percentage": float(finite_mask.mean() * 100),
        "coordinate_in_bounds_pct": float(coordinate_in_bounds.mean() * 100),
        "sampling_order": SAMPLING_ORDER,
        "sampling_mode": SAMPLING_MODE,
        "coordinate_indexing": "0-based after subtracting 1 from transform output",
        "complete_finite_profile_pct": float(finite_mask.all(axis=1).mean() * 100),
        "elapsed_s": time.time() - started,
    }
    args.output.with_suffix(".qc.json").write_text(json.dumps(qc, indent=2))
    logging.info("Saved option3 output %s", args.output)


if __name__ == "__main__":
    main()
