#!/usr/bin/env python3
"""Compare fsaverage and fsaverage6 using existing O3-Coordinate profiles.

This script does not run surface mapping or sample an MD volume. It reads the
already generated O3 profiles, builds one fixed fsaverage-to-fsaverage6
correspondence from the registration spheres, and calculates resolution and
surface-area-matched spatial-group summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict, deque
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


DEPTHS = np.linspace(0.0, 1.0, 21)
TEMPLATES = ("fsaverage", "fsaverage6")
HEMISPHERES = ("lh", "rh")
GROUP_SIZES = (1, 50, 100, 200, 400)
LOGGER = logging.getLogger(__name__)


def profile_path(
    o3_root: Path,
    subject: str,
    template: str,
    hemisphere: str,
) -> Path:
    """Return the existing O3 profile path for one subject and hemisphere."""

    filename = f"{subject}_{hemisphere}_profiles.npz"
    return o3_root / subject / template / "o3_coordinate" / filename


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load MD profiles and a strict 21-depth validity mask."""

    with np.load(path, allow_pickle=False) as data:
        profiles = np.asarray(data["md_profiles"], dtype=np.float64)
        mapping_valid = np.asarray(data["mapping_valid"], dtype=bool)

    valid = mapping_valid & np.isfinite(profiles).all(axis=1)
    return profiles, valid


def load_surface(
    subjects_dir: Path,
    template: str,
    hemisphere: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load white-surface coordinates, registration sphere, and faces."""

    surf_dir = subjects_dir / template / "surf"
    white, faces = nib.freesurfer.io.read_geometry(
        str(surf_dir / f"{hemisphere}.white")
    )
    sphere, sphere_faces = nib.freesurfer.io.read_geometry(
        str(surf_dir / f"{hemisphere}.sphere.reg")
    )
    if not np.array_equal(faces, sphere_faces):
        raise ValueError(f"White and sphere.reg faces differ: {template} {hemisphere}")
    return white.astype(np.float64), sphere.astype(np.float64), faces.astype(np.int64)


def normalize_sphere(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center and normalize a registration sphere to unit directions."""

    center = np.median(coords, axis=0)
    radii = np.linalg.norm(coords - center, axis=1)
    radius = float(np.median(radii))
    if radius <= 0:
        raise ValueError("Invalid registration sphere radius")
    directions = (coords - center) / radii[:, None]
    return directions, center, radius


def build_correspondence(
    source_sphere: np.ndarray,
    target_sphere: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Map each fsaverage vertex to its nearest fsaverage6 sphere direction."""

    source_unit, source_center, source_radius = normalize_sphere(source_sphere)
    target_unit, target_center, target_radius = normalize_sphere(target_sphere)
    tree = cKDTree(target_unit)
    distances, target_ids = tree.query(source_unit, k=1)
    metadata = {
        "source_sphere_radius": source_radius,
        "target_sphere_radius": target_radius,
        "source_center_x": float(source_center[0]),
        "source_center_y": float(source_center[1]),
        "source_center_z": float(source_center[2]),
        "target_center_x": float(target_center[0]),
        "target_center_y": float(target_center[1]),
        "target_center_z": float(target_center[2]),
        "nearest_direction_distance_median": float(np.median(distances)),
        "nearest_direction_distance_95th": float(np.percentile(distances, 95)),
        "nearest_direction_distance_max": float(np.max(distances)),
    }
    return target_ids.astype(np.int64), distances.astype(np.float64), metadata


def make_adjacency(faces: np.ndarray, n_vertices: int) -> csr_matrix:
    """Build a symmetric sparse vertex adjacency matrix from triangular faces."""

    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    reverse = edges[:, ::-1]
    edges = np.concatenate((edges, reverse), axis=0)
    data = np.ones(edges.shape[0], dtype=np.uint8)
    return csr_matrix(
        (data, (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices),
    )


def vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Assign one third of each triangle area to each incident vertex."""

    triangles = coords[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    output = np.zeros(coords.shape[0], dtype=np.float64)
    np.add.at(output, faces.ravel(), np.repeat(areas / 3.0, 3))
    return output


def build_area_groups(
    adjacency: csr_matrix,
    areas: np.ndarray,
    target_area: float,
    exact_singletons: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic connected groups until a target area is reached."""

    if exact_singletons:
        return np.arange(areas.size, dtype=np.int32), areas.copy()

    labels = np.full(areas.size, -1, dtype=np.int32)
    group_areas: list[float] = []
    next_group = 0
    for seed in range(areas.size):
        if labels[seed] >= 0:
            continue
        queue = deque([seed])
        group_area = 0.0
        while queue and (group_area < target_area or group_area == 0.0):
            vertex = queue.popleft()
            if labels[vertex] >= 0:
                continue
            labels[vertex] = next_group
            group_area += areas[vertex]
            start = adjacency.indptr[vertex]
            stop = adjacency.indptr[vertex + 1]
            for neighbor in adjacency.indices[start:stop]:
                if labels[neighbor] < 0:
                    queue.append(int(neighbor))
        group_areas.append(group_area)
        next_group += 1
    return labels, np.asarray(group_areas, dtype=np.float64)


def aggregate_profiles(
    profiles: np.ndarray,
    valid: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate one mean 21-point profile per spatial group."""

    keep = valid & (group_ids >= 0)
    group_count = int(group_ids.max()) + 1
    output = np.full((group_count, profiles.shape[1]), np.nan, dtype=np.float64)
    counts = np.bincount(group_ids[keep], minlength=group_count).astype(float)
    for depth_index in range(profiles.shape[1]):
        sums = np.bincount(
            group_ids[keep],
            weights=profiles[keep, depth_index],
            minlength=group_count,
        )
        output[:, depth_index] = np.divide(
            sums,
            counts,
            out=np.full(group_count, np.nan),
            where=counts > 0,
        )
    return output, counts


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation for two one-dimensional arrays."""

    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calculate_cell_metrics(
    source_profiles: np.ndarray,
    source_valid: np.ndarray,
    target_profiles: np.ndarray,
    target_valid: np.ndarray,
    source_to_target: np.ndarray,
) -> list[dict[str, object]]:
    """Summarize fsaverage within-cell variation and fsaverage6 agreement."""

    rows: list[dict[str, object]] = []
    source_cell_means, source_cell_counts = aggregate_profiles(
        source_profiles,
        source_valid,
        source_to_target,
    )
    for depth_index, depth in enumerate(DEPTHS):
        source_values = source_profiles[:, depth_index]
        source_keep = source_valid
        cell_ids = source_to_target[source_keep]
        values = source_values[source_keep]
        sums = np.bincount(
            cell_ids,
            weights=values,
            minlength=source_cell_means.shape[0],
        )
        sums_sq = np.bincount(
            cell_ids,
            weights=values * values,
            minlength=source_cell_means.shape[0],
        )
        counts = source_cell_counts
        variances = np.divide(
            sums_sq,
            counts,
            out=np.full(counts.shape, np.nan),
            where=counts > 0,
        ) - source_cell_means[:, depth_index] ** 2
        within_sd = np.sqrt(np.maximum(variances, 0.0))
        cell_keep = np.isfinite(within_sd) & (counts >= 2)
        cell_means = source_cell_means[:, depth_index]
        between_keep = np.isfinite(cell_means) & (counts > 0)
        target_keep = target_valid & np.isfinite(cell_means)
        difference = target_profiles[target_keep, depth_index] - cell_means[target_keep]
        rows.append(
            {
                "depth": float(depth),
                "n_source_vertices": int(source_keep.sum()),
                "n_target_cells": int(between_keep.sum()),
                "n_cells_with_2plus_source_vertices": int(cell_keep.sum()),
                "within_cell_sd_median": float(np.nanmedian(within_sd[cell_keep])),
                "between_cell_sd": float(np.nanstd(cell_means[between_keep], ddof=1)),
                "within_between_sd_ratio": float(
                    np.nanmedian(within_sd[cell_keep])
                    / np.nanstd(cell_means[between_keep], ddof=1)
                    if np.nanstd(cell_means[between_keep], ddof=1) > 0
                    else np.nan
                ),
                "target_vs_cellmean_correlation": correlation(
                    target_profiles[target_keep, depth_index],
                    cell_means[target_keep],
                ),
                "target_vs_cellmean_mae": float(np.mean(np.abs(difference))),
                "target_vs_cellmean_rmse": float(np.sqrt(np.mean(difference**2))),
            }
        )
    return rows


def calculate_adjacent_metrics(
    profiles: np.ndarray,
    valid: np.ndarray,
    faces: np.ndarray,
    source_to_target: np.ndarray,
) -> list[dict[str, object]]:
    """Compare adjacent fsaverage profiles within and across mapped cells."""

    edges = np.unique(
        np.sort(
            np.concatenate(
                (
                    faces[:, [0, 1]],
                    faces[:, [1, 2]],
                    faces[:, [2, 0]],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    keep = valid[edges[:, 0]] & valid[edges[:, 1]]
    edges = edges[keep]
    rows: list[dict[str, object]] = []
    for relation, relation_keep in (
        (
            "within_cell",
            source_to_target[edges[:, 0]] == source_to_target[edges[:, 1]],
        ),
        (
            "between_cells",
            source_to_target[edges[:, 0]] != source_to_target[edges[:, 1]],
        ),
    ):
        selected = edges[relation_keep]
        if selected.size == 0:
            continue
        left = profiles[selected[:, 0]]
        right = profiles[selected[:, 1]]
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
            right_centered,
            axis=1,
        )
        correlations = np.divide(
            np.sum(left_centered * right_centered, axis=1),
            denominator,
            out=np.full(denominator.shape, np.nan),
            where=denominator > 0,
        )
        absolute_errors = np.mean(np.abs(left - right), axis=1)
        rows.append(
            {
                "relation": relation,
                "n_edges": int(selected.shape[0]),
                "profile_correlation_median": float(np.nanmedian(correlations)),
                "profile_mae_median": float(np.nanmedian(absolute_errors)),
            }
        )
    return rows


def discover_subjects(o3_root: Path) -> list[str]:
    """Find subjects with complete O3 template and hemisphere outputs."""

    subjects = []
    for subject_dir in sorted(path for path in o3_root.iterdir() if path.is_dir()):
        complete = all(
            profile_path(o3_root, subject_dir.name, template, hemisphere).is_file()
            for template in TEMPLATES
            for hemisphere in HEMISPHERES
        )
        if complete:
            subjects.append(subject_dir.name)
    return subjects


def save_correspondence(
    output_root: Path,
    hemisphere: str,
    source_to_target: np.ndarray,
    nearest_distances: np.ndarray,
    metadata: dict[str, float],
) -> None:
    """Save fixed correspondence arrays and geometry metadata."""

    output_root.joinpath("correspondence").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "correspondence" / f"fsaverage_to_fsaverage6_{hemisphere}.npz",
        source_to_target=source_to_target,
        nearest_direction_distance=nearest_distances,
        metadata_json=json.dumps(metadata),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--o3-root",
        type=Path,
        default=Path(
            "/mnt/newStor/paros/paros_WORK/xintong/results/"
            "o3_coordinate_profiles"
        ),
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=Path("/home/apps/freesurfer/subjects"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=Path("manifests/method_comparison_common_30_subjects.csv"),
    )
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.subjects_file.suffix.lower() == ".csv":
        with args.subjects_file.open(newline="") as stream:
            subjects = [row["subject_id"] for row in csv.DictReader(stream)]
    else:
        subjects = discover_subjects(args.o3_root)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    if not subjects:
        raise RuntimeError("No complete subjects found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    surfaces = {}
    correspondence = {}
    correspondence_rows = []
    for hemisphere in HEMISPHERES:
        for template in TEMPLATES:
            surfaces[(template, hemisphere)] = load_surface(
                args.subjects_dir,
                template,
                hemisphere,
            )
        source_to_target, nearest_distances, metadata = build_correspondence(
            surfaces[("fsaverage", hemisphere)][1],
            surfaces[("fsaverage6", hemisphere)][1],
        )
        correspondence[hemisphere] = source_to_target
        save_correspondence(
            args.output_root,
            hemisphere,
            source_to_target,
            nearest_distances,
            metadata,
        )
        correspondence_rows.append(
            {
                "hemisphere": hemisphere,
                "n_fsaverage_vertices": int(source_to_target.size),
                "n_fsaverage6_vertices": int(
                    surfaces[("fsaverage6", hemisphere)][0].shape[0]
                ),
                **metadata,
            }
        )

    group_memberships = {}
    source_area_median = {}
    for template in TEMPLATES:
        for hemisphere in HEMISPHERES:
            white, _, faces = surfaces[(template, hemisphere)]
            adjacency = make_adjacency(faces, white.shape[0])
            areas = vertex_areas(white, faces)
            source_white, _, source_faces = surfaces[("fsaverage", hemisphere)]
            source_area_median[hemisphere] = float(
                np.median(vertex_areas(source_white, source_faces))
            )
            for requested_size in GROUP_SIZES:
                target_area = requested_size * source_area_median[hemisphere]
                labels, group_areas = build_area_groups(
                    adjacency,
                    areas,
                    target_area,
                    exact_singletons=requested_size == 1,
                )
                group_memberships[(template, hemisphere, requested_size)] = (
                    labels,
                    group_areas,
                )

    cell_rows = []
    adjacent_rows = []
    group_rows = []
    group_profiles = defaultdict(list)
    processed_subjects = []
    for subject_index, subject in enumerate(subjects, start=1):
        LOGGER.info(
            "PROGRESS subject %d/%d: %s",
            subject_index,
            len(subjects),
            subject,
        )
        subject_complete = True
        for hemisphere in HEMISPHERES:
            source_path = profile_path(args.o3_root, subject, "fsaverage", hemisphere)
            target_path = profile_path(args.o3_root, subject, "fsaverage6", hemisphere)
            try:
                source_profiles, source_valid = load_profile(source_path)
                target_profiles, target_valid = load_profile(target_path)
            except (OSError, ValueError) as error:
                LOGGER.exception("Failed subject=%s hemisphere=%s", subject, hemisphere)
                subject_complete = False
                continue

            source_to_target = correspondence[hemisphere]
            cell_metrics = calculate_cell_metrics(
                source_profiles,
                source_valid,
                target_profiles,
                target_valid,
                source_to_target,
            )
            for row in cell_metrics:
                cell_rows.append(
                    {"subject": subject, "hemisphere": hemisphere, **row}
                )

            source_faces = surfaces[("fsaverage", hemisphere)][2]
            for row in calculate_adjacent_metrics(
                source_profiles,
                source_valid,
                source_faces,
                source_to_target,
            ):
                adjacent_rows.append(
                    {"subject": subject, "hemisphere": hemisphere, **row}
                )

            for template, profiles, valid in (
                ("fsaverage", source_profiles, source_valid),
                ("fsaverage6", target_profiles, target_valid),
            ):
                for requested_size in GROUP_SIZES:
                    labels, group_areas = group_memberships[
                        (template, hemisphere, requested_size)
                    ]
                    grouped, counts = aggregate_profiles(profiles, valid, labels)
                    group_profiles[
                        (template, hemisphere, requested_size)
                    ].append(grouped)
                    for depth_index, depth in enumerate(DEPTHS):
                        finite = np.isfinite(grouped[:, depth_index])
                        group_rows.append(
                            {
                                "subject": subject,
                                "hemisphere": hemisphere,
                                "template": template,
                                "requested_group_size": requested_size,
                                "depth": float(depth),
                                "n_groups": int(finite.sum()),
                                "median_group_area_mm2": float(np.median(group_areas)),
                                "median_group_vertex_count": float(
                                    np.median(counts[counts > 0])
                                ),
                                "within_template_group_sd": float(
                                    np.nanstd(grouped[finite, depth_index], ddof=1)
                                ),
                            }
                        )
        if subject_complete:
            processed_subjects.append(subject)

    cross_rows = []
    for (template, hemisphere, requested_size), arrays in group_profiles.items():
        stack = np.stack(arrays, axis=0)
        for depth_index, depth in enumerate(DEPTHS):
            values = stack[:, :, depth_index]
            with np.errstate(invalid="ignore"):
                between_subject_sd = np.nanstd(values, axis=0, ddof=1)
            cross_rows.append(
                {
                    "template": template,
                    "hemisphere": hemisphere,
                    "requested_group_size": requested_size,
                    "depth": float(depth),
                    "n_subjects": int(stack.shape[0]),
                    "n_groups": int(np.isfinite(between_subject_sd).sum()),
                    "cross_subject_sd_median": float(np.nanmedian(between_subject_sd)),
                }
            )

    tables_root = args.output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(correspondence_rows).to_csv(
        tables_root / "template_correspondence_qc.csv",
        index=False,
    )
    pd.DataFrame(cell_rows).to_csv(
        tables_root / "template_cell_variation.csv",
        index=False,
    )
    pd.DataFrame(adjacent_rows).to_csv(
        tables_root / "adjacent_profile_similarity.csv",
        index=False,
    )
    pd.DataFrame(group_rows).to_csv(
        tables_root / "area_matched_group_metrics.csv",
        index=False,
    )
    pd.DataFrame(cross_rows).to_csv(
        tables_root / "area_matched_cross_subject_variability.csv",
        index=False,
    )
    metadata = {
        "subjects": processed_subjects,
        "subjects_file": str(args.subjects_file),
        "n_subjects": len(processed_subjects),
        "templates": list(TEMPLATES),
        "hemispheres": list(HEMISPHERES),
        "depths": DEPTHS.tolist(),
        "group_sizes": list(GROUP_SIZES),
        "correspondence": "nearest fsaverage6 sphere.reg unit direction",
        "profile_source": "existing O3-Coordinate md_profiles.npz",
        "volume_sampling_performed": False,
    }
    logs_root = args.output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Completed %d subjects", len(processed_subjects))


if __name__ == "__main__":
    main()
