#!/usr/bin/env python3
"""Export Option 3 cortical columns and MD sampling markers for Freeview."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


N_DEPTHS = 21

OCTAHEDRON_VERTICES = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
])

OCTAHEDRON_FACES = np.array([
    [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
    [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
], dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns-npz", type=Path, required=True)
    parser.add_argument("--md-profiles-npz", type=Path, required=True)
    parser.add_argument("--reference-t1", type=Path, required=True)
    parser.add_argument("--white-surface", type=Path, required=True)
    parser.add_argument("--pial-surface", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--region", help="Template DK label, e.g. bankssts")
    selection.add_argument("--vertex-ids", type=int, nargs="+")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--main-vertex-id", type=int)
    parser.add_argument("--markers", choices=("main", "all"), default="main")
    parser.add_argument("--radius-mm", type=float, default=0.12)
    parser.add_argument("--sides", type=int, default=6)
    parser.add_argument("--point-radius-mm", type=float, default=0.10)

    args = parser.parse_args()
    if args.fraction is not None:
        if args.region is None or not 0 < args.fraction <= 1:
            parser.error("--fraction requires --region and a value in (0, 1]")
    for radius in (args.radius_mm, args.point_radius_mm):
        if not np.isfinite(radius) or radius <= 0:
            parser.error("Both radii must be positive and finite")
    if args.sides < 3:
        parser.error("--sides must be at least 3")
    return args


def vertex_ids(data, n_rows, source):
    ids = data["fsaverage_vertex_id"]
    if (
        ids.shape != (n_rows,)
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or np.unique(ids).size != n_rows
    ):
        raise ValueError(f"Invalid fsaverage_vertex_id in {source}")
    return ids


def read_inputs(args):
    with (
        np.load(args.columns_npz, allow_pickle=False) as columns,
        np.load(args.md_profiles_npz, allow_pickle=False) as metric,
    ):
        for key in ("subject_id", "hemisphere", "template", "depth_direction"):
            if columns[key].item() != metric[key].item():
                raise ValueError(f"NPZ files disagree on {key}")

        if columns["depth_direction"].item() != "pial_to_white":
            raise ValueError("Expected Option 3 pial-to-white depth order")
        expected_depths = np.linspace(0, 1, N_DEPTHS)
        for data in (columns, metric):
            depths = data["depth_fractions"]
            if depths.shape != expected_depths.shape or not np.allclose(
                depths, expected_depths
            ):
                raise ValueError("Expected 21 equally spaced depths from 0 to 1")

        hemisphere = columns["hemisphere"].item()
        for path in (args.white_surface, args.pial_surface):
            if not path.name.startswith(f"{hemisphere}."):
                raise ValueError(f"Expected a {hemisphere} surface: {path}")

        points = columns["column_points"]
        profiles = metric["profiles"]
        if points.ndim != 3 or points.shape[1:] != (N_DEPTHS, 3):
            raise ValueError(f"Unexpected column_points shape: {points.shape}")
        if profiles.ndim != 2 or profiles.shape[1] != N_DEPTHS:
            raise ValueError(f"Unexpected profiles shape: {profiles.shape}")

        column_ids = vertex_ids(columns, len(points), args.columns_npz)
        profile_ids = vertex_ids(metric, len(profiles), args.md_profiles_npz)
        ids, column_rows, profile_rows = np.intersect1d(
            column_ids, profile_ids, assume_unique=True, return_indices=True
        )

        points = points[column_rows]
        profiles = profiles[profile_rows]
        labels = columns["template_label_name"][column_rows].astype(str)
        valid = (
            columns["in_analysis_mask"][column_rows].astype(bool)
            & columns["column_valid"][column_rows].astype(bool)
            & metric["sampling_valid"][profile_rows].astype(bool)
            & np.isfinite(points).all(axis=(1, 2))
            & np.isfinite(profiles).all(axis=1)
            & (profiles != 0).all(axis=1)
            & (np.linalg.norm(points[:, -1] - points[:, 0], axis=1) > 0)
        )

        metadata = {
            key: columns[key].item()
            for key in ("subject_id", "hemisphere", "template", "depth_direction")
        }
        metadata["metric"] = metric["metric_name"].item()

    return ids, points, profiles, labels, valid, metadata


def select_rows(args, ids, labels, valid):
    if args.vertex_ids is not None:
        valid_rows = {int(ids[row]): row for row in np.flatnonzero(valid)}
        missing = sorted(set(args.vertex_ids) - valid_rows.keys())
        if missing:
            raise ValueError(f"Requested IDs missing or invalid: {missing}")
        rows = np.array([valid_rows[vid] for vid in dict.fromkeys(args.vertex_ids)])
        return rows, None

    eligible = np.flatnonzero(valid & (labels == args.region))
    if not eligible.size:
        raise ValueError(f"No valid matched columns for region {args.region!r}")

    fraction = 0.40 if args.fraction is None else args.fraction
    count = max(1, round(len(eligible) * fraction))

    # Sample evenly in sorted template ID order, as in the original ROI script.
    positions = np.linspace(0, len(eligible) - 1, count, dtype=int)
    rows = eligible[positions]

    if args.main_vertex_id is not None:
        match = eligible[ids[eligible] == args.main_vertex_id]
        if not match.size:
            raise ValueError("--main-vertex-id must be valid and inside the ROI")
        if match[0] not in rows:
            nearest = np.argmin(np.abs(rows - match[0]))
            rows[nearest] = match[0]
            rows.sort()

    return rows, int(len(eligible))


def tube_mesh(points, radius, sides):
    # Option 3 columns are straight, so each tube uses one frame at all depths.
    tangent = points[:, -1] - points[:, 0]
    tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
    reference = np.eye(3)[np.argmin(np.abs(tangent), axis=1)]
    u = np.cross(tangent, reference)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tangent, u)

    angles = 2 * np.pi * np.arange(sides) / sides
    offsets = radius * (
        u[:, None, :] * np.cos(angles)[None, :, None]
        + v[:, None, :] * np.sin(angles)[None, :, None]
    )
    vertices = (points[:, :, None, :] + offsets[:, None, :, :]).reshape(-1, 3)

    rings = np.arange(N_DEPTHS - 1)[:, None] * sides
    side = np.arange(sides)[None, :]
    a = rings + side
    b = rings + (side + 1) % sides
    c = a + sides
    d = b + sides
    local_faces = np.stack((a, b, c, b, d, c), axis=-1).reshape(-1, 3)
    bases = np.arange(len(points), dtype=np.int64) * N_DEPTHS * sides
    faces = (local_faces[None, :, :] + bases[:, None, None]).reshape(-1, 3)
    return vertices, faces


def marker_mesh(points, radius):
    centers = points.reshape(-1, 3)
    vertices = centers[:, None, :] + radius * OCTAHEDRON_VERTICES[None, :, :]
    bases = np.arange(len(centers), dtype=np.int64) * len(OCTAHEDRON_VERTICES)
    faces = OCTAHEDRON_FACES[None, :, :] + bases[:, None, None]
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


def main():
    args = parse_args()
    for path in (args.reference_t1, args.white_surface, args.pial_surface):
        if not path.is_file():
            raise FileNotFoundError(path)

    ids, points, profiles, labels, valid, metadata = read_inputs(args)
    rows, eligible_count = select_rows(args, ids, labels, valid)

    main_id = args.main_vertex_id
    if main_id is None:
        main_id = int(ids[rows[len(rows) // 2 if args.region else 0]])
    main_rows = rows[ids[rows] == main_id]
    if not main_rows.size:
        raise ValueError("--main-vertex-id must be among the selected IDs")
    marker_rows = rows if args.markers == "all" else main_rows

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tube_path = output / "columns.surf"
    point_path = output / "depth_points.surf"
    overlay_path = output / "depth_points_metric.curv"

    vertices, faces = tube_mesh(points[rows], args.radius_mm, args.sides)
    fsio.write_geometry(
        str(tube_path), vertices.astype(np.float32), faces.astype(np.int32)
    )

    vertices, faces = marker_mesh(points[marker_rows], args.point_radius_mm)
    fsio.write_geometry(
        str(point_path), vertices.astype(np.float32), faces.astype(np.int32)
    )
    overlay = np.repeat(profiles[marker_rows].ravel(), len(OCTAHEDRON_VERTICES))
    fsio.write_morph_data(str(overlay_path), overlay.astype(np.float32))

    command = shlex.join([
        "freeview",
        "-v", str(args.reference_t1.resolve()),
        "-f",
        f"{args.white_surface.resolve()}:edgecolor=white",
        f"{args.pial_surface.resolve()}:edgecolor=cyan",
        f"{tube_path}:color=0,114,178",
        f"{point_path}:overlay={overlay_path}",
    ])
    (output / "freeview_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n",
        encoding="utf-8",
    )

    metadata.update({
        "coordinate_space": "FreeSurfer native surface RAS (tkReg RAS)",
        "reference_t1": str(args.reference_t1.resolve()),
        "white_surface": str(args.white_surface.resolve()),
        "pial_surface": str(args.pial_surface.resolve()),
        "region": args.region,
        "eligible_region_columns": eligible_count,
        "displayed_columns": len(rows),
        "display_fraction": len(rows) / eligible_count if eligible_count else None,
        "template_vertex_ids": ids[rows].tolist(),
        "main_template_vertex_id": main_id,
        "marker_template_vertex_ids": ids[marker_rows].tolist(),
        "points_per_column": N_DEPTHS,
        "tube_radius_mm": args.radius_mm,
        "tube_sides": args.sides,
        "point_radius_mm": args.point_radius_mm,
        "outputs": [str(tube_path), str(point_path), str(overlay_path)],
    })
    (output / "freeview_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(rows)} tubes and {len(marker_rows) * N_DEPTHS} markers")
    print(command)


if __name__ == "__main__":
    main()