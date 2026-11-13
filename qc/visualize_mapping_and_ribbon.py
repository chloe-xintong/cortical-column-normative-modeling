#!/usr/bin/env python3
"""Create presentation figures from a real template-to-subject mapping.

Description
-----------
The script intentionally *reads* the mapping recorded by
``build_template_columns.py`` rather than recomputing it.  It produces two
figures for one fsaverage vertex:

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subjects-dir`` command-line input
- ``--subject-id`` command-line input
- ``--columns-npz`` command-line input
- ``--hemisphere`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/visualize_mapping_and_ribbon.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_FACE_LIMIT = 18_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--columns-npz", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument(
        "--vertex-id",
        type=int,
        help="fsaverage target vertex ID; default: first valid cortical column",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg"),
        default=("png", "svg"),
        help="Output formats (default: png svg)",
    )
    parser.add_argument(
        "--face-limit",
        type=int,
        default=DEFAULT_FACE_LIMIT,
        help="Maximum faces drawn for each full sphere (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-rings",
        type=int,
        default=4,
        help="Native mesh neighbourhood rings for the ribbon patch (default: %(default)s)",
    )
    return parser.parse_args()


def scalar(value: np.ndarray | str | bytes) -> str:
    """Convert a scalar saved by np.savez to text."""

    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def load_columns(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Columns file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "mapping_valid",
        "native_triangle_id",
        "native_triangle_vertex_ids",
        "barycentric_weights",
        "column_points",
        "depth_fractions",
        "target_sphere_path",
        "closest_point_on_source_sphere",
        "source_sphere_center",
        "source_sphere_radius",
        "target_sphere_center",
        "target_sphere_radius",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            "This file does not contain the mapping fields needed for the figure: "
            + ", ".join(missing)
        )
    return data


def read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Surface file not found: {path}")
    points, faces = fsio.read_geometry(str(path))
    return np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64)


def normalize_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Sphere radius in columns file must be finite and positive")
    return (points - center) / radius


def choose_vertex(data: dict[str, np.ndarray], requested: int | None) -> int:
    n_columns = len(data["mapping_valid"])
    if requested is not None:
        if not 0 <= requested < n_columns:
            raise ValueError(f"--vertex-id must be in [0, {n_columns - 1}], got {requested}")
        candidate = requested
    else:
        valid = data["mapping_valid"].astype(bool)
        if "column_valid" in data:
            valid &= data["column_valid"].astype(bool)
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError("No valid columns are available in the columns file")
        candidate = int(indices[0])
    if not bool(data["mapping_valid"][candidate]):
        raise ValueError(f"Vertex {candidate} has no valid subject-sphere mapping")
    return candidate


def decimate_faces(faces: np.ndarray, face_limit: int) -> np.ndarray:
    if face_limit < 1:
        raise ValueError("--face-limit must be positive")
    if len(faces) <= face_limit:
        return faces
    step = int(np.ceil(len(faces) / face_limit))
    return faces[::step]


def orient_view(ax, focus: np.ndarray) -> None:
    """Place the selected sphere point near the front of both 3D panels."""

    x, y, z = focus / np.linalg.norm(focus)
    azimuth = np.degrees(np.arctan2(y, x)) - 90.0
    elevation = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    ax.view_init(elev=float(elevation), azim=float(azimuth))


def style_3d_axis(ax) -> None:
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))


def add_sphere_mesh(ax, points: np.ndarray, faces: np.ndarray) -> None:
    ax.plot_trisurf(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        triangles=faces,
        color="#c9ced3",
        edgecolor="#8f969d",
        linewidth=0.08,
        alpha=0.72,
        shade=False,
        antialiased=True,
    )


def add_source_triangle(
    ax,
    triangle: np.ndarray,
    mapped_point: np.ndarray,
    weights: np.ndarray,
    annotate_weights: bool = True,
    marker_lift: float = 0.025,
) -> None:
    polygon = Poly3DCollection(
        [triangle], facecolors="#ffbc42", edgecolors="#9a5d00", linewidths=2.2, alpha=0.88
    )
    ax.add_collection3d(polygon)
    ax.scatter(
        triangle[:, 0], triangle[:, 1], triangle[:, 2],
        s=48, c="#1565c0", edgecolors="white", linewidths=0.8, depthshade=False,
    )
    # A point exactly on an opaque triangle can be hidden by Matplotlib's 3-D
    # depth sorting.  The short dotted tether preserves its exact location;
    # only the marker is lifted radially by 2.5% for legibility.
    visible_point = mapped_point + marker_lift * mapped_point / np.linalg.norm(mapped_point)
    ax.plot(
        [mapped_point[0], visible_point[0]],
        [mapped_point[1], visible_point[1]],
        [mapped_point[2], visible_point[2]],
        color="#d62828", linestyle=":", linewidth=1.4,
    )
    ax.scatter(
        *visible_point, s=86, c="#d62828", edgecolors="white", linewidths=1.2, depthshade=False,
    )
    if annotate_weights:
        triangle_center = triangle.mean(axis=0)
        label_offset = 0.20 * np.max(np.linalg.norm(triangle - triangle_center, axis=1))
        for index, (vertex, weight) in enumerate(zip(triangle, weights), start=1):
            direction = vertex - triangle_center
            direction /= np.linalg.norm(direction)
            ax.text(
                *(vertex + label_offset * direction), f"v{index}\nw{index}={weight:.3f}",
                fontsize=9, color="#12355b", ha="center", va="center",
            )


def plot_sphere_mapping(
    target_sphere: np.ndarray,
    target_faces: np.ndarray,
    source_sphere: np.ndarray,
    source_faces: np.ndarray,
    vertex_id: int,
    triangle_ids: np.ndarray,
    mapped_point: np.ndarray,
    weights: np.ndarray,
    subject_id: str,
    hemisphere: str,
    face_limit: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(17.0, 6.7), constrained_layout=True)
    ax_target = fig.add_subplot(1, 3, 1, projection="3d")
    ax_source = fig.add_subplot(1, 3, 2, projection="3d")
    ax_zoom = fig.add_subplot(1, 3, 3, projection="3d")
    target_point = target_sphere[vertex_id]

    add_sphere_mesh(ax_target, target_sphere, decimate_faces(target_faces, face_limit))
    ax_target.scatter(
        *target_point, s=98, c="#d62828", edgecolors="white", linewidths=1.3, depthshade=False,
    )
    ax_target.set_title(
        f"fsaverage sphere.reg\ntarget vertex {vertex_id}", fontsize=15, fontweight="bold", pad=12
    )
    orient_view(ax_target, target_point)
    style_3d_axis(ax_target)

    add_sphere_mesh(ax_source, source_sphere, decimate_faces(source_faces, face_limit))
    triangle = source_sphere[triangle_ids]
    add_source_triangle(ax_source, triangle, mapped_point, weights, annotate_weights=False)
    ax_source.set_title(
        f"{subject_id} {hemisphere}.sphere.reg\nsource triangle + mapped point", fontsize=15,
        fontweight="bold", pad=12,
    )
    orient_view(ax_source, mapped_point)
    style_3d_axis(ax_source)

    # A close-up keeps the exact barycentric weights legible in slides while
    # the preceding panel retains the selected triangle's global location.
    local_faces = mesh_neighbourhood(source_faces, triangle_ids, rings=2)
    add_sphere_mesh(ax_zoom, source_sphere, local_faces)
    add_source_triangle(ax_zoom, triangle, mapped_point, weights, marker_lift=0.002)
    set_equal_limits(
        ax_zoom,
        np.vstack((source_sphere[np.unique(local_faces)], mapped_point[None, :])),
        pad_fraction=0.45,
        minimum_half_width=0.0,
    )
    ax_zoom.set_title("Mapped source triangle (zoom)", fontsize=15, fontweight="bold", pad=12)
    orient_view(ax_zoom, mapped_point)
    style_3d_axis(ax_zoom)

    fig.suptitle(
        "Template-to-subject surface mapping (recorded pipeline output)",
        fontsize=18, fontweight="bold",
    )
    fig.text(
        0.5, 0.02,
        "red = target/mapped point (source marker is tethered to its exact position)   ·   "
        "blue = source triangle vertices   ·   amber = mapped triangle",
        ha="center", fontsize=10, color="#38434f",
    )
    return fig


def mesh_neighbourhood(faces: np.ndarray, seeds: np.ndarray, rings: int) -> np.ndarray:
    """Return faces within a small topological neighbourhood of seed vertices."""

    if rings < 0:
        raise ValueError("--patch-rings must be non-negative")
    selected_vertices = set(int(vertex) for vertex in seeds)
    selected_faces = np.zeros(len(faces), dtype=bool)
    for _ in range(rings + 1):
        hits = np.any(np.isin(faces, list(selected_vertices)), axis=1)
        selected_faces |= hits
        selected_vertices.update(np.unique(faces[hits]).tolist())
    return faces[selected_faces]


def set_equal_limits(
    ax,
    points: np.ndarray,
    pad_fraction: float = 0.12,
    minimum_half_width: float = 0.5,
) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    half_width = max(float(np.max(high - low)) / 2.0, minimum_half_width) * (1.0 + pad_fraction)
    center = (low + high) / 2.0
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_zlim(center[2] - half_width, center[2] + half_width)
    ax.set_box_aspect((1, 1, 1))


def plot_cortical_ribbon(
    white: np.ndarray,
    pial: np.ndarray,
    patch_faces: np.ndarray,
    column_points: np.ndarray,
    depth_fractions: np.ndarray,
    vertex_id: int,
    subject_id: str,
    hemisphere: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.8, 8.4), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            pial[patch_faces], facecolors="#9ecae1", edgecolors="#4f7488",
            linewidths=0.25, alpha=0.42,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            white[patch_faces], facecolors="#bdbdbd", edgecolors="#5e5e5e",
            linewidths=0.25, alpha=0.58,
        )
    )

    colours = cm.viridis(depth_fractions)
    ax.plot(
        column_points[:, 0], column_points[:, 1], column_points[:, 2],
        color="#272727", linewidth=2.1, zorder=4,
    )
    ax.scatter(
        column_points[:, 0], column_points[:, 1], column_points[:, 2],
        c=colours, s=39, edgecolors="white", linewidths=0.65, depthshade=False, zorder=5,
    )
    ax.text(*column_points[0], "  pial (depth 0.00)", fontsize=9, color="#173f5f")
    ax.text(*column_points[-1], "  white (depth 1.00)", fontsize=9, color="#333333")

    all_points = np.vstack((white[np.unique(patch_faces)], pial[np.unique(patch_faces)], column_points))
    set_equal_limits(ax, all_points)
    ax.view_init(elev=19, azim=-56)
    ax.set_axis_off()
    ax.set_title(
        f"Native cortical ribbon and reconstructed column\n{subject_id} {hemisphere}, fsaverage vertex {vertex_id}",
        fontsize=15, fontweight="bold", pad=13,
    )
    colour_map = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    colour_map.set_array([])
    colour_bar = fig.colorbar(colour_map, ax=ax, shrink=0.58, pad=0.02)
    colour_bar.set_label("Saved depth fraction (pial → white)", fontsize=10)
    colour_bar.set_ticks((0, 0.5, 1))
    return fig


def save_figure(fig: plt.Figure, stem: Path, formats: tuple[str, ...] | list[str]) -> list[Path]:
    saved = []
    for output_format in formats:
        path = stem.with_suffix(f".{output_format}")
        options = {"bbox_inches": "tight", "facecolor": "white"}
        if output_format == "png":
            options["dpi"] = 300
        fig.savefig(path, **options)
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    data = load_columns(args.columns_npz)
    if "subject_id" in data and scalar(data["subject_id"]) != args.subject_id:
        raise ValueError(
            f"--subject-id ({args.subject_id}) does not match columns file "
            f"({scalar(data['subject_id'])})"
        )
    if "hemisphere" in data and scalar(data["hemisphere"]) != args.hemisphere:
        raise ValueError(
            f"--hemisphere ({args.hemisphere}) does not match columns file "
            f"({scalar(data['hemisphere'])})"
        )
    vertex_id = choose_vertex(data, args.vertex_id)
    subject_root = args.subjects_dir / args.subject_id
    source_path = subject_root / "surf" / f"{args.hemisphere}.sphere.reg"
    white_path = subject_root / "surf" / f"{args.hemisphere}.white"
    pial_path = subject_root / "surf" / f"{args.hemisphere}.pial"
    target_path = Path(scalar(data["target_sphere_path"]))

    source_raw, source_faces = read_surface(source_path)
    target_raw, target_faces = read_surface(target_path)
    white, white_faces = read_surface(white_path)
    pial, pial_faces = read_surface(pial_path)
    if not np.array_equal(white_faces, pial_faces):
        raise ValueError("Native white and pial surfaces do not have matching topology")
    if len(source_raw) != len(white) or len(source_raw) != len(pial):
        raise ValueError("sphere.reg, white, and pial surfaces must share vertex indexing")
    if len(target_raw) != len(data["mapping_valid"]):
        raise ValueError(
            "Target sphere vertex count does not match the mapping rows in the columns file"
        )

    source_sphere = normalize_sphere(
        source_raw, np.asarray(data["source_sphere_center"], dtype=float),
        float(data["source_sphere_radius"]),
    )
    target_sphere = normalize_sphere(
        target_raw, np.asarray(data["target_sphere_center"], dtype=float),
        float(data["target_sphere_radius"]),
    )
    triangle_ids = np.asarray(data["native_triangle_vertex_ids"][vertex_id], dtype=np.int64)
    if np.any(triangle_ids < 0) or np.any(triangle_ids >= len(source_sphere)):
        raise ValueError(f"Vertex {vertex_id} contains invalid source triangle vertex IDs")
    weights = np.asarray(data["barycentric_weights"][vertex_id], dtype=float)
    mapped_point = np.asarray(data["closest_point_on_source_sphere"][vertex_id], dtype=float)
    if not np.isfinite(weights).all() or not np.isfinite(mapped_point).all():
        raise ValueError(f"Vertex {vertex_id} has incomplete saved mapping data")
    column_points = np.asarray(data["column_points"][vertex_id], dtype=float)
    depth_fractions = np.asarray(data["depth_fractions"], dtype=float)
    if len(column_points) != len(depth_fractions) or not np.isfinite(column_points).all():
        raise ValueError(f"Vertex {vertex_id} has incomplete saved column points")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    identifier = f"{args.subject_id}_{args.hemisphere}_fsavg-{vertex_id}"
    mapping_figure = plot_sphere_mapping(
        target_sphere, target_faces, source_sphere, source_faces, vertex_id,
        triangle_ids, mapped_point, weights, args.subject_id, args.hemisphere, args.face_limit,
    )
    saved = save_figure(mapping_figure, args.output_dir / f"{identifier}_sphere_mapping", args.formats)
    patch_faces = mesh_neighbourhood(white_faces, triangle_ids, args.patch_rings)
    ribbon_figure = plot_cortical_ribbon(
        white, pial, patch_faces, column_points, depth_fractions, vertex_id,
        args.subject_id, args.hemisphere,
    )
    saved.extend(save_figure(ribbon_figure, args.output_dir / f"{identifier}_cortical_ribbon", args.formats))

    provenance = {
        "subject_id": args.subject_id,
        "hemisphere": args.hemisphere,
        "fsaverage_vertex_id": vertex_id,
        "columns_npz": str(args.columns_npz.resolve()),
        "target_sphere": str(target_path),
        "source_sphere": str(source_path),
        "white_surface": str(white_path),
        "pial_surface": str(pial_path),
        "native_triangle_id": int(data["native_triangle_id"][vertex_id]),
        "native_triangle_vertex_ids": triangle_ids.tolist(),
        "barycentric_weights": weights.tolist(),
        "depth_point_count": int(len(column_points)),
        "outputs": [str(path.resolve()) for path in saved],
    }
    provenance_path = args.output_dir / f"{identifier}_figure_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Selected fsaverage vertex: {vertex_id}")
    print(f"Native triangle ID: {provenance['native_triangle_id']}")
    print("Barycentric weights: " + ", ".join(f"{value:.6f}" for value in weights))
    for path in saved:
        print(f"Saved: {path}")
    print(f"Saved: {provenance_path}")


if __name__ == "__main__":
    main()
