"""
Visualize cortical trajectories and their sampling points for a selected
FreeSurfer cortical region.

Two visualization modes are supported:

1. T1 mode
   - Uses trajectory coordinates in T1 / FreeSurfer space.
   - Displays cortical trajectories and their 21 sampling points.

2. Metric mode
   - Uses the same T1-space coordinates for visualization.
   - Uses the matched DWI-space trajectory coordinates to sample a
     diffusion metric image such as ADC / MD.
   - Removes trajectories containing NaN or zero-valued samples.
   - Writes the sampled metric values as a FreeSurfer surface overlay.

Outputs
-------
T1 mode:
    *_trajectories.surf
    *_points.surf

Metric mode:
    *_trajectories.surf
    *_points.surf
    *_points_<metric>.curv
"""

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize cortical trajectories and sampling points "
            "in T1 space, optionally colored by a diffusion metric."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["t1", "metric"],
        default="t1",
        help=(
            "Visualization mode. "
            "'t1' creates T1-space geometry only. "
            "'metric' additionally samples ADC / MD in DWI space."
        )
    )

    parser.add_argument(
        "--trajectory",
        required=True,
        help="T1-space trajectory .mat file."
    )

    parser.add_argument(
        "--trajectory-dwi",
        default=None,
        help=(
            "Matched DWI-space trajectory .mat file. "
            "Required for --mode metric."
        )
    )

    parser.add_argument(
        "--metric-image",
        default=None,
        help=(
            "3D diffusion metric image, e.g. ADC / MD. "
            "Required for --mode metric."
        )
    )

    parser.add_argument(
        "--metric-name",
        default="adc",
        help="Metric name used in output filenames."
    )

    parser.add_argument(
        "--annot",
        required=True,
        help="FreeSurfer aparc annotation file."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject ID, e.g. D0007."
    )

    parser.add_argument(
        "--hemi",
        required=True,
        choices=["lh", "rh"],
        help="Hemisphere."
    )

    parser.add_argument(
        "--region",
        required=True,
        help="FreeSurfer cortical region name, e.g. bankssts."
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.70,
        help="Fraction of region trajectories to display."
    )

    parser.add_argument(
        "--points-per-trajectory",
        type=int,
        default=21,
        help="Number of sampling points per cortical trajectory."
    )

    parser.add_argument(
        "--tube-radius",
        type=float,
        default=0.12,
        help="Trajectory tube radius."
    )

    parser.add_argument(
        "--tube-sides",
        type=int,
        default=6,
        help="Number of sides in each tube cross-section."
    )

    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.10,
        help="Sampling-point marker radius."
    )

    parser.add_argument(
        "--dwi-coordinate-base",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Index convention used by the saved DWI-space trajectory "
            "coordinates. Use 1 for MATLAB-style 1-based voxel indices."
        )
    )

    return parser.parse_args()


def find_numeric_array(mat_dict, preferred_keys):
    """
    Find the trajectory coordinate array in a MATLAB file.
    """

    for key in preferred_keys:
        if key in mat_dict:
            return key, np.asarray(mat_dict[key])

    candidates = []

    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue

        arr = np.asarray(value)

        if np.issubdtype(arr.dtype, np.number):
            candidates.append(
                (key, arr)
            )

    if len(candidates) == 1:
        return candidates[0]

    available = [
        (key, arr.shape)
        for key, arr in candidates
    ]

    raise KeyError(
        "Could not uniquely identify the trajectory array. "
        f"Available numeric arrays: {available}"
    )


def normalize_xyz_array(arr, name):
    """
    Convert common trajectory coordinate formats to an N x 3 XYZ array.

    Supported shapes:
        N x 3
        3 x N
        4 x N
        N x 4

    For homogeneous 4D coordinates, only XYZ are retained.
    """

    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array, got {arr.shape}."
        )

    if arr.shape[1] == 3:
        xyz = arr

    elif arr.shape[0] == 3:
        xyz = arr.T

    elif arr.shape[0] == 4:
        xyz = arr[:3, :].T

    elif arr.shape[1] == 4:
        xyz = arr[:, :3]

    else:
        raise ValueError(
            f"Cannot interpret {name} shape {arr.shape} "
            "as XYZ coordinates."
        )

    return np.asarray(
        xyz,
        dtype=float
    )


def load_trajectories(
    mat_file,
    hemi,
    points_per_trajectory,
    space
):
    """
    Load trajectory coordinates and reshape to:
        n_trajectories x n_points x 3
    """

    data = loadmat(
        mat_file
    )

    if space == "t1":
        preferred_keys = [
            f"{hemi}_trajectory_points",
            f"{hemi}_column_points",
        ]

    elif space == "dwi":
        preferred_keys = [
            f"{hemi}_trajectory_dwi",
            f"{hemi}_trajectory_points_dwi",
            f"trajectory_{hemi}_dwi",
        ]

    else:
        raise ValueError(
            f"Unknown trajectory space: {space}"
        )

    key, raw = find_numeric_array(
        data,
        preferred_keys
    )

    xyz = normalize_xyz_array(
        raw,
        key
    )

    if (
        xyz.shape[0]
        % points_per_trajectory
        != 0
    ):
        raise ValueError(
            f"{key}: total point count {xyz.shape[0]} "
            f"is not divisible by {points_per_trajectory}."
        )

    trajectories = xyz.reshape(
        -1,
        points_per_trajectory,
        3
    )

    return key, trajectories


def load_region_indices(
    annot_file,
    region_name
):
    """
    Get surface vertex indices belonging to one aparc region.
    """

    labels, ctab, names = fsio.read_annot(
        annot_file
    )

    names = [
        n.decode("utf-8")
        if isinstance(n, bytes)
        else str(n)
        for n in names
    ]

    if region_name not in names:
        raise ValueError(
            f"Region '{region_name}' not found.\n"
            f"Available regions: {names}"
        )

    region_index = names.index(
        region_name
    )

    vertex_indices = np.where(
        labels == region_index
    )[0]

    if len(vertex_indices) == 0:
        raise ValueError(
            f"No vertices found for region '{region_name}'."
        )

    return vertex_indices


def select_region_trajectories(
    trajectories,
    vertex_indices
):
    """
    Select trajectories associated with vertices in the chosen region.
    """

    if np.max(vertex_indices) >= len(
        trajectories
    ):
        raise ValueError(
            "Annotation vertex indices exceed the number "
            "of trajectories. Check that the annotation and "
            "trajectory file belong to the same subject and hemisphere."
        )

    return trajectories[
        vertex_indices
    ]


def uniform_subsample(
    trajectories,
    fraction
):
    """
    Uniformly select a fraction of trajectories.
    """

    if not 0 < fraction <= 1:
        raise ValueError(
            "--fraction must be greater than 0 and <= 1."
        )

    n_show = max(
        1,
        int(
            round(
                len(trajectories)
                * fraction
            )
        )
    )

    indices = np.linspace(
        0,
        len(trajectories) - 1,
        n_show,
        dtype=int
    )

    return indices


def make_frame(direction):
    """
    Construct two unit vectors perpendicular to a trajectory direction.
    """

    norm = np.linalg.norm(
        direction
    )

    if norm == 0:
        direction = np.array(
            [0.0, 0.0, 1.0],
            dtype=float
        )
    else:
        direction = (
            direction / norm
        )

    helper = np.array(
        [1.0, 0.0, 0.0],
        dtype=float
    )

    if abs(
        np.dot(
            direction,
            helper
        )
    ) > 0.9:
        helper = np.array(
            [0.0, 1.0, 0.0],
            dtype=float
        )

    u = np.cross(
        direction,
        helper
    )

    u_norm = np.linalg.norm(
        u
    )

    if u_norm == 0:
        helper = np.array(
            [0.0, 0.0, 1.0],
            dtype=float
        )

        u = np.cross(
            direction,
            helper
        )

        u_norm = np.linalg.norm(
            u
        )

    u = (
        u / u_norm
    )

    v = np.cross(
        direction,
        u
    )

    v = (
        v
        / np.linalg.norm(v)
    )

    return u, v


def build_tube_mesh(
    trajectories,
    points_per_trajectory,
    radius,
    sides
):
    """
    Convert cortical trajectories into thin tube meshes.
    """

    vertices = []
    faces = []

    for trajectory in trajectories:
        base_index = len(
            vertices
        )

        for j in range(
            points_per_trajectory
        ):

            if j == 0:
                direction = (
                    trajectory[1]
                    - trajectory[0]
                )

            elif j == (
                points_per_trajectory - 1
            ):
                direction = (
                    trajectory[-1]
                    - trajectory[-2]
                )

            else:
                direction = (
                    trajectory[j + 1]
                    - trajectory[j - 1]
                )

            u, v = make_frame(
                direction
            )

            for s in range(
                sides
            ):
                theta = (
                    2.0
                    * np.pi
                    * s
                    / sides
                )

                offset = (
                    radius
                    * (
                        np.cos(theta) * u
                        + np.sin(theta) * v
                    )
                )

                vertices.append(
                    trajectory[j]
                    + offset
                )

        for j in range(
            points_per_trajectory - 1
        ):
            ring1 = (
                base_index
                + j * sides
            )

            ring2 = (
                base_index
                + (j + 1) * sides
            )

            for s in range(
                sides
            ):
                s_next = (
                    s + 1
                ) % sides

                a = ring1 + s
                b = ring1 + s_next
                c = ring2 + s
                d = ring2 + s_next

                faces.append(
                    [a, c, b]
                )

                faces.append(
                    [b, c, d]
                )

    return (
        np.asarray(
            vertices,
            dtype=np.float32
        ),
        np.asarray(
            faces,
            dtype=np.int32
        )
    )


def build_point_mesh(
    trajectories,
    radius,
    values=None
):
    """
    Convert sampling coordinates into small octahedral marker surfaces.

    If values are provided, each sampling value is assigned to all six
    vertices belonging to that marker.
    """

    point_vertices = []
    point_faces = []
    overlay_values = []

    octa_offsets = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=float
    )

    octa_faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32
    )

    for trajectory_index, trajectory in enumerate(
        trajectories
    ):

        for point_index, point in enumerate(
            trajectory
        ):
            base_index = len(
                point_vertices
            )

            marker_vertices = (
                point
                + radius
                * octa_offsets
            )

            point_vertices.extend(
                marker_vertices
            )

            point_faces.extend(
                octa_faces
                + base_index
            )

            if values is not None:
                value = values[
                    trajectory_index,
                    point_index
                ]

                overlay_values.extend(
                    [value]
                    * len(octa_offsets)
                )

    point_vertices = np.asarray(
        point_vertices,
        dtype=np.float32
    )

    point_faces = np.asarray(
        point_faces,
        dtype=np.int32
    )

    if values is None:
        overlay_values = None
    else:
        overlay_values = np.asarray(
            overlay_values,
            dtype=np.float32
        )

    return (
        point_vertices,
        point_faces,
        overlay_values
    )


def sample_metric(
    metric_data,
    dwi_trajectories,
    coordinate_base
):
    """
    Sample a 3D metric image at DWI voxel-space trajectory coordinates
    using trilinear interpolation.
    """

    coordinates = np.asarray(
        dwi_trajectories,
        dtype=float
    ).copy()

    if coordinate_base == 1:
        coordinates -= 1.0

    flat_coordinates = coordinates.reshape(
        -1,
        3
    )

    sampled = map_coordinates(
        metric_data,
        [
            flat_coordinates[:, 0],
            flat_coordinates[:, 1],
            flat_coordinates[:, 2],
        ],
        order=1,
        mode="constant",
        cval=np.nan
    )

    return sampled.reshape(
        coordinates.shape[0],
        coordinates.shape[1]
    )


def run_t1_mode(
    args,
    selected_t1,
    output_dir
):
    """
    Create T1-space trajectory and sampling-point surfaces.
    """

    tube_vertices, tube_faces = build_tube_mesh(
        selected_t1,
        args.points_per_trajectory,
        args.tube_radius,
        args.tube_sides
    )

    (
        point_vertices,
        point_faces,
        _
    ) = build_point_mesh(
        selected_t1,
        args.point_radius
    )

    prefix = (
        f"{args.subject}_{args.hemi}_{args.region}_"
        f"T1_{int(args.fraction * 100)}percent"
    )

    tube_out = (
        output_dir
        / f"{prefix}_trajectories.surf"
    )

    point_out = (
        output_dir
        / f"{prefix}_points.surf"
    )

    fsio.write_geometry(
        tube_out,
        tube_vertices,
        tube_faces
    )

    fsio.write_geometry(
        point_out,
        point_vertices,
        point_faces
    )

    print("")
    print("Mode: T1")
    print("Saved trajectory surface:")
    print(tube_out)

    print("")
    print("Saved sampling-point surface:")
    print(point_out)

    print("")
    print(
        "Displayed trajectories:",
        len(selected_t1)
    )

    print(
        "Sampling points per trajectory:",
        args.points_per_trajectory
    )

    print(
        "Total displayed sampling points:",
        len(selected_t1)
        * args.points_per_trajectory
    )


def run_metric_mode(
    args,
    region_t1,
    selection,
    output_dir
):
    """
    Sample a diffusion metric using DWI-space coordinates and create
    metric-colored T1-space sampling-point geometry.
    """

    trajectory_dwi_file = Path(
        args.trajectory_dwi
    )

    metric_file = Path(
        args.metric_image
    )

    if not trajectory_dwi_file.exists():
        raise FileNotFoundError(
            trajectory_dwi_file
        )

    if not metric_file.exists():
        raise FileNotFoundError(
            metric_file
        )

    dwi_key, dwi_trajectories = load_trajectories(
        trajectory_dwi_file,
        args.hemi,
        args.points_per_trajectory,
        space="dwi"
    )

    print("")
    print(
        f"DWI trajectory variable: {dwi_key}"
    )

    print(
        "Whole DWI trajectory shape:",
        dwi_trajectories.shape
    )

    # region_t1 is already region-selected.
    # Apply the same aparc vertex selection to DWI coordinates.
    vertex_indices = load_region_indices(
        Path(args.annot),
        args.region
    )

    if np.max(vertex_indices) >= len(
        dwi_trajectories
    ):
        raise ValueError(
            "Annotation vertex indices exceed the number "
            "of DWI trajectories."
        )

    region_dwi = dwi_trajectories[
        vertex_indices
    ]

    if len(region_t1) != len(region_dwi):
        raise ValueError(
            "T1-space and DWI-space trajectory counts "
            "within the region do not match."
        )

    selected_t1 = region_t1[
        selection
    ]

    selected_dwi = region_dwi[
        selection
    ]

    metric_img = nib.load(
        metric_file
    )

    metric_data = metric_img.get_fdata(
        dtype=np.float32
    )

    if metric_data.ndim != 3:
        raise ValueError(
            f"Metric image must be 3D, got {metric_data.shape}."
        )

    print("")
    print(
        "Metric image:",
        metric_file
    )

    print(
        "Metric image shape:",
        metric_data.shape
    )

    sampled_values = sample_metric(
        metric_data,
        selected_dwi,
        args.dwi_coordinate_base
    )

    print("")
    print(
        "Sampled metric shape:",
        sampled_values.shape
    )

    # Remove trajectories containing NaN or zero-valued samples.
    valid_mask = (
        np.all(
            np.isfinite(sampled_values),
            axis=1
        )
        &
        np.all(
            sampled_values != 0,
            axis=1
        )
    )

    clean_t1 = selected_t1[
        valid_mask
    ]

    clean_values = sampled_values[
        valid_mask
    ]

    print("")
    print(
        "Trajectories before cleaning:",
        len(selected_t1)
    )

    print(
        "Trajectories after cleaning:",
        len(clean_t1)
    )

    print(
        "Removed NaN / zero trajectories:",
        len(selected_t1)
        - len(clean_t1)
    )

    if len(clean_t1) == 0:
        raise ValueError(
            "No valid trajectories remain after cleaning."
        )

    print("")
    print(
        f"{args.metric_name.upper()} minimum:",
        float(
            np.nanmin(clean_values)
        )
    )

    print(
        f"{args.metric_name.upper()} maximum:",
        float(
            np.nanmax(clean_values)
        )
    )

    print(
        f"{args.metric_name.upper()} mean:",
        float(
            np.nanmean(clean_values)
        )
    )

    tube_vertices, tube_faces = build_tube_mesh(
        clean_t1,
        args.points_per_trajectory,
        args.tube_radius,
        args.tube_sides
    )

    (
        point_vertices,
        point_faces,
        overlay_values
    ) = build_point_mesh(
        clean_t1,
        args.point_radius,
        values=clean_values
    )

    metric_label = (
        args.metric_name.upper()
    )

    prefix = (
        f"{args.subject}_{args.hemi}_{args.region}_"
        f"{metric_label}_{int(args.fraction * 100)}percent"
    )

    tube_out = (
        output_dir
        / f"{prefix}_trajectories.surf"
    )

    point_out = (
        output_dir
        / f"{prefix}_points.surf"
    )

    overlay_out = (
        output_dir
        / f"{prefix}_points_{args.metric_name}.curv"
    )

    fsio.write_geometry(
        tube_out,
        tube_vertices,
        tube_faces
    )

    fsio.write_geometry(
        point_out,
        point_vertices,
        point_faces
    )

    fsio.write_morph_data(
        overlay_out,
        overlay_values
    )

    print("")
    print("Mode: metric")

    print("")
    print("Saved trajectory surface:")
    print(tube_out)

    print("")
    print("Saved sampling-point surface:")
    print(point_out)

    print("")
    print(
        f"Saved {metric_label} overlay:"
    )
    print(overlay_out)

    print("")
    print(
        "Displayed valid trajectories:",
        len(clean_t1)
    )

    print(
        "Sampling points per trajectory:",
        args.points_per_trajectory
    )

    print(
        "Displayed sampling points:",
        len(clean_t1)
        * args.points_per_trajectory
    )


def main():
    args = parse_args()

    if args.mode == "metric":
        if args.trajectory_dwi is None:
            raise ValueError(
                "--trajectory-dwi is required "
                "when --mode metric."
            )

        if args.metric_image is None:
            raise ValueError(
                "--metric-image is required "
                "when --mode metric."
            )

    trajectory_file = Path(
        args.trajectory
    )

    annot_file = Path(
        args.annot
    )

    output_dir = Path(
        args.output_dir
    )

    if not trajectory_file.exists():
        raise FileNotFoundError(
            trajectory_file
        )

    if not annot_file.exists():
        raise FileNotFoundError(
            annot_file
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load T1-space trajectories.
    t1_key, t1_trajectories = load_trajectories(
        trajectory_file,
        args.hemi,
        args.points_per_trajectory,
        space="t1"
    )

    print(
        f"T1 trajectory variable: {t1_key}"
    )

    print(
        "Whole T1 trajectory shape:",
        t1_trajectories.shape
    )

    print(
        "Total hemisphere trajectories:",
        len(t1_trajectories)
    )

    # Select cortical region.
    vertex_indices = load_region_indices(
        annot_file,
        args.region
    )

    region_t1 = select_region_trajectories(
        t1_trajectories,
        vertex_indices
    )

    print("")
    print(
        "Region:",
        args.region
    )

    print(
        "Region trajectories:",
        len(region_t1)
    )

    # Use exactly the same subset in both T1 and metric modes.
    selection = uniform_subsample(
        region_t1,
        args.fraction
    )

    selected_t1 = region_t1[
        selection
    ]

    print(
        "Selected trajectories:",
        len(selected_t1)
    )

    print(
        f"Selected fraction: "
        f"{args.fraction * 100:.0f}%"
    )

    if args.mode == "t1":
        run_t1_mode(
            args,
            selected_t1,
            output_dir
        )

    else:
        run_metric_mode(
            args,
            region_t1,
            selection,
            output_dir
        )


if __name__ == "__main__":
    main()