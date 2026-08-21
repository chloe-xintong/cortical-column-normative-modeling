"""
Create a surface visualization of cortical trajectories for a selected
FreeSurfer cortical region.

The script loads cortical trajectory points, selects trajectories whose
surface vertices belong to a specified aparc region, optionally subsamples
them, and converts the trajectories into tube-like surface geometry for
visualization.

The current script uses manually specified input/output paths and parameters.
""" 
import numpy as np
from scipy.io import loadmat
import nibabel.freesurfer.io as fsio

# ============================================================
# Paths
# ============================================================

column_file = (
    "/mnt/newStor/paros/paros_WORK/xintong/test/output/"
    "S00775/columns/S00775_column_lh.mat"
)

annot_file = (
    "/mnt/newStor/paros/paros_WORK/xintong/test/input/"
    "S00775/S00775/label/lh.aparc.annot"
)

out_file = (
    "/mnt/newStor/paros/paros_WORK/xintong/test/output/"
    "S00775/columns/sample_lh_bankssts_T1_70percent_columns.surf"
)

# ============================================================
# Parameters
# ============================================================

POINTS_PER_COLUMN = 21
REGION_NAME = "bankssts"

# select percentage of the columns in the region
FRACTION = 0.70

# tube radius
RADIUS = 0.12

# number of sides for the tube cross-section
SIDES = 6


# ============================================================
# 1. Load all LH columns
# ============================================================

d = loadmat(column_file)

if "lh_column_points" not in d:
    raise KeyError(
        "lh_column_points not found in column file."
    )

pts = d["lh_column_points"]

print(
    "Loaded lh_column_points:",
    pts.shape,
    pts.dtype
)

if pts.ndim != 2 or pts.shape[1] != 3:
    raise ValueError(
        f"Unexpected lh_column_points shape: {pts.shape}"
    )

if pts.shape[0] % POINTS_PER_COLUMN != 0:
    raise ValueError(
        f"Total point number {pts.shape[0]} "
        f"is not divisible by {POINTS_PER_COLUMN}."
    )

columns = pts.reshape(
    -1,
    POINTS_PER_COLUMN,
    3
)

print(
    "Whole LH columns shape:",
    columns.shape
)

print(
    "Total LH columns:",
    columns.shape[0]
)


# ============================================================
# 2. Load FreeSurfer annotation
# ============================================================

labels, ctab, names = fsio.read_annot(
    annot_file
)

names = [
    n.decode("utf-8")
    if isinstance(n, bytes)
    else str(n)
    for n in names
]

if REGION_NAME not in names:
    raise ValueError(
        f"Region '{REGION_NAME}' not found."
    )

region_index = names.index(
    REGION_NAME
)

vertex_indices = np.where(
    labels == region_index
)[0]

print("")
print(
    "Region:",
    REGION_NAME
)

print(
    "Number of region vertices / columns:",
    len(vertex_indices)
)


# ============================================================
# 3. Select all columns belonging to bankssts
# ============================================================

# column i 对应 surface vertex i
region_columns = columns[
    vertex_indices
]

print(
    "Region columns shape:",
    region_columns.shape
)

if len(region_columns) == 0:
    raise ValueError(
        f"No columns found for {REGION_NAME}."
    )


# ============================================================
# 4. Uniformly select percentage of the columns in the region
# ============================================================

n_show = max(
    1,
    int(
        round(
            len(region_columns)
            * FRACTION
        )
    )
)

idx = np.linspace(
    0,
    len(region_columns) - 1,
    n_show,
    dtype=int
)

selected = region_columns[
    idx
]

print("")
print(
    "Selected columns shape:",
    selected.shape
)

print(
    f"Selected {n_show} columns "
    f"({FRACTION * 100:.0f}%) "
    f"from {REGION_NAME}"
)


# ============================================================
# 5. Helper function for building tube cross-sections
# ============================================================

def make_frame(direction):

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

    u = u / u_norm

    v = np.cross(
        direction,
        u
    )

    v = (
        v
        / np.linalg.norm(v)
    )

    return u, v


# ============================================================
# 6. Convert every selected column into a thin tube
# ============================================================

vertices = []
faces = []

for col in selected:

    base_index = len(
        vertices
    )

    # --------------------------------------------------------
    # Create one small ring around every depth point
    # --------------------------------------------------------

    for j in range(
        POINTS_PER_COLUMN
    ):

        if j == 0:

            direction = (
                col[1]
                - col[0]
            )

        elif j == (
            POINTS_PER_COLUMN - 1
        ):

            direction = (
                col[-1]
                - col[-2]
            )

        else:

            direction = (
                col[j + 1]
                - col[j - 1]
            )

        u, v = make_frame(
            direction
        )

        for s in range(
            SIDES
        ):

            theta = (
                2.0
                * np.pi
                * s
                / SIDES
            )

            offset = (
                RADIUS
                * (
                    np.cos(theta)
                    * u
                    +
                    np.sin(theta)
                    * v
                )
            )

            vertices.append(
                col[j]
                + offset
            )

    # --------------------------------------------------------
    # Connect neighboring rings with triangles
    # --------------------------------------------------------

    for j in range(
        POINTS_PER_COLUMN - 1
    ):

        ring1 = (
            base_index
            + j * SIDES
        )

        ring2 = (
            base_index
            + (j + 1)
            * SIDES
        )

        for s in range(
            SIDES
        ):

            s_next = (
                s + 1
            ) % SIDES

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


# ============================================================
# 7. Convert lists to arrays
# ============================================================

vertices = np.asarray(
    vertices,
    dtype=np.float32
)

faces = np.asarray(
    faces,
    dtype=np.int32
)

print("")
print(
    "Mesh vertices:",
    vertices.shape
)

print(
    "Mesh faces:",
    faces.shape
)


# ============================================================
# 8. Write native FreeSurfer surface
# ============================================================

fsio.write_geometry(
    out_file,
    vertices,
    faces
)

print("")
print(
    "Saved FreeSurfer surface to:"
)

print(
    out_file
)