"""Shared coordinate transforms for native T1-space cortical columns."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_transform_matrix(transform_path: Path) -> np.ndarray:
    """Read the 4x4 matrix embedded in a FreeSurfer registration file."""

    transform_path = Path(transform_path)
    if not transform_path.is_file():
        raise FileNotFoundError(f"Transform file not found: {transform_path}")

    lines = transform_path.read_text().splitlines()
    for start in range(len(lines) - 3):
        rows = []
        for line in lines[start : start + 4]:
            values = line.strip().replace(",", " ").split()
            if len(values) < 4:
                break
            try:
                rows.append([float(value) for value in values[:4]])
            except ValueError:
                break
        if len(rows) == 4:
            return np.asarray(rows, dtype=float)

    raise ValueError(f"Could not find a 4x4 transform matrix in {transform_path}")


def vox2ras_tkreg(shape: tuple[int, ...], voxel_size: tuple[float, ...]) -> np.ndarray:
    """Construct the FreeSurfer tkregister voxel-to-RAS matrix."""

    nx, ny, nz = [float(value) for value in shape[:3]]
    dx, dy, dz = [float(value) for value in voxel_size[:3]]
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0] = -dx
    matrix[0, 3] = dx * nx / 2.0
    matrix[1, 2] = dz
    matrix[1, 3] = -dz * nz / 2.0
    matrix[2, 1] = -dy
    matrix[2, 3] = dy * ny / 2.0
    matrix[3, 3] = 1.0
    return matrix


def vox2ras_0to1(matrix: np.ndarray) -> np.ndarray:
    """Convert the tkregister matrix to the pipeline's voxel convention."""

    offset = np.zeros((4, 4), dtype=float)
    offset[:3, 3] = 1.0
    return np.linalg.inv(np.linalg.inv(matrix) + offset)


def load_dwi_geometry(dwi_path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    """Return the DWI shape and the tkregister voxel geometry matrix."""

    image = nib.load(str(dwi_path))
    shape = image.shape[:3]
    voxel_size = image.header.get_zooms()[:3]
    t_mov = vox2ras_0to1(vox2ras_tkreg(shape, voxel_size))
    return shape, t_mov


def transform_points_to_dwi(
    points_t1: np.ndarray,
    transform: np.ndarray,
    t_mov: np.ndarray,
) -> np.ndarray:
    """Transform points from native T1 RAS to DWI voxel coordinates."""

    points_t1 = np.asarray(points_t1, dtype=float)
    if points_t1.ndim != 2 or points_t1.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points_t1.shape}")

    homogeneous = np.ones((4, len(points_t1)), dtype=float)
    homogeneous[:3] = points_t1.T
    return (np.linalg.inv(t_mov) @ transform @ homogeneous).T[:, :3]
