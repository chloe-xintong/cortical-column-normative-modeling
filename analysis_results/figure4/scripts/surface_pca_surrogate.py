"""Utilities for empirically smoothness-matched surface PCA surrogates."""

from __future__ import annotations

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def load_real_profiles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned canonical profiles and the final real-data valid mask."""
    with np.load(path, allow_pickle=True) as data:
        profiles = np.asarray(data["profiles"], dtype=np.float32)
        target_ids = np.asarray(data["target_vertex_id"], dtype=np.int64)
        direction = str(data["depth_direction"].item())
        depths = np.asarray(data["depth_fractions"], dtype=float)
        valid = np.asarray(data["sampling_valid"], dtype=bool)
        if "in_analysis_mask" in data.files:
            valid &= np.asarray(data["in_analysis_mask"], dtype=bool)
        if "mapping_valid" in data.files:
            valid &= np.asarray(data["mapping_valid"], dtype=bool)
        if "column_valid" in data.files:
            valid &= np.asarray(data["column_valid"], dtype=bool)
    if direction != "pial_to_white":
        raise ValueError(f"Expected pial_to_white: {path}")
    if profiles.shape[1] != 21 or not np.allclose(depths, np.linspace(0, 1, 21)):
        raise ValueError(f"Expected canonical 21-depth profiles: {path}")
    if np.unique(target_ids).size != len(target_ids):
        raise ValueError(f"Duplicate target vertex IDs: {path}")
    valid &= np.isfinite(profiles).all(axis=1)
    valid &= np.var(profiles, axis=1) > 0
    n_vertices = int(target_ids.max()) + 1
    aligned = np.full((n_vertices, 21), np.nan, dtype=np.float32)
    aligned_valid = np.zeros(n_vertices, dtype=bool)
    aligned[target_ids] = profiles
    aligned_valid[target_ids] = valid
    return aligned, aligned_valid


def fit_pca(
    profiles_by_hemisphere: dict[str, np.ndarray],
    valid_by_hemisphere: dict[str, np.ndarray],
    variance_explained: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    observed = np.concatenate(
        [profiles_by_hemisphere[h][valid_by_hemisphere[h]] for h in ("lh", "rh")]
    ).astype(np.float64)
    mean_profile = np.median(observed, axis=0)
    residual = observed - mean_profile
    _, singular_values, loadings = np.linalg.svd(residual, full_matrices=False)
    fraction = singular_values**2 / np.sum(singular_values**2)
    n_components = min(
        len(fraction),
        int(np.searchsorted(np.cumsum(fraction), variance_explained) + 1),
    )
    loadings = loadings[:n_components]
    scores = {}
    for hemisphere in ("lh", "rh"):
        score = np.full((len(profiles_by_hemisphere[hemisphere]), n_components), np.nan)
        mask = valid_by_hemisphere[hemisphere]
        score[mask] = (
            profiles_by_hemisphere[hemisphere][mask].astype(np.float64) - mean_profile
        ) @ loadings.T
        scores[hemisphere] = score
    return mean_profile, loadings, scores, fraction[:n_components]


def _frame(unit: np.ndarray, first: int, second: int) -> np.ndarray:
    first_axis = unit[first]
    second_axis = unit[second] - np.dot(unit[second], first_axis) * first_axis
    second_axis /= np.linalg.norm(second_axis)
    return np.column_stack((first_axis, second_axis, np.cross(first_axis, second_axis)))


def icosahedral_rotation(unit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Choose one of the exact orientation-preserving fsaverage ico rotations."""
    base = np.arange(12)  # the 12 degree-five vertices of the ico tessellation
    directed_edges = []
    for first in base:
        similarity = unit[base] @ unit[first]
        neighbors = base[np.argsort(similarity)[-6:-1]]
        directed_edges.extend((int(first), int(second)) for second in neighbors)
    reference = directed_edges[0]
    choices = [edge for edge in directed_edges if edge != reference]
    target = choices[int(rng.integers(len(choices)))]
    return _frame(unit, *target) @ _frame(unit, *reference).T


def masked_icosahedral_permutation(
    unit: np.ndarray, valid: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Return a bijection of valid vertices induced by an ico rotation.

    Exact ico rotations permute all fsaverage vertices one-to-one.  Where the
    rotated medial wall overlaps cortex, unmatched valid vertices are paired
    to the nearest still-unused valid source so the subject mask remains exact
    without duplicate sampling.
    """
    rotation = icosahedral_rotation(unit, rng)
    rotated = unit @ rotation.T
    _, full_permutation = cKDTree(unit).query(rotated, k=1)
    valid_ids = np.flatnonzero(valid)
    mapped = full_permutation[valid_ids]
    repair = ~valid[mapped]
    repair_fraction = float(repair.mean())
    if not repair.any():
        return mapped, repair_fraction
    used = mapped[~repair]
    available = np.setdiff1d(valid_ids, used, assume_unique=False)
    remaining_targets = valid_ids[repair]
    remaining_sources = available
    repaired = np.empty(len(remaining_targets), dtype=np.int64)
    open_positions = np.arange(len(remaining_targets))
    while len(open_positions):
        tree = cKDTree(unit[remaining_sources])
        neighbor_count = min(64, len(remaining_sources))
        distance, candidate = tree.query(
            rotated[remaining_targets[open_positions]], k=neighbor_count
        )
        if neighbor_count == 1:
            distance = distance[:, None]
            candidate = candidate[:, None]
        flat_order = np.argsort(distance.ravel(), kind="mergesort")
        target_candidate = np.repeat(np.arange(len(open_positions)), neighbor_count)
        source_candidate = candidate.ravel()
        target_taken = np.zeros(len(open_positions), dtype=bool)
        source_taken = np.zeros(len(remaining_sources), dtype=bool)
        assigned_target = []
        assigned_source = []
        for edge in flat_order:
            target_index = target_candidate[edge]
            source_index = source_candidate[edge]
            if target_taken[target_index] or source_taken[source_index]:
                continue
            target_taken[target_index] = True
            source_taken[source_index] = True
            assigned_target.append(target_index)
            assigned_source.append(source_index)
        assigned_target = np.asarray(assigned_target, dtype=int)
        assigned_source = np.asarray(assigned_source, dtype=int)
        repaired[open_positions[assigned_target]] = remaining_sources[assigned_source]
        open_positions = open_positions[~target_taken]
        remaining_sources = remaining_sources[~source_taken]
    mapped[repair] = repaired
    if np.unique(mapped).size != len(valid_ids):
        raise RuntimeError("Mask-constrained ico rotation was not bijective")
    return mapped, repair_fraction


def icosahedral_permutation(
    unit: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return an exact full-sphere vertex permutation for one ico rotation."""
    rotation = icosahedral_rotation(unit, rng)
    rotated = unit @ rotation.T
    _, permutation = cKDTree(unit).query(rotated, k=1)
    if np.unique(permutation).size != len(unit):
        raise RuntimeError("Icosahedral rotation was not a full-sphere bijection")
    return permutation


def harmonic_extend_scores(
    scores: np.ndarray, valid: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Harmonically extend valid score maps over the complete sphere mesh."""
    edge = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    row = np.concatenate((edge[:, 0], edge[:, 1]))
    column = np.concatenate((edge[:, 1], edge[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(row)), (row, column)), shape=(len(valid), len(valid))
    ).tocsr()
    adjacency.data[:] = 1.0
    invalid_ids = np.flatnonzero(~valid)
    valid_ids = np.flatnonzero(valid)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    system = sparse.diags(degree[invalid_ids]) - adjacency[invalid_ids][:, invalid_ids]
    rhs = adjacency[invalid_ids][:, valid_ids] @ scores[valid_ids]
    extended = scores.astype(np.float64, copy=True)
    extended[invalid_ids] = spsolve(system.tocsc(), rhs)
    if not np.isfinite(extended).all():
        raise RuntimeError("Harmonic score extension produced non-finite values")
    return extended


def rank_remap(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    remapped = np.empty_like(values)
    remapped[order] = np.sort(reference)
    return remapped


def normalized_masked_adjacency(
    faces: np.ndarray, valid: np.ndarray
) -> sparse.csr_matrix:
    """Build a row-normalized white-mesh adjacency restricted to valid cortex."""
    valid_ids = np.flatnonzero(valid)
    local_id = np.full(len(valid), -1, dtype=np.int64)
    local_id[valid_ids] = np.arange(len(valid_ids))
    edge = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    keep = valid[edge[:, 0]] & valid[edge[:, 1]]
    edge = edge[keep]
    row = np.concatenate((local_id[edge[:, 0]], local_id[edge[:, 1]]))
    column = np.concatenate((local_id[edge[:, 1]], local_id[edge[:, 0]]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, column)),
        shape=(len(valid_ids), len(valid_ids)),
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    isolated = np.flatnonzero(degree == 0)
    if isolated.size:
        # Subject-specific sampling masks can leave a few isolated valid
        # vertices.  A self-loop makes diffusion the identity at those sites.
        adjacency = adjacency.tolil()
        adjacency[isolated, isolated] = 1.0
        adjacency = adjacency.tocsr()
        degree[isolated] = 1.0
    return sparse.diags(1.0 / degree) @ adjacency


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(float(eigenvalues.max()) * 1e-12, np.finfo(float).eps)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * (eigenvalues**power)) @ eigenvectors.T


def match_mean_and_covariance(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Affine recolor values to the reference 21-depth mean and covariance."""
    values_centered = values - values.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    value_covariance = np.cov(values_centered, rowvar=False)
    reference_covariance = np.cov(reference, rowvar=False)
    transform = _symmetric_matrix_power(value_covariance, -0.5) @ _symmetric_matrix_power(
        reference_covariance, 0.5
    )
    return values_centered @ transform + reference_mean


def diffuse_on_surface(
    values: np.ndarray,
    adjacency: sparse.csr_matrix,
    strength: float,
    steps: int = 1,
) -> np.ndarray:
    """Apply convex white-mesh diffusion steps to all depth profiles."""
    if not -1.0 < strength < 1.0:
        raise ValueError("graph-filter strength must be in (-1, 1)")
    if steps < 1:
        raise ValueError("diffusion steps must be positive")
    if strength == 0:
        return values
    output = values
    for _ in range(steps):
        output = (1.0 - strength) * output + strength * (adjacency @ output)
    return output


def generate_surrogate(
    mean_profile: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
    sphere_vertices: np.ndarray,
    real_profiles: np.ndarray,
    rng: np.random.Generator,
    do_rank_remap: bool = True,
    adjacency: sparse.csr_matrix | None = None,
    diffusion_strength: float = 0.0,
    diffusion_steps: int = 1,
    rotation_mode: str = "independent_components",
    extended_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomize PCA score maps on the sphere and reconstruct profiles.

    Independent component rotations are the inferential null. A shared
    rotation is retained only for diagnostic backward compatibility: with all
    components retained it nearly rotates complete profiles intact.
    """
    unit = sphere_vertices / np.linalg.norm(sphere_vertices, axis=1, keepdims=True)
    valid_ids = np.flatnonzero(valid)
    if rotation_mode == "shared":
        source_ids, repair = masked_icosahedral_permutation(unit, valid, rng)
        rotated_scores = scores[source_ids].astype(np.float64)
        unique_fraction = np.repeat(
            np.unique(source_ids).size / len(valid_ids), loadings.shape[0]
        )
        repair_fraction = np.repeat(repair, loadings.shape[0])
        imputed_source_fraction = np.zeros(loadings.shape[0])
    elif rotation_mode == "independent_components":
        if extended_scores is None:
            raise ValueError("extended_scores are required for independent rotations")
        rotated_scores = np.empty((len(valid_ids), loadings.shape[0]), dtype=np.float64)
        unique_fraction = np.empty(loadings.shape[0], dtype=np.float64)
        repair_fraction = np.zeros(loadings.shape[0], dtype=np.float64)
        imputed_source_fraction = np.empty(loadings.shape[0], dtype=np.float64)
        for component in range(loadings.shape[0]):
            full_permutation = icosahedral_permutation(unit, rng)
            source_ids = full_permutation[valid_ids]
            rotated_scores[:, component] = extended_scores[source_ids, component]
            unique_fraction[component] = np.unique(source_ids).size / len(valid_ids)
            imputed_source_fraction[component] = (~valid[source_ids]).mean()
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
    reconstructed = mean_profile + rotated_scores @ loadings
    if diffusion_strength:
        if adjacency is None:
            raise ValueError("adjacency is required when diffusion_strength > 0")
        reconstructed = diffuse_on_surface(
            reconstructed, adjacency, diffusion_strength, diffusion_steps
        )
    if diffusion_strength or rotation_mode == "independent_components":
        reconstructed = match_mean_and_covariance(
            reconstructed, real_profiles[valid_ids].astype(np.float64)
        )
    if do_rank_remap:
        for depth in range(reconstructed.shape[1]):
            reconstructed[:, depth] = rank_remap(
                reconstructed[:, depth], real_profiles[valid_ids, depth]
            )
    output = np.full(real_profiles.shape, np.nan, dtype=np.float32)
    output[valid_ids] = reconstructed.astype(np.float32)
    return output, unique_fraction, repair_fraction, imputed_source_fraction


def load_spheres(subjects_dir: Path, template: str = "fsaverage") -> dict[str, np.ndarray]:
    return {
        hemisphere: fsio.read_geometry(
            str(subjects_dir / template / "surf" / f"{hemisphere}.sphere")
        )[0]
        for hemisphere in ("lh", "rh")
    }


def load_white_faces(
    subjects_dir: Path, template: str = "fsaverage"
) -> dict[str, np.ndarray]:
    return {
        hemisphere: fsio.read_geometry(
            str(subjects_dir / template / "surf" / f"{hemisphere}.white")
        )[1]
        for hemisphere in ("lh", "rh")
    }
