"""Deterministic one-dimensional matching helpers for Figure 4D.

Description
-----------
Deterministic one-dimensional matching helpers for Figure 4D.

Methods
-------
1. Provide the reusable computations defined in this module.
2. Return results to the calling analysis code.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
Import this module and call ``distance_bin_index(...)``; it has no command-line entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DISTANCE_EDGES_0_10 = np.arange(0.0, 12.0, 2.0)


def distance_bin_index(
    distances: np.ndarray,
    edges: np.ndarray = DISTANCE_EDGES_0_10,
) -> np.ndarray:
    """Assign the same right-closed bins used by the frozen Figure 4A code.

    The intervals are (0, 2], (2, 4], ..., (8, 10]. Values outside the
    positive 0--10-mm range receive -1. Although prose sometimes writes the
    bins as [0, 2), this reproduces the actual frozen A--C implementation.
    """
    values = np.asarray(distances, dtype=float)
    indices = np.searchsorted(edges, values, side="left") - 1
    valid = np.isfinite(values) & (values > edges[0]) & (values <= edges[-1])
    return np.where(valid, indices, -1).astype(np.int8)


def valid_profile_mask(profiles: np.ndarray, base_valid: np.ndarray) -> np.ndarray:
    """Require existing validity, finite depths, and nonconstant profiles."""
    profiles = np.asarray(profiles, dtype=float)
    valid = np.asarray(base_valid, dtype=bool).copy()
    valid &= np.isfinite(profiles).all(axis=1)
    valid &= np.ptp(profiles, axis=1) > 0
    return valid


def filter_candidate_frame(
    candidates: pd.DataFrame,
    profile_valid: np.ndarray,
) -> pd.DataFrame:
    """Apply subject validity and remove self-pairs from a candidate table."""
    anchor = candidates["anchor_vertex"].to_numpy(dtype=np.int64)
    neighbor = candidates["candidate_vertex"].to_numpy(dtype=np.int64)
    keep = (
        (anchor != neighbor)
        & profile_valid[anchor]
        & profile_valid[neighbor]
    )
    return candidates.loc[keep].copy()


def _closest_sorted_pair(
    within_distance: np.ndarray,
    within_vertex: np.ndarray,
    across_distance: np.ndarray,
    across_vertex: np.ndarray,
) -> tuple[int, int, float] | None:
    """Find the deterministic closest pair between two sorted 1D arrays."""
    if not len(within_distance) or not len(across_distance):
        return None
    # The closest across value to each within value must be either the first
    # value at/above it or the last distinct value below it. NumPy performs
    # this search in compiled code; no n-by-m cost matrix is constructed.
    position = np.searchsorted(across_distance, within_distance, side="left")
    within_indices = []
    across_indices = []
    valid_upper = position < len(across_distance)
    if valid_upper.any():
        within_indices.append(np.flatnonzero(valid_upper))
        across_indices.append(position[valid_upper])
    valid_lower = position > 0
    if valid_lower.any():
        local_within = np.flatnonzero(valid_lower)
        local_across = position[valid_lower] - 1
        # If several across candidates have the same lower distance, use the
        # smallest vertex ID, which is first in the stable distance/ID sort.
        local_across = np.searchsorted(
            across_distance, across_distance[local_across], side="left"
        )
        within_indices.append(local_within)
        across_indices.append(local_across)
    if not within_indices:
        return None
    within_index = np.concatenate(within_indices)
    across_index = np.concatenate(across_indices)
    difference = np.abs(
        within_distance[within_index] - across_distance[across_index]
    )
    order = np.lexsort(
        (
            across_vertex[across_index],
            within_vertex[within_index],
            difference,
        )
    )
    selected = int(order[0])
    return (
        int(within_index[selected]),
        int(across_index[selected]),
        float(difference[selected]),
    )


def deterministic_distance_match(
    within: pd.DataFrame,
    across: pd.DataFrame,
    maximum_matches: int = 15,
    caliper_mm: float = 0.1,
) -> pd.DataFrame:
    """Greedily select the closest unique within/across distances.

    Each iteration finds the global closest remaining 1D pair in O(n+m),
    selects it if it passes the caliper, and removes both candidates. Thus the
    total cost is O(K(n+m)) rather than an n-by-m cost matrix. Ties are broken
    by within then across candidate vertex ID.
    """
    columns = [
        "within_candidate_vertex", "within_candidate_dk_label",
        "within_distance_mm", "across_candidate_vertex",
        "across_candidate_dk_label", "across_distance_mm",
        "absolute_distance_difference_mm",
    ]
    if maximum_matches <= 0 or within.empty or across.empty:
        return pd.DataFrame(columns=columns)
    within_order = np.lexsort(
        (
            within["candidate_vertex"].to_numpy(np.int64),
            within["geodesic_distance_mm"].to_numpy(float),
        )
    )
    across_order = np.lexsort(
        (
            across["candidate_vertex"].to_numpy(np.int64),
            across["geodesic_distance_mm"].to_numpy(float),
        )
    )
    within_distance = within["geodesic_distance_mm"].to_numpy(float)[within_order]
    within_vertex = within["candidate_vertex"].to_numpy(np.int64)[within_order]
    within_label = within["candidate_dk_label"].to_numpy(np.int64)[within_order]
    across_distance = across["geodesic_distance_mm"].to_numpy(float)[across_order]
    across_vertex = across["candidate_vertex"].to_numpy(np.int64)[across_order]
    across_label = across["candidate_dk_label"].to_numpy(np.int64)[across_order]
    within_active = np.ones(len(within_distance), dtype=bool)
    across_active = np.ones(len(across_distance), dtype=bool)
    rows: list[dict[str, object]] = []
    for _ in range(min(maximum_matches, len(within), len(across))):
        within_ids = np.flatnonzero(within_active)
        across_ids = np.flatnonzero(across_active)
        closest = _closest_sorted_pair(
            within_distance[within_ids], within_vertex[within_ids],
            across_distance[across_ids], across_vertex[across_ids],
        )
        if closest is None:
            break
        local_within_index, local_across_index, difference = closest
        if difference > caliper_mm + 1e-12:
            break
        within_index = int(within_ids[local_within_index])
        across_index = int(across_ids[local_across_index])
        rows.append(
            {
                "within_candidate_vertex": int(within_vertex[within_index]),
                "within_candidate_dk_label": int(within_label[within_index]),
                "within_distance_mm": float(within_distance[within_index]),
                "across_candidate_vertex": int(across_vertex[across_index]),
                "across_candidate_dk_label": int(across_label[across_index]),
                "across_distance_mm": float(across_distance[across_index]),
                "absolute_distance_difference_mm": float(difference),
            }
        )
        within_active[within_index] = False
        across_active[across_index] = False
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result.insert(0, "match_id", np.arange(1, len(result) + 1, dtype=np.int8))
    else:
        result.insert(0, "match_id", pd.Series(dtype=np.int8))
    return result


def label_candidate_pools(
    candidates: pd.DataFrame,
    anchor_label: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    within = candidates.loc[
        candidates["candidate_dk_label"].eq(anchor_label)
    ].copy()
    across = candidates.loc[
        candidates["candidate_dk_label"].ne(anchor_label)
    ].copy()
    return within, across
