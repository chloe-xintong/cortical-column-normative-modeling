"""Shared profile-pair calculations for Figure 4 real and null data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_metrics(
    profiles: np.ndarray,
    valid_vertices: np.ndarray,
    pairs: pd.DataFrame,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    dissimilarity = np.full(len(pairs), np.nan, dtype=np.float32)
    mae = np.full(len(pairs), np.nan, dtype=np.float32)
    rmse = np.full(len(pairs), np.nan, dtype=np.float32)
    anchors = pairs["anchor_vertex"].to_numpy(dtype=np.int64)
    neighbors = pairs["neighbor_vertex"].to_numpy(dtype=np.int64)

    for start in range(0, len(pairs), chunk_size):
        stop = min(start + chunk_size, len(pairs))
        anchor_ids = anchors[start:stop]
        neighbor_ids = neighbors[start:stop]
        valid = valid_vertices[anchor_ids] & valid_vertices[neighbor_ids]
        if not valid.any():
            continue
        local = np.flatnonzero(valid)
        left = profiles[anchor_ids[local]]
        right = profiles[neighbor_ids[local]]
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
            right_centered,
            axis=1,
        )
        nonconstant = denominator > 0
        correlation = np.full(len(local), np.nan, dtype=float)
        correlation[nonconstant] = (
            np.sum(
                left_centered[nonconstant] * right_centered[nonconstant],
                axis=1,
            )
            / denominator[nonconstant]
        )
        difference = left - right
        output_ids = start + local
        dissimilarity[output_ids] = (1.0 - correlation).astype(np.float32)
        mae[output_ids] = np.mean(np.abs(difference), axis=1).astype(np.float32)
        rmse[output_ids] = np.sqrt(np.mean(difference**2, axis=1)).astype(np.float32)

    result = pairs.copy()
    result["profile_dissimilarity"] = dissimilarity
    result["profile_mae"] = mae
    result["profile_rmse"] = rmse
    result["pair_valid"] = np.isfinite(dissimilarity)
    return result


def aggregate_anchor_bins(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics.loc[metrics["pair_valid"]].copy()
    return (
        valid.groupby(
            [
                "hemisphere",
                "anchor_vertex",
                "distance_bin_index",
                "distance_bin",
                "pair_type",
                "split_half",
            ],
            observed=True,
        )
        .agg(
            median_dissimilarity=("profile_dissimilarity", "median"),
            median_profile_mae=("profile_mae", "median"),
            median_profile_rmse=("profile_rmse", "median"),
            median_geodesic_distance_mm=("geodesic_distance_mm", "median"),
            n_pairs=("profile_dissimilarity", "size"),
        )
        .reset_index()
    )


def aggregate_subject_bins(anchor_bins: pd.DataFrame) -> pd.DataFrame:
    anchor_bins = anchor_bins.copy()
    anchor_bins["hemisphere_anchor"] = (
        anchor_bins["hemisphere"].astype(str)
        + ":"
        + anchor_bins["anchor_vertex"].astype(str)
    )
    return (
        anchor_bins.groupby(
            ["distance_bin_index", "distance_bin", "pair_type"],
            observed=True,
        )
        .agg(
            median_dissimilarity=("median_dissimilarity", "median"),
            median_profile_mae=("median_profile_mae", "median"),
            median_profile_rmse=("median_profile_rmse", "median"),
            median_geodesic_distance_mm=(
                "median_geodesic_distance_mm",
                "median",
            ),
            n_anchors=("hemisphere_anchor", "nunique"),
            n_pairs=("n_pairs", "sum"),
        )
        .reset_index()
    )
