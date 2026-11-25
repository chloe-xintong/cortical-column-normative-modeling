#!/usr/bin/env python3
"""Compute one subject's K=30/100/300 platform sampling sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[5]
DISTANCE_ROOT = REPO / "analysis_results/figure6/确定扰动因素/distance"
sys.path.insert(0, str(REPO / "analysis_results/figure4/scripts"))
from compute_real_spatial_curves import load_profiles  # noqa: E402


K_VALUES = (30, 100, 300)
SEEDS = range(5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distance-root", type=Path, default=DISTANCE_ROOT)
    p.add_argument("--roster", type=Path, default=DISTANCE_ROOT /
        "local_similarity_stability/young_cn37_roster.csv")
    p.add_argument("--manifest", type=Path, default=REPO /
        "manifests/figure4_coverage95_pass_313_subjects.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int)
    g.add_argument("--subject-id")
    p.add_argument("--chunk-size", type=int, default=100_000)
    return p.parse_args()


def correlations(profiles: np.ndarray, valid: np.ndarray, pairs: pd.DataFrame,
                 chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    anchor = pairs.anchor_vertex.to_numpy(np.int64)
    neighbor = pairs.neighbor_vertex.to_numpy(np.int64)
    result = np.full(len(pairs), np.nan, dtype=np.float32)
    okay_result = np.zeros(len(pairs), dtype=bool)
    for start in range(0, len(pairs), chunk_size):
        stop = min(start + chunk_size, len(pairs))
        a, n = anchor[start:stop], neighbor[start:stop]
        okay = valid[a] & valid[n]
        ids = np.flatnonzero(okay)
        if not len(ids):
            continue
        left, right = profiles[a[ids]], profiles[n[ids]]
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        nonconstant = denominator > 0
        value = np.full(len(ids), np.nan)
        value[nonconstant] = np.sum(left[nonconstant] * right[nonconstant], axis=1) / denominator[nonconstant]
        result[start + ids] = value.astype(np.float32)
        okay_result[start + ids] = np.isfinite(value)
    return result, okay_result


def main() -> None:
    a = parse_args()
    roster = pd.read_csv(a.roster, dtype={"subject_id": str})
    manifest = pd.read_csv(a.manifest, dtype={"subject_id": str})
    if len(roster) != 37 or roster.subject_id.nunique() != 37:
        raise RuntimeError("Expected frozen 37-person roster")
    cohort = roster[["loo_row_index", "subject_id"]].merge(
        manifest, on="subject_id", how="left", validate="one_to_one")
    row = cohort.iloc[a.index] if a.index is not None else cohort.loc[
        cohort.subject_id.eq(a.subject_id)].iloc[0]
    subject = str(row.subject_id)
    source = a.distance_root / "local_similarity_stability"
    pilot = a.distance_root / "platform_sampling_sensitivity"
    out = pilot / "features" / "subjects"
    out.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(source / "ico4_nodes.csv")
    counts = pd.read_parquet(source / "anchor_bin_template_counts.parquet")
    counts = counts.loc[counts.distance_bin_index.between(10, 14)].copy()
    bin_parts = []
    for hemi in ("lh", "rh"):
        profiles, valid = load_profiles(Path(row[f"clean_profile_{hemi}"]))
        base = counts.loc[counts.hemisphere.eq(hemi)].copy()
        for seed in SEEDS:
            pairs = pd.read_parquet(
                pilot / "pairs" / f"seed_{seed:02d}_platform_k300_nested.parquet",
                filters=[("hemisphere", "==", hemi)],
            )
            r, pair_valid = correlations(profiles, valid, pairs, a.chunk_size)
            pairs = pairs[["node_order", "distance_bin_index", "sample_rank"]].copy()
            pairs["r"] = r
            pairs["pair_valid"] = pair_valid
            for k in K_VALUES:
                selected = pairs.loc[pairs.sample_rank.le(k) & pairs.pair_valid]
                grouped = selected.groupby(
                    ["node_order", "distance_bin_index"], observed=True, sort=False
                ).agg(r_platform_bin=("r", "median"), n_valid_pairs=("r", "size")).reset_index()
                current = base.merge(grouped, on=["node_order", "distance_bin_index"],
                                     how="left", validate="one_to_one")
                current["n_valid_pairs"] = current.n_valid_pairs.fillna(0).astype(np.int16)
                current["n_template_selected"] = np.minimum(current.n_geometry, k).astype(np.int16)
                current["candidate_exhausted"] = current.n_geometry.le(k)
                current["seed"] = np.int8(seed); current["platform_k"] = np.int16(k)
                bin_parts.append(current[[
                    "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
                    "distance_bin_index", "distance_bin", "seed", "platform_k", "n_geometry",
                    "n_template_selected", "candidate_exhausted", "n_valid_pairs", "r_platform_bin",
                ]])
    bins = pd.concat(bin_parts, ignore_index=True)
    platform = bins.groupby([
        "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region", "seed", "platform_k"
    ], observed=True, sort=False).agg(
        p_real=("r_platform_bin", "median"),
        n_valid_platform_bins=("r_platform_bin", "count"),
        minimum_valid_pairs=("n_valid_pairs", "min"),
        median_valid_pairs=("n_valid_pairs", "median"),
        n_exhausted_bins=("candidate_exhausted", "sum"),
        any_candidate_exhausted=("candidate_exhausted", "any"),
    ).reset_index()
    platform["platform_complete"] = platform.n_valid_platform_bins.eq(5)
    platform.insert(0, "subject_id", subject)
    bins.insert(0, "subject_id", subject)

    # Fix local r to the existing K=30, seed=0 realization; vary platform only.
    old = pd.read_parquet(source / "features/subjects" / f"{subject}.parquet")
    local = old.loc[
        old.k.eq(30) & old.seed.eq(0) & old.distance_bin_index.lt(5),
        ["subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
         "distance_bin_index", "distance_bin", "r_real", "n_valid_pairs"],
    ].rename(columns={"r_real": "r_local_fixed_k30_seed0",
                      "n_valid_pairs": "n_valid_local_pairs"})
    propagated = local.merge(
        platform.drop(columns=["hemisphere", "anchor_dk_label", "anchor_dk_region"]),
        on=["subject_id", "node_order"], how="left", validate="many_to_many")
    propagated["L_platform_only"] = (
        propagated.r_local_fixed_k30_seed0 - propagated.p_real
    ).astype(np.float32)
    bins.to_parquet(out / f"{subject}_platform_bins.parquet", index=False, compression="zstd")
    platform.to_parquet(out / f"{subject}_platform.parquet", index=False, compression="zstd")
    propagated.to_parquet(out / f"{subject}_local_L.parquet", index=False, compression="zstd")
    report = {
        "status": "complete", "subject_id": subject,
        "fixed_local_definition": "existing K=30 seed=0 r_real for bins 0-10mm",
        "n_platform_bin_rows": len(bins), "n_platform_rows": len(platform),
        "n_propagated_local_rows": len(propagated),
    }
    (out / f"{subject}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "subject_id"}, indent=2))


if __name__ == "__main__":
    main()
