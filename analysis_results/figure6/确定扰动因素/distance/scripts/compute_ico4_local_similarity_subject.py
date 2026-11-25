#!/usr/bin/env python3
"""Compute one frozen young-CN subject's real ico4 local-similarity features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "analysis_results/figure4/scripts"))
from compute_real_spatial_curves import load_profiles  # noqa: E402


SEEDS = range(5)
K_VALUES = (15, 30)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-dir", type=Path, default=REPO /
        "analysis_results/figure6/确定扰动因素/distance/local_similarity_stability")
    p.add_argument("--roster", type=Path, default=REPO /
        "analysis_results/figure6/figure6_new/outputs/full/young_reference_roster.csv")
    p.add_argument("--manifest", type=Path, default=REPO /
        "manifests/figure4_coverage95_pass_313_subjects.csv")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--subject-id")
    p.add_argument("--chunk-size", type=int, default=100_000)
    return p.parse_args()


def correlations(profiles: np.ndarray, valid: np.ndarray, pairs: pd.DataFrame,
                 chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    anchor = pairs.anchor_vertex.to_numpy(np.int64)
    neighbor = pairs.neighbor_vertex.to_numpy(np.int64)
    result = np.full(len(pairs), np.nan, dtype=np.float32)
    pair_valid = np.zeros(len(pairs), dtype=bool)
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
        local_r = np.full(len(ids), np.nan, dtype=np.float64)
        local_r[nonconstant] = np.sum(left[nonconstant] * right[nonconstant], axis=1) / denominator[nonconstant]
        result[start + ids] = local_r.astype(np.float32)
        pair_valid[start + ids] = np.isfinite(local_r)
    return result, pair_valid


def main() -> None:
    a = args()
    roster = pd.read_csv(a.roster, dtype={"subject_id": str})
    manifest = pd.read_csv(a.manifest, dtype={"subject_id": str})
    if len(roster) != 37 or roster.subject_id.nunique() != 37:
        raise RuntimeError("Expected frozen 37-person young-CN roster")
    joined = roster[["loo_row_index", "subject_id"]].merge(
        manifest, on="subject_id", how="left", validate="one_to_one"
    )
    if joined.clean_profile_lh.isna().any() or joined.clean_profile_rh.isna().any():
        raise RuntimeError("Young roster does not map one-to-one to formal Figure 4 manifest")
    if a.index is not None:
        row = joined.iloc[a.index]
    else:
        selected = joined.loc[joined.subject_id.eq(a.subject_id)]
        if len(selected) != 1:
            raise RuntimeError("Subject ID is not unique in frozen roster")
        row = selected.iloc[0]
    subject = str(row.subject_id)
    out_dir = a.pilot_dir / "features" / "subjects"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{subject}.parquet"
    completion = out_dir / f"{subject}.json"

    base = pd.read_parquet(a.pilot_dir / "anchor_bin_template_counts.parquet")
    all_parts = []
    for hemi in ("lh", "rh"):
        profiles, valid = load_profiles(Path(row[f"clean_profile_{hemi}"]))
        hemi_base = base.loc[base.hemisphere.eq(hemi)].copy()
        anchor_valid_lookup = valid[hemi_base.anchor_vertex.to_numpy(np.int64)]
        for seed in SEEDS:
            pairs = pd.read_parquet(
                a.pilot_dir / "pairs" / f"seed_{seed:02d}_k30_nested.parquet",
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
                ).agg(r_real=("r", "median"), n_valid_pairs=("r", "size")).reset_index()
                current = hemi_base.merge(
                    grouped, on=["node_order", "distance_bin_index"], how="left", validate="one_to_one"
                )
                current["n_valid_pairs"] = current.n_valid_pairs.fillna(0).astype(np.int16)
                current["r_real"] = current.r_real.astype(np.float32)
                current["anchor_valid"] = anchor_valid_lookup
                current["seed"] = np.int8(seed)
                current["k"] = np.int8(k)
                current["n_template_selected"] = current[f"n_template_k{k}"].astype(np.int16)
                all_parts.append(current[[
                    "node_order", "hemisphere", "ico4_node_id", "anchor_vertex",
                    "anchor_dk_label", "anchor_dk_region", "distance_bin_index", "distance_bin",
                    "seed", "k", "n_geometry", "n_template_selected", "anchor_valid",
                    "n_valid_pairs", "r_real",
                ]])

    features = pd.concat(all_parts, ignore_index=True)
    platform = features.loc[features.distance_bin_index.between(10, 14)].groupby(
        ["node_order", "seed", "k"], observed=True, sort=False
    ).agg(
        p_real=("r_real", "median"),
        n_valid_platform_bins=("r_real", "count"),
    ).reset_index()
    platform["platform_complete"] = platform.n_valid_platform_bins.eq(5)
    features = features.merge(platform, on=["node_order", "seed", "k"], validate="many_to_one")
    features["L_real"] = (features.r_real - features.p_real).astype(np.float32)
    features.insert(0, "subject_id", subject)
    expected = 4687 * 15 * 5 * 2
    if len(features) != expected or features.duplicated(
        ["subject_id", "node_order", "distance_bin_index", "seed", "k"]
    ).any():
        raise RuntimeError(f"Feature table has unexpected shape: {len(features)} vs {expected}")
    features.to_parquet(output, index=False, compression="zstd")
    report = {
        "status": "complete", "subject_id": subject, "young_roster_index": int(row.loo_row_index),
        "n_rows": len(features), "n_anchor_invalid_rows": int((~features.anchor_valid).sum()),
        "n_r_missing": int(features.r_real.isna().sum()),
        "n_platform_incomplete_rows": int((~features.platform_complete).sum()),
        "output": str(output.resolve()),
    }
    completion.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "subject_id"}, indent=2))


if __name__ == "__main__":
    main()
