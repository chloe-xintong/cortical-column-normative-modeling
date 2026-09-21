#!/usr/bin/env python3
"""Physical-space column spatial QC with paired and continuous summaries.

This preserves the original nearby-versus-distant definition: 20 nearest
columns versus 20 columns sampled from the farthest 25% within the same
subject, hemisphere, and aparc region.  It additionally reports cleaning
exclusion, paired-unit counts, sampled continuous distance trends, and
within-group Spearman correlations.  It does not change any profile data.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

DEPTHS = 21


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True); b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    result = np.full(len(a), np.nan); ok = denominator > 0
    result[ok] = (a[ok] * b[ok]).sum(1) / denominator[ok]
    return result


def rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def mae(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=1)


def mrtrix_transform(path: Path) -> np.ndarray:
    output = subprocess.run(["mrinfo", str(path), "-transform"], text=True, capture_output=True, check=True).stdout
    rows = [[float(v) for v in line.split()] for line in output.splitlines() if len(line.split()) == 4]
    if len(rows) != 4:
        raise ValueError(f"Cannot parse MRtrix transform: {path}")
    return np.asarray(rows, dtype=float)


def load_mask_count(profile_dir: Path, hemi: str, region: str) -> int:
    path = profile_dir / "bad_trajectory_mask.json"
    if not path.is_file():
        return 0
    with path.open() as f:
        regions = json.load(f).get("regions", {})
    entry = regions.get(f"{hemi}_{region}", {})
    return int(entry.get("bad_count", len(entry.get("bad_indices", []))))


def parse_profile(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("_")
    return parts[0], parts[1], "_".join(parts[2:-1])


def write_csv(path: Path, rows: list[dict], fallback_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else fallback_fields
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def process_group(x: np.ndarray, coords: np.ndarray, *, k: int, continuous_pairs: int, bins: int, rng: np.random.Generator) -> tuple[dict | None, list[dict]]:
    n = len(x)
    if n < 2 * k + 1:
        return None, []
    tree = cKDTree(coords)
    _, neighbors = tree.query(coords, k=k + 1)
    far_pool_size = max(k, int(np.ceil(.25 * (n - 1))))
    near_r = []; far_r = []; near_rmse = []; far_rmse = []; near_mae = []; far_mae = []
    near_d = []; far_d = []
    for i in range(n):
        distance = np.linalg.norm(coords - coords[i], axis=1)
        near = neighbors[i, 1:k + 1]
        candidates = np.flatnonzero(np.arange(n) != i)
        far_pool = candidates[np.argpartition(distance[candidates], -far_pool_size)[-far_pool_size:]]
        far = rng.choice(far_pool, size=k, replace=False)
        anchor_near = np.repeat(x[i:i + 1], k, axis=0)
        anchor_far = np.repeat(x[i:i + 1], k, axis=0)
        near_r.extend(corr(anchor_near, x[near])); far_r.extend(corr(anchor_far, x[far]))
        near_rmse.extend(rmse(anchor_near, x[near])); far_rmse.extend(rmse(anchor_far, x[far]))
        near_mae.extend(mae(anchor_near, x[near])); far_mae.extend(mae(anchor_far, x[far]))
        near_d.extend(np.linalg.norm(coords[near] - coords[i], axis=1)); far_d.extend(np.linalg.norm(coords[far] - coords[i], axis=1))
    near_r = np.asarray(near_r); far_r = np.asarray(far_r)
    near_rmse = np.asarray(near_rmse); far_rmse = np.asarray(far_rmse)
    near_mae = np.asarray(near_mae); far_mae = np.asarray(far_mae)
    near_d = np.asarray(near_d); far_d = np.asarray(far_d)
    m = min(continuous_pairs, n * (n - 1) // 2)
    left = rng.integers(0, n, size=m); right = rng.integers(0, n - 1, size=m); right += right >= left
    distance = np.linalg.norm(coords[left] - coords[right], axis=1)
    pair_corr = corr(x[left], x[right]); pair_rmse = rmse(x[left], x[right]); pair_mae = mae(x[left], x[right])
    rho_corr = spearmanr(distance, pair_corr, nan_policy="omit").statistic
    rho_rmse = spearmanr(distance, pair_rmse, nan_policy="omit").statistic
    rho_mae = spearmanr(distance, pair_mae, nan_policy="omit").statistic
    binned = []
    for bin_id, indices in enumerate(np.array_split(np.argsort(distance), bins), 1):
        if len(indices):
            binned.append({"distance_bin": bin_id, "pairs": len(indices), "distance_mm_median": float(np.median(distance[indices])), "profile_corr_median": float(np.nanmedian(pair_corr[indices])), "profile_rmse_median": float(np.nanmedian(pair_rmse[indices])), "profile_mae_median": float(np.nanmedian(pair_mae[indices]))})
    summary = {
        "n_columns": n, "nearby_pair_count": len(near_r), "distant_pair_count": len(far_r), "continuous_pair_count": m,
        "nearest_mean_distance_mm": float(np.median(near_d)), "distant_mean_distance_mm": float(np.median(far_d)),
        "nearest_profile_corr": float(np.nanmedian(near_r)), "distant_profile_corr": float(np.nanmedian(far_r)), "delta_r": float(np.nanmedian(near_r) - np.nanmedian(far_r)),
        "nearest_profile_rmse": float(np.nanmedian(near_rmse)), "distant_profile_rmse": float(np.nanmedian(far_rmse)), "delta_rmse": float(np.nanmedian(far_rmse) - np.nanmedian(near_rmse)),
        "nearest_profile_mae": float(np.nanmedian(near_mae)), "distant_profile_mae": float(np.nanmedian(far_mae)), "delta_mae": float(np.nanmedian(far_mae) - np.nanmedian(near_mae)),
        "distance_profile_corr_spearman": float(rho_corr), "distance_profile_rmse_spearman": float(rho_rmse), "distance_profile_mae_spearman": float(rho_mae),
    }
    return summary, binned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--dwi-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--continuous-pairs", type=int, default=3000)
    ap.add_argument("--distance-bins", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index.")
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of disjoint file shards.")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("Require 0 <= --shard-index < --num-shards")
    all_files = sorted(args.input_root.glob("D*/profiles/*_md.csv"))
    files = all_files[args.shard_index::args.num_shards]
    if args.max_files:
        files = files[:args.max_files]
    print(f"Shard {args.shard_index + 1}/{args.num_shards}: {len(files)}/{len(all_files)} profile files", flush=True)
    summary_rows: list[dict] = []; bin_rows: list[dict] = []; status_rows: list[dict] = []; affine_cache = {}
    for number, path in enumerate(files, 1):
        subject, hemi, region = parse_profile(path); profile_dir = path.parent
        status = {"subject": subject, "hemisphere": hemi, "region": region, "status": "skipped", "reason": ""}
        try:
            x = np.loadtxt(path, delimiter=",", ndmin=2)
            coordinate_path = args.input_root / subject / "regions" / f"{hemi}_{region}.csv"
            if not coordinate_path.is_file(): raise FileNotFoundError(f"missing coordinate CSV: {coordinate_path}")
            c = np.loadtxt(coordinate_path, delimiter=",", ndmin=2)
            if x.shape[1] != DEPTHS or c.shape[0] not in (3, 4) or c.shape[1] != x.shape[0] * DEPTHS:
                raise ValueError(f"profile/coordinate shape mismatch: profile={x.shape}, coords={c.shape}")
            if subject not in affine_cache: affine_cache[subject] = mrtrix_transform(args.dwi_root / subject / f"{subject}_04_preprocessed.mif")
            voxel = c[:3, ::DEPTHS].T - 1.0
            coords = (affine_cache[subject] @ np.c_[voxel, np.ones(len(voxel))].T).T[:, :3]
            valid = np.isfinite(x).all(axis=1) & np.isfinite(coords).all(axis=1)
            original = len(x) + load_mask_count(profile_dir, hemi, region)
            result, bins = process_group(x[valid], coords[valid], k=args.k, continuous_pairs=args.continuous_pairs, bins=args.distance_bins, rng=np.random.default_rng(args.seed + number))
            status.update(original_columns=original, cleaned_columns=len(x), usable_columns=int(valid.sum()), cleaning_excluded_columns=original-len(x), cleaning_excluded_pct=100.0*(original-len(x))/original if original else np.nan)
            if result is None:
                status["reason"] = f"too_few_usable_columns_for_k={args.k}"
            else:
                status["status"] = "valid"; status["reason"] = ""
                result.update(subject=subject, hemisphere=hemi, region=region, original_columns=original, cleaned_columns=len(x), usable_columns=int(valid.sum()), cleaning_excluded_columns=original-len(x), cleaning_excluded_pct=status["cleaning_excluded_pct"])
                summary_rows.append(result)
                for row in bins: row.update(subject=subject, hemisphere=hemi, region=region)
                bin_rows.extend(bins)
        except Exception as exc:
            status["reason"] = str(exc)
        status_rows.append(status)
        if number % 25 == 0: print(f"Processed {number}/{len(files)} files; valid groups={len(summary_rows)}", flush=True)
    write_csv(args.output_dir / "spatial_similarity_group_summary_v2.csv", summary_rows, ["subject", "hemisphere", "region", "status"])
    write_csv(args.output_dir / "spatial_similarity_distance_bins_v2.csv", bin_rows, ["subject", "hemisphere", "region", "distance_bin"])
    write_csv(args.output_dir / "spatial_similarity_group_status_v2.csv", status_rows, ["subject", "hemisphere", "region", "status", "reason"])
    print(f"Finished {len(files)} files; valid groups={len(summary_rows)}; output={args.output_dir}")


if __name__ == "__main__":
    main()
