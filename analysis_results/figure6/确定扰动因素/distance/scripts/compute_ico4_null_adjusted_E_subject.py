#!/usr/bin/env python3
"""Compute one subject's formal acquisition-null-adjusted ico4 E pilot."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[5]
DISTANCE = REPO / "analysis_results/figure6/确定扰动因素/distance"
F4 = REPO / "analysis_results/figure4/scripts"
sys.path.insert(0, str(F4))
from compute_spatial_null import (  # noqa: E402
    load_sampling_geometry, recolor_bilateral_profiles, sample_noise,
    stable_null_seed,
)
from surface_pca_surrogate import load_real_profiles  # noqa: E402


PAIR_SEEDS = (0, 1, 2)
LOCAL_BINS = range(5)
PLATFORM_BINS = range(10, 15)
NULL_BASE_SEED = 20261020


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-root", type=Path, default=DISTANCE / "null_adjusted_E_stability")
    p.add_argument("--manifest", type=Path, default=REPO /
        "manifests/figure4_coverage95_pass_313_subjects.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int)
    g.add_argument("--subject-id")
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=100_000)
    return p.parse_args()


def pair_matrices(path: Path, bins: range, maximum: int) -> dict[str, np.ndarray]:
    table = pd.read_parquet(path, filters=[
        ("distance_bin_index", ">=", bins.start),
        ("distance_bin_index", "<", bins.stop),
    ], columns=["hemisphere", "node_order", "anchor_vertex", "neighbor_vertex",
                "distance_bin_index", "sample_rank", "n_geometry"])
    n_rows = 4687 * len(bins)
    result: dict[str, np.ndarray] = {}
    for hemi in ("lh", "rh"):
        part = table.loc[table.hemisphere.eq(hemi)]
        row = part.node_order.to_numpy(np.int64) * len(bins) + (
            part.distance_bin_index.to_numpy(np.int64) - bins.start)
        rank = part.sample_rank.to_numpy(np.int64) - 1
        anchor = np.full((n_rows, maximum), -1, dtype=np.int32)
        neighbor = np.full((n_rows, maximum), -1, dtype=np.int32)
        anchor[row, rank] = part.anchor_vertex.to_numpy(np.int32)
        neighbor[row, rank] = part.neighbor_vertex.to_numpy(np.int32)
        result[f"{hemi}_anchor"] = anchor
        result[f"{hemi}_neighbor"] = neighbor
    geometry = np.zeros(n_rows, dtype=np.int32)
    unique = table.drop_duplicates(["node_order", "distance_bin_index"])
    row = unique.node_order.to_numpy(np.int64) * len(bins) + (
        unique.distance_bin_index.to_numpy(np.int64) - bins.start)
    geometry[row] = unique.n_geometry.to_numpy(np.int32)
    result["n_geometry"] = geometry
    result["n_template_selected"] = np.minimum(geometry, maximum).astype(np.int16)
    result["candidate_exhausted"] = geometry <= maximum
    return result


def median_pair_correlation(
    profiles: dict[str, np.ndarray], valid: dict[str, np.ndarray], matrices: dict[str, np.ndarray],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full_like(matrices["lh_anchor"], np.nan, dtype=np.float32)
    for hemi in ("lh", "rh"):
        anchor = matrices[f"{hemi}_anchor"].ravel()
        neighbor = matrices[f"{hemi}_neighbor"].ravel()
        output = values.ravel()
        for start in range(0, len(anchor), chunk_size):
            stop = min(start + chunk_size, len(anchor))
            a, n = anchor[start:stop], neighbor[start:stop]
            okay = (a >= 0) & (n >= 0)
            ids = np.flatnonzero(okay)
            if not len(ids):
                continue
            aa, nn = a[ids], n[ids]
            mask = valid[hemi][aa] & valid[hemi][nn]
            ids = ids[mask]
            if not len(ids):
                continue
            aa, nn = a[ids], n[ids]
            left, right = profiles[hemi][aa].astype(np.float64), profiles[hemi][nn].astype(np.float64)
            left -= left.mean(axis=1, keepdims=True); right -= right.mean(axis=1, keepdims=True)
            denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            nonzero = denominator > 0
            corr = np.full(len(ids), np.nan)
            corr[nonzero] = np.sum(left[nonzero] * right[nonzero], axis=1) / denominator[nonzero]
            output[start + ids] = corr.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=1).astype(np.float32)
    count = np.isfinite(values).sum(axis=1).astype(np.int16)
    return median, count


def metadata_rows(nodes: pd.DataFrame, bins: range) -> pd.DataFrame:
    repeated = nodes.loc[nodes.index.repeat(len(bins)), [
        "node_order", "hemisphere", "ico_node_id", "fsaverage_vertex_id",
        "dk_label_id", "dk_label_name",
    ]].reset_index(drop=True)
    repeated["distance_bin_index"] = np.tile(np.arange(bins.start, bins.stop), len(nodes))
    repeated["distance_bin"] = repeated.distance_bin_index.map(
        lambda b: f"({2*b},{2*(b+1)}]")
    return repeated.rename(columns={
        "fsaverage_vertex_id": "anchor_vertex", "dk_label_id": "anchor_dk_label",
        "dk_label_name": "anchor_dk_region",
    })


def platform_median(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values.reshape(4687, 5), axis=1).astype(np.float32)


def main() -> None:
    a = parse_args(); started = time.time()
    roster = pd.read_csv(a.pilot_root / "pilot_roster.csv", dtype={"subject_id": str})
    manifest = pd.read_csv(a.manifest, dtype={"subject_id": str})
    cohort = roster.merge(manifest, on="subject_id", how="left", validate="one_to_one")
    row = cohort.iloc[a.index] if a.index is not None else cohort.loc[
        cohort.subject_id.eq(a.subject_id)].iloc[0]
    subject = str(row.subject_id)
    out = a.pilot_root / "subjects" / subject; out.mkdir(parents=True, exist_ok=True)
    source = DISTANCE / "local_similarity_stability"
    platform_source = DISTANCE / "platform_sampling_sensitivity"
    nodes = pd.read_csv(source / "ico4_nodes.csv").sort_values("node_order").reset_index(drop=True)
    local_meta = metadata_rows(nodes, LOCAL_BINS)
    platform_meta = metadata_rows(nodes, PLATFORM_BINS)
    matrices = {}
    for seed in PAIR_SEEDS:
        matrices[(seed, "local")] = pair_matrices(
            source / "pairs" / f"seed_{seed:02d}_k30_nested.parquet", LOCAL_BINS, 30)
        matrices[(seed, "platform")] = pair_matrices(
            platform_source / "pairs" / f"seed_{seed:02d}_platform_k300_nested.parquet",
            PLATFORM_BINS, 300)

    md = nib.load(str(row.md_image)); shape = md.shape[:3]
    geometries, real, final_valid = {}, {}, {}
    mask_rows = []
    for hemi in ("lh", "rh"):
        coordinates, geometry_valid, _ = load_sampling_geometry(Path(row[f"clean_profile_{hemi}"]))
        profiles, real_valid = load_real_profiles(Path(row[f"clean_profile_{hemi}"]))
        geometries[hemi] = coordinates; real[hemi] = profiles
        final_valid[hemi] = geometry_valid & real_valid
        mask_rows.append({"subject_id": subject, "hemisphere": hemi,
                          "n_final_valid": int(final_valid[hemi].sum()),
                          "formal_mask_components": "geometry_valid & real_final_valid"})

    # Recompute real under exactly the formal null mask.
    real_local, real_platform = {}, {}
    real_rows = []
    for seed in PAIR_SEEDS:
        lr, ln = median_pair_correlation(real, final_valid, matrices[(seed, "local")], a.chunk_size)
        fr, fn = median_pair_correlation(real, final_valid, matrices[(seed, "platform")], a.chunk_size)
        p = platform_median(fr)
        n_bins = np.isfinite(fr.reshape(4687, 5)).sum(axis=1).astype(np.int8)
        real_local[seed] = (lr, ln); real_platform[seed] = (fr, fn, p, n_bins)
        cur_local = local_meta.copy(); cur_local["pair_seed"] = seed
        cur_local["r_real"] = lr; cur_local["n_valid_real_pairs"] = ln
        cur_local["p_real"] = np.repeat(p, 5); cur_local["n_valid_platform_bins_real"] = np.repeat(n_bins, 5)
        cur_local["platform_complete_real"] = cur_local.n_valid_platform_bins_real.eq(5)
        cur_local["L_real"] = cur_local.r_real - cur_local.p_real
        real_rows.append(cur_local)
    real_table = pd.concat(real_rows, ignore_index=True)
    real_table.insert(0, "subject_id", subject)
    real_table.to_parquet(out / "real_local_features.parquet", index=False, compression="zstd")

    replicate_seeds = [stable_null_seed(NULL_BASE_SEED, subject, r) for r in range(a.replicates)]
    local_outputs, platform_outputs = [], []
    for replicate, replicate_seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(replicate_seed)
        noise = rng.standard_normal(shape, dtype=np.float32)  # Formal FWHM=0: no Gaussian filter.
        null = {hemi: sample_noise(noise, geometries[hemi], final_valid[hemi]) for hemi in ("lh", "rh")}
        null = recolor_bilateral_profiles(null, real, final_valid)
        for seed in PAIR_SEEDS:
            lr, ln = median_pair_correlation(null, final_valid, matrices[(seed, "local")], a.chunk_size)
            fr, fn = median_pair_correlation(null, final_valid, matrices[(seed, "platform")], a.chunk_size)
            p_null = platform_median(fr)
            n_bins_null = np.isfinite(fr.reshape(4687, 5)).sum(axis=1).astype(np.int8)
            r_real, n_real = real_local[seed]
            _, _, p_real, n_bins_real = real_platform[seed]
            local = local_meta.copy(); local["replicate"] = replicate
            local["replicate_seed"] = np.uint64(replicate_seed); local["pair_seed"] = seed
            local["r_real"] = r_real; local["r_null"] = lr
            local["p_real"] = np.repeat(p_real, 5); local["p_null"] = np.repeat(p_null, 5)
            local["L_real"] = local.r_real - local.p_real
            local["L_null"] = local.r_null - local.p_null
            local["E_realization"] = local.L_real - local.L_null
            local["n_valid_real_pairs"] = n_real; local["n_valid_null_pairs"] = ln
            local["n_valid_platform_bins_real"] = np.repeat(n_bins_real, 5)
            local["n_valid_platform_bins_null"] = np.repeat(n_bins_null, 5)
            local["platform_complete_real"] = local.n_valid_platform_bins_real.eq(5)
            local["platform_complete_null"] = local.n_valid_platform_bins_null.eq(5)
            lm = matrices[(seed, "local")]; pm = matrices[(seed, "platform")]
            local["local_n_template_selected"] = lm["n_template_selected"]
            local["local_candidate_exhausted"] = lm["candidate_exhausted"]
            local["platform_any_candidate_exhausted"] = np.repeat(
                pm["candidate_exhausted"].reshape(4687, 5).any(axis=1), 5)
            local_outputs.append(local)
            far = platform_meta.copy(); far["replicate"] = replicate
            far["replicate_seed"] = np.uint64(replicate_seed); far["pair_seed"] = seed
            far["r_real"] = real_platform[seed][0]; far["r_null"] = fr
            far["n_valid_real_pairs"] = real_platform[seed][1]; far["n_valid_null_pairs"] = fn
            far["n_template_selected"] = pm["n_template_selected"]
            far["candidate_exhausted"] = pm["candidate_exhausted"]
            platform_outputs.append(far)

    local_all = pd.concat(local_outputs, ignore_index=True); local_all.insert(0, "subject_id", subject)
    platform_all = pd.concat(platform_outputs, ignore_index=True); platform_all.insert(0, "subject_id", subject)
    local_all.to_parquet(out / "null_adjusted_local_realizations.parquet", index=False, compression="zstd")
    platform_all.to_parquet(out / "null_platform_bin_realizations.parquet", index=False, compression="zstd")
    pd.DataFrame(mask_rows).to_csv(out / "formal_mask_qc.csv", index=False)
    report = {
        "status": "complete", "subject_id": subject, "replicates": a.replicates,
        "pair_seeds": list(PAIR_SEEDS), "null_base_seed": NULL_BASE_SEED,
        "replicate_seeds": replicate_seeds,
        "null_method": "formal_Figure4B_acquisition_sampling",
        "psf_fwhm_mm": 0.0, "laminar_recolored": True,
        "random_field_support": "complete_native_dwi_fov",
        "sampling_coordinates": "real_o3_dwi_voxel_coordinates",
        "interpolation_order": 1, "interpolation_mode": "nearest",
        "same_null_reused_across_pair_seeds": True,
        "runtime_seconds": time.time() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (out / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"subject_id", "replicate_seeds"}}, indent=2))


if __name__ == "__main__":
    main()
